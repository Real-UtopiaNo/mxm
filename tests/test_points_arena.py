from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from pet_alliance import AutomationError, Screenshot
from pet_alliance.daily_hero import DailyHero
from pet_alliance.points_arena import PointsArena
from pet_alliance.vision import VisualFrame


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "points_arena"
DAILY_FIXTURES = ROOT / "tests" / "fixtures" / "daily_hero"


def arena_bounds(left, top, right, bottom):
    """Independent hit areas read from the user's original 849 x 477 images."""
    return (left * 1920 / 849, top * 1080 / 477,
            right * 1920 / 849, bottom * 1080 / 477)


BOUNDS = {
    "world_button": (1600, 760, 1905, 960),
    "adventure": (1570, 705, 1710, 785),
    "town": arena_bounds(60, 210, 111, 252),
    "arena": arena_bounds(247, 308, 330, 405),
    "points": arena_bounds(650, 130, 755, 363),
    "join": arena_bounds(306, 288, 390, 327),
    "entry": arena_bounds(313, 321, 398, 355),
    "Hello": arena_bounds(388, 258, 485, 388),
    "Hello_native": (890, 450, 1080, 490),
    "Hell_native": (890, 450, 1080, 490),
    "coco": arena_bounds(600, 117, 698, 248),
    "Hello_moved": arena_bounds(600, 258, 698, 388),
    "roster_close": arena_bounds(691, 53, 723, 85),
    "result": arena_bounds(502, 377, 655, 410),
    "close": arena_bounds(681, 94, 712, 125),
    # Only the blue exit object is interactive, not the red text beneath it.
    "exit": arena_bounds(79, 186, 148, 341),
    "skip": (880, 0, 1040, 145),
    "exit_battle": (1000, 800, 1245, 950),
}

# Independent geometry of the complete cards in the original game capture.
CARD_POSITIONS = [(x, y) for y in (117, 256) for x in (280, 387, 494, 600)]
for slot, (x, y) in enumerate(CARD_POSITIONS, start=1):
    BOUNDS[f"slot_{slot}"] = arena_bounds(x, y, x + 98, y + 132)


def fixture_image(name):
    if name in {"battle", "victory", "complete", "defeat", "region_before"}:
        path = DAILY_FIXTURES / f"{name}.jpg"
    else:
        path = FIXTURES / f"{name}.png"
        if not path.exists():
            path = FIXTURES / f"{name}.jpg"
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AssertionError(f"Cannot decode replay fixture: {path}")
    return cv2.resize(image, (1920, 1080))


def without_target_names():
    image = fixture_image("6")
    for bounds in ((285, 199, 373, 217), (391, 337, 481, 355), (605, 199, 694, 217)):
        left, top, right, bottom = map(round, arena_bounds(*bounds))
        image[top:bottom, left:right] = 0
    return image


def lobby_with_tickets(glyph):
    """Keep the real ticket suffix/UI, replacing only its count with a game glyph."""
    image = cv2.imdecode(np.fromfile(str(FIXTURES / "4.png"), np.uint8), cv2.IMREAD_COLOR)
    image[382:404, 219:236] = (20, 30, 39)
    width = round(glyph.shape[1] / glyph.shape[0] * 16)
    mask = cv2.resize(glyph, (width, 16), interpolation=cv2.INTER_AREA)
    patch = image[385:401, 228 - width // 2:228 - width // 2 + width]
    alpha = mask[:, :, None] / 255
    patch[:] = patch * (1 - alpha) + 255 * alpha
    return cv2.resize(image, (1920, 1080))


def rearranged_roster(cards):
    """Move complete real cards, including portrait/name/margins, into any slot."""
    original = cv2.imdecode(np.fromfile(str(FIXTURES / "6.png"), np.uint8), cv2.IMREAD_COLOR)
    image = original.copy()
    empty = original[256:388, 600:698].copy()
    for x, y in CARD_POSITIONS:
        image[y:y + 132, x:x + 98] = empty
    sources = {"Hell": 0, "Hello": 5, "coco": 3, "foreign": 1}
    for destination, name in cards.items():
        x, y = CARD_POSITIONS[destination]
        source_x, source_y = CARD_POSITIONS[sources[name]]
        image[y:y + 132, x:x + 98] = original[source_y:source_y + 132, source_x:source_x + 98]
    return cv2.resize(image, (1920, 1080))


@dataclass(frozen=True)
class Page:
    image: str
    control: str | None = None


class ReplayClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def wait(self, seconds):
        self.now += seconds


class ArenaReplayContext:
    """Run the real click/observe loop; reject every tap outside its current control."""

    def __init__(self, pages, size=(1920, 1080), **options):
        self.project_root = ROOT
        self.settings = {"daily_hero": {
            "transition_timeout": 2, "poll_interval": .1,
            "min_click_interval": .1, "click_retry_interval": .4, **options,
        }}
        self.log = logging.getLogger("points-arena-replay")
        self.dry_run = False
        self.pages = pages
        self.size = size
        self.index = 0
        self.inputs = []
        self.input_times = []
        self.clock = ReplayClock()
        self.wait = self.clock.wait
        self.overrides = {}
        self.control_overrides = {}
        self.after_miss = {}
        self.after_miss_controls = {}
        self.dropped_clicks = {}
        self.loading_delays = {}
        self.transient_images = {}
        self.entered_at = {0: self.clock.now}
        self.pending = None
        self.fixtures = {page.image: fixture_image(page.image) for page in pages}

    def screenshot(self, image):
        image = cv2.resize(image, self.size)
        return Screenshot(cv2.imencode(".png", image)[1].tobytes(), *self.size)

    def capture(self):
        if self.pending is not None:
            next_index, until = self.pending
            if self.clock.now < until:
                return self.screenshot(np.zeros((1080, 1920, 3), np.uint8))
            self.index = next_index
            self.entered_at[next_index] = self.clock.now
            self.pending = None
        image = self.overrides.get(self.index, self.fixtures[self.pages[self.index].image])
        if self.index in self.transient_images:
            transient, seconds = self.transient_images[self.index]
            if self.clock.now - self.entered_at[self.index] < seconds:
                image = transient
        return self.screenshot(image)

    def tap(self, x, y):
        if self.pending is not None:
            raise AssertionError("A loading screen must not receive input at old coordinates")
        page = self.pages[self.index]
        control = self.control_overrides.get(self.index, page.control)
        if control is None:
            raise AssertionError(f"Unexpected tap on terminal/unknown page {page.image}: {(x, y)}")
        rx, ry = x * 1920 / self.size[0], y * 1080 / self.size[1]
        left, top, right, bottom = BOUNDS[control]
        if not left <= rx <= right or not top <= ry <= bottom:
            raise AssertionError(f"Tap {(rx, ry)} misses {control} on {page.image}")
        self.inputs.append((self.index, control, x, y))
        self.input_times.append(self.clock.now)
        if self.dropped_clicks.get(self.index, 0):
            self.dropped_clicks[self.index] -= 1
            if self.index in self.after_miss:
                self.overrides[self.index] = self.after_miss[self.index]
            if self.index in self.after_miss_controls:
                self.control_overrides[self.index] = self.after_miss_controls[self.index]
            return
        next_index = self.index + 1
        if next_index >= len(self.pages):
            raise AssertionError("Unexpected input after the replay ended")
        if self.index in self.loading_delays:
            self.pending = (next_index, self.clock.now + self.loading_delays[self.index])
        else:
            self.index = next_index
            self.entered_at[next_index] = self.clock.now


class PointsArenaFlowTests(unittest.TestCase):
    def arena(self, directory, pages, *, size=(1920, 1080), **options):
        ctx = ArenaReplayContext(pages, size=size, **options)
        hero = DailyHero(ctx)
        clock_patch = patch("pet_alliance.daily_hero.time.monotonic", ctx.clock.monotonic)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        hero.started = ctx.clock.now
        hero.run_dir.rmdir()
        hero.run_dir = Path(directory) / "run"
        hero.run_dir.mkdir()
        return hero, PointsArena(hero)

    @staticmethod
    def visit_pages(*, no_tickets=False, result_returns_roster=False):
        pages = [Page("1", "town"), Page("2", "arena"), Page("3", "points")]
        if not no_tickets:
            pages += [Page("lobby_ticket_1", "join"), Page("5", "entry")]
            for outcome in ("victory", "defeat", "complete"):
                pages += [Page("6", "Hello"), Page("battle", "skip"), Page(outcome, "exit_battle")]
            pages.append(Page("7", "result"))
            if result_returns_roster:
                pages.append(Page("6", "roster_close"))
        pages += [Page("4", "close"), Page("3", "exit"), Page("2")]
        return pages

    def test_real_images_complete_three_battles_and_both_known_return_paths(self):
        normalized_taps = {}
        for size in ((849, 477), (1920, 1080)):
            for returns_roster in (False, True):
                with self.subTest(size=size, returns_roster=returns_roster), tempfile.TemporaryDirectory() as directory:
                    pages = self.visit_pages(result_returns_roster=returns_roster)
                    hero, arena = self.arena(directory, pages, size=size)
                    arena.run()
                    self.assertEqual(hero.ctx.index, len(pages) - 1)
                    self.assertEqual(hero.report.arena_battles, 3)
                    self.assertEqual(hero.report.arena_no_tickets, 1)
                    controls = [control for _, control, _, _ in hero.ctx.inputs]
                    self.assertEqual(controls, [page.control for page in pages[:-1]])
                    self.assertEqual(controls.count("Hello"), 3)
                    self.assertEqual(controls.count("exit_battle"), 3)
                    self.assertTrue(all(b - a >= .1 - 1e-9
                                        for a, b in zip(hero.ctx.input_times, hero.ctx.input_times[1:])))
                    normalized_taps[size, returns_roster] = [
                        (x * 1920 / size[0], y * 1080 / size[1]) for _, _, x, y in hero.ctx.inputs
                    ]
        for returns_roster in (False, True):
            for small, native in zip(normalized_taps[(849, 477), returns_roster],
                                     normalized_taps[(1920, 1080), returns_roster]):
                self.assertLessEqual(abs(small[0] - native[0]), 3)
                self.assertLessEqual(abs(small[1] - native[1]), 3)

    def test_zero_tickets_closes_without_joining_or_using_a_ticket(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = self.visit_pages(no_tickets=True)
            hero, arena = self.arena(directory, pages)
            arena.run()
            self.assertEqual([control for _, control, _, _ in hero.ctx.inputs],
                             ["town", "arena", "points", "close", "exit"])
            self.assertEqual(hero.report.arena_no_tickets, 1)
            self.assertEqual(hero.report.arena_battles, 0)

    def test_real_clouded_town_confirms_arrival_and_completes_arena_visit(self):
        for size in ((849, 477), (1920, 1080)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                # Actual failed run: clouds cover the old camp-word anchor,
                # while the town scene and Arena entrance are already visible.
                pages = [Page("1", "town"),
                         Page("town_clouds_20261107", "arena"),
                         Page("3", "points"), Page("4", "close"),
                         Page("3", "exit"), Page("town_clouds_20261107")]
                hero, arena = self.arena(directory, pages, size=size)

                arena.run()

                controls = [control for _, control, _, _ in hero.ctx.inputs]
                self.assertEqual(controls, ["town", "arena", "points", "close", "exit"])
                self.assertEqual(controls.count("town"), 1)
                self.assertEqual(hero.ctx.index, len(pages) - 1)
                self.assertEqual(hero.report.arena_no_tickets, 1)
                self.assertEqual(hero.report.arena_battles, 0)
                self.assertIsNotNone(arena.find("town_page", hero.frame))

    def test_return_to_real_clouded_town_confirms_one_town_click(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = [Page("1", "town"), Page("town_clouds_20261107")]
            hero, _ = self.arena(directory, pages)

            hero.return_to_town()

            self.assertEqual([control for _, control, _, _ in hero.ctx.inputs], ["town"])
            self.assertEqual(hero.ctx.index, len(pages) - 1)
            self.assertIsNotNone(hero.town_find("town_page", hero.frame))

    def test_latest_real_clouded_town_confirms_arena_exit_without_extra_navigation(self):
        # Verbatim last_frame.jpg from Daily_Hero_20261108_003424_105381:
        # the arena exit worked, but a cloud crosses the town's red eye.
        for size in ((849, 477), (1920, 1080)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                pages = self.visit_pages(no_tickets=True)
                pages[-1] = Page("town_clouds_20261108")
                hero, arena = self.arena(directory, pages, size=size)

                arena.run()

                controls = [control for _, control, _, _ in hero.ctx.inputs]
                self.assertEqual(controls, ["town", "arena", "points", "close", "exit"])
                self.assertEqual(controls.count("exit"), 1)
                self.assertEqual(controls.count("town"), 1)
                self.assertEqual(hero.ctx.index, len(pages) - 1)
                self.assertIsNotNone(arena.find("town_page", hero.frame))
                self.assertEqual(hero.report.arena_no_tickets, 1)
                self.assertEqual(hero.report.arena_battles, 0)

    def test_arena_exit_waits_40_seconds_for_town_clouds_without_reclicking_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = self.visit_pages(no_tickets=True)
            hero, arena = self.arena(directory, pages, poll_interval=5, transition_timeout=30)
            town_index = len(pages) - 1
            obscured = fixture_image("town_clouds_20261108")
            # Synthetic prolonged cloud coverage of both independent landmarks:
            # retain the real town background, but conceal the dragon building
            # and family building until the clear town frame appears at 40s.
            obscured[110:410, 1050:1390] = 0
            obscured[290:580, 755:1015] = 0
            self.assertIsNone(arena.find("town_page", VisualFrame(obscured)))
            self.assertIsNone(arena.find("arena_page", VisualFrame(obscured)))
            hero.ctx.transient_images[town_index] = (obscured, 40)

            arena.run()

            controls = [control for _, control, _, _ in hero.ctx.inputs]
            self.assertEqual(controls, ["town", "arena", "points", "close", "exit"])
            self.assertEqual(controls.count("exit"), 1)
            self.assertEqual(hero.ctx.index, town_index)
            self.assertGreaterEqual(hero.ctx.clock.now - hero.ctx.entered_at[town_index], 40)
            self.assertLess(hero.ctx.clock.now - hero.ctx.entered_at[town_index], 300)
            self.assertIsNotNone(arena.find("town_page", hero.frame))

    def test_return_to_town_already_in_real_clouded_town_needs_no_input(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, _ = self.arena(directory, [Page("town_clouds_20261107")])

            hero.return_to_town()

            self.assertEqual(hero.ctx.inputs, [])
            self.assertEqual(hero.ctx.index, 0)

    @classmethod
    def repeated_visit_pages(cls, *, result_returns_roster=False):
        single = cls.visit_pages(result_returns_roster=result_returns_roster)
        return single[:3] + single[3:-3] + single[3:-3] + single[-3:]

    def test_two_tickets_repeat_six_battles_before_zero_and_preserve_each_entry_snapshot(self):
        for returns_roster in (False, True):
            with self.subTest(returns_roster=returns_roster), tempfile.TemporaryDirectory() as directory:
                pages = self.repeated_visit_pages(result_returns_roster=returns_roster)
                hero, arena = self.arena(directory, pages)
                lobby_indices = [index for index, page in enumerate(pages) if page.control == "join"]
                first_lobby, second_lobby = lobby_indices
                zero_lobby = len(pages) - 3
                hero.ctx.overrides[first_lobby] = lobby_with_tickets(arena.vision.digits[2])
                self.assertEqual(arena.vision.ticket_status(VisualFrame(hero.ctx.overrides[first_lobby])), 2)
                hero.report.arena_visits = 4
                roster_indices = [index for index, page in enumerate(pages) if page.control == "Hello"]
                # The next entry has an entirely fresh roster, at different slots.
                for index, slot in zip(roster_indices[3:], (7, 2, 4)):
                    hero.ctx.overrides[index] = rearranged_roster({slot: "Hello", (slot + 1) % 8: "foreign"})
                    hero.ctx.control_overrides[index] = f"slot_{slot + 1}"
                observed_tickets = []
                ticket_status = arena.vision.ticket_status

                def read_tickets(frame):
                    count = ticket_status(frame)
                    observed_tickets.append((hero.ctx.index, count))
                    return count

                with patch.object(arena.vision, "ticket_status", side_effect=read_tickets):
                    arena.run()

                self.assertEqual(observed_tickets,
                                 [(first_lobby, 2), (second_lobby, 1), (zero_lobby, 0)])
                self.assertEqual(hero.report.arena_battles, 6)
                self.assertEqual(hero.report.arena_no_tickets, 1)
                self.assertEqual(hero.ctx.index, len(pages) - 1)
                controls = [control for _, control, _, _ in hero.ctx.inputs]
                self.assertEqual(controls.count("join"), 2)
                self.assertEqual(controls.count("entry"), 2)
                self.assertEqual(controls.count("result"), 2)
                self.assertEqual(controls.count("roster_close"), 2 if returns_roster else 0)
                self.assertEqual(controls[-2:], ["close", "exit"])
                self.assertNotIn("close", controls[:-2])
                self.assertNotIn("exit", controls[:-2])
                snapshots = sorted(hero.run_dir.glob("arena_visit_4_entry_*_round_*_select_1.png"))
                self.assertEqual([path.name for path in snapshots], [
                    f"arena_visit_4_entry_{entry}_round_{number}_select_1.png"
                    for entry in (1, 2) for number in (1, 2, 3)
                ])
                for path, index in zip(snapshots, roster_indices):
                    expected = hero.ctx.overrides.get(index, fixture_image("6"))
                    saved = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
                    self.assertTrue(np.array_equal(saved, expected), path.name)

    def test_second_entry_join_retry_relocates_and_rereads_tickets_after_results(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = self.repeated_visit_pages()
            hero, arena = self.arena(directory, pages)
            first_lobby, second_lobby = [index for index, page in enumerate(pages) if page.control == "join"]
            hero.ctx.overrides[first_lobby] = lobby_with_tickets(arena.vision.digits[2])
            lobby = fixture_image("lobby_ticket_1")
            original = arena.find("join", VisualFrame(lobby))
            self.assertIsNotNone(original)
            moved = lobby.copy()
            x, y, width, height = original.x, original.y, original.width, original.height
            moved[y:y + height, x:x + width] = 0
            moved[y:y + height, x + 10:x + width + 10] = lobby[y:y + height, x:x + width]
            self.assertIsNotNone(arena.find("join", VisualFrame(moved)))
            hero.ctx.dropped_clicks[second_lobby] = 1
            hero.ctx.after_miss[second_lobby] = moved

            arena.run()

            join_taps = [(entry, when) for entry, when in zip(hero.ctx.inputs, hero.ctx.input_times)
                         if entry[0] == second_lobby]
            self.assertEqual(len(join_taps), 2)
            self.assertGreaterEqual(join_taps[1][0][2] - join_taps[0][0][2], 7)
            self.assertGreaterEqual(join_taps[1][1] - join_taps[0][1],
                                    hero.options.click_retry_interval - 1e-9)
            self.assertEqual(hero.report.arena_battles, 6)
            self.assertEqual(hero.ctx.index, len(pages) - 1)
            self.assertEqual(hero.report.arena_no_tickets, 1)

    def test_unknown_ticket_count_after_result_stops_before_rejoining_or_returning_to_town(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = self.visit_pages()
            hero, arena = self.arena(directory, pages)
            result_index = next(index for index, page in enumerate(pages) if page.control == "result")
            returned_lobby = result_index + 1
            unknown = fixture_image("4")
            left, top, right, bottom = map(round, arena_bounds(215, 380, 263, 405))
            unknown[top:bottom, left:right] = 0
            hero.ctx.overrides[returned_lobby] = unknown

            with self.assertRaisesRegex(AutomationError, "入场券"):
                arena.run()

            controls = [control for _, control, _, _ in hero.ctx.inputs]
            self.assertEqual(controls.count("join"), 1)
            self.assertEqual(controls.count("entry"), 1)
            self.assertEqual(controls[-1], "result")
            self.assertNotIn("close", controls)
            self.assertNotIn("exit", controls)
            self.assertEqual(hero.ctx.index, returned_lobby)
            self.assertEqual(hero.report.arena_battles, 3)
            self.assertEqual(hero.report.arena_no_tickets, 0)

    def test_arena_return_waits_for_clouded_adventure_before_resuming_local_map(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = [Page("region_before", "world_button"),
                     *self.visit_pages(no_tickets=True)[:-1],
                     Page("2", "adventure"), Page("region_before")]
            hero, _ = self.arena(directory, pages, poll_interval=5)
            town_index = len(pages) - 2
            clouded_town = fixture_image("2")
            # Keep the real town and its other controls visible; hide only the
            # adventure name, as a passing cloud does on the user's screen.
            clouded_town[705:785, 1570:1710] = 0
            self.assertIsNone(hero.find("adventure", VisualFrame(clouded_town)))
            hero.ctx.transient_images[town_index] = (clouded_town, 45)
            region = next(r for r in hero.bank.profile["regions"] if r["key"] == "ring")

            hero.run_points_arena(region)

            self.assertEqual(hero.ctx.index, len(pages) - 1)
            self.assertEqual([control for _, control, _, _ in hero.ctx.inputs],
                             [page.control for page in pages[:-1]])
            adventure_times = [when for (_, control, _, _), when
                               in zip(hero.ctx.inputs, hero.ctx.input_times) if control == "adventure"]
            self.assertEqual(len(adventure_times), 1)
            self.assertGreaterEqual(adventure_times[0] - hero.ctx.entered_at[town_index], 45)
            self.assertEqual(hero.report.arena_visits, 1)
            self.assertEqual(hero.report.arena_no_tickets, 1)
            self.assertTrue(hero.is_map(hero.frame))

    @staticmethod
    def clouded_arena_town():
        image = fixture_image("2")
        left, top, right, bottom = map(round, arena_bounds(258, 317, 327, 403))
        # Obscure only the Arena landmark; retain the real town anchor so the
        # source page remains recognizable while the cloud passes.
        image[top:bottom, left:right] = 0
        return image

    def test_clouded_arena_entry_waits_45_seconds_before_entering(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = self.visit_pages(no_tickets=True)
            hero, arena = self.arena(directory, pages, poll_interval=5)
            clouded_town = self.clouded_arena_town()
            self.assertIsNotNone(arena.find("town_page", VisualFrame(clouded_town)))
            self.assertIsNone(arena.find("arena", VisualFrame(clouded_town)))
            hero.ctx.transient_images[1] = (clouded_town, 45)

            arena.run()

            self.assertEqual([control for _, control, _, _ in hero.ctx.inputs],
                             [page.control for page in pages[:-1]])
            arena_times = [when for (_, control, _, _), when
                           in zip(hero.ctx.inputs, hero.ctx.input_times) if control == "arena"]
            self.assertEqual(len(arena_times), 1)
            self.assertGreaterEqual(arena_times[0] - hero.ctx.entered_at[1], 45)
            self.assertEqual(hero.ctx.index, len(pages) - 1)
            self.assertEqual(hero.report.arena_no_tickets, 1)

    def test_clouded_arena_entry_times_out_after_five_minutes_without_tapping_town(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, arena = self.arena(directory, self.visit_pages(no_tickets=True), poll_interval=5)
            clouded_town = self.clouded_arena_town()
            self.assertIsNotNone(arena.find("town_page", VisualFrame(clouded_town)))
            self.assertIsNone(arena.find("arena", VisualFrame(clouded_town)))
            hero.ctx.overrides[1] = clouded_town

            with self.assertRaisesRegex(AutomationError, "天星镇 → 竞技场"):
                arena.run()

            # The initial world-map town tap is allowed. There must be no
            # guessed arena tap or other input on the clouded town page.
            self.assertEqual([control for _, control, _, _ in hero.ctx.inputs], ["town"])
            self.assertEqual(hero.ctx.index, 1)
            # The preceding town click is confirmed after its first poll; the
            # Arena transition then gets its own complete 300-second budget.
            self.assertEqual(hero.ctx.clock.now - hero.ctx.entered_at[1],
                             300 + hero.options.poll_interval)
            self.assertEqual(hero.report.arena_battles, 0)

    def test_cloud_after_missed_arena_tap_relocates_then_obeys_retry_limit(self):
        for recovers in (True, False):
            with self.subTest(recovers=recovers), tempfile.TemporaryDirectory() as directory:
                pages = self.visit_pages(no_tickets=True)
                hero, arena = self.arena(directory, pages, poll_interval=5, max_click_retries=1)
                town = fixture_image("2")
                original = arena.find("arena", VisualFrame(town))
                self.assertIsNotNone(original)
                moved = town.copy()
                x, y, width, height = original.x, original.y, original.width, original.height
                moved[y:y + height, x:x + width] = 0
                moved[y:y + height, x + 8:x + width + 8] = town[y:y + height, x:x + width]
                relocated = arena.find("arena", VisualFrame(moved))
                self.assertIsNotNone(relocated)
                self.assertGreaterEqual(relocated.x - original.x, 7)
                hero.ctx.dropped_clicks[1] = 1 if recovers else 99
                hero.ctx.after_miss[1] = moved
                tap = hero.ctx.tap

                def tap_with_cloud(x, y):
                    missed_arena = hero.ctx.index == 1 and hero.ctx.dropped_clicks.get(1, 0)
                    tap(x, y)
                    if missed_arena and 1 not in hero.ctx.transient_images:
                        hero.ctx.transient_images[1] = (self.clouded_arena_town(), 45)
                        hero.ctx.entered_at[1] = hero.ctx.clock.now

                with patch.object(hero.ctx, "tap", side_effect=tap_with_cloud):
                    if recovers:
                        arena.run()
                    else:
                        with self.assertRaisesRegex(AutomationError, "补点 1 次"):
                            arena.run()

                arena_taps = [(entry, when) for entry, when
                              in zip(hero.ctx.inputs, hero.ctx.input_times) if entry[1] == "arena"]
                self.assertEqual(len(arena_taps), 2)
                self.assertGreaterEqual(arena_taps[1][1] - arena_taps[0][1], 45)
                self.assertGreaterEqual(arena_taps[1][0][2] - arena_taps[0][0][2], 7)
                if recovers:
                    self.assertEqual(hero.ctx.index, len(pages) - 1)
                    self.assertEqual(hero.report.arena_no_tickets, 1)
                else:
                    self.assertEqual([control for _, control, _, _ in hero.ctx.inputs],
                                     ["town", "arena", "arena"])
                    self.assertEqual(hero.ctx.index, 1)
                    self.assertLess(hero.ctx.clock.now - hero.ctx.entered_at[1], 300)

    def test_every_arena_control_recovers_from_one_miss_and_waits_through_loading(self):
        for minimum, retry in ((.2, .5), (.5, .2)):
            with self.subTest(minimum=minimum, retry=retry), tempfile.TemporaryDirectory() as directory:
                pages = [Page("region_before", "world_button"),
                         *self.visit_pages(result_returns_roster=True)]
                hero, arena = self.arena(
                    directory, pages, min_click_interval=minimum,
                    click_retry_interval=retry, max_click_retries=1,
                )
                hero.ctx.dropped_clicks = {index: 1 for index in range(len(pages) - 1)}
                # Every successful retry leaves an unclickable loading image,
                # longer than either retry interval, before the next page.
                hero.ctx.loading_delays = {index: .7 for index in range(len(pages) - 1)}

                arena.run()

                self.assertEqual(hero.report.arena_battles, 3)
                self.assertEqual(hero.ctx.index, len(pages) - 1)
                self.assertTrue(all(count == 0 for count in hero.ctx.dropped_clicks.values()))
                self.assertEqual([control for _, control, _, _ in hero.ctx.inputs],
                                 [page.control for page in pages[:-1] for _ in range(2)])
                for index in range(len(pages) - 1):
                    times = [when for entry, when in zip(hero.ctx.inputs, hero.ctx.input_times)
                             if entry[0] == index]
                    self.assertEqual(len(times), 2)
                    self.assertGreaterEqual(times[1] - times[0], max(minimum, retry) - 1e-9)
                self.assertTrue(all(b - a >= minimum - 1e-9
                                    for a, b in zip(hero.ctx.input_times, hero.ctx.input_times[1:])))

    def test_second_roster_can_finish_rendering_names_before_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = self.visit_pages()
            hero, arena = self.arena(directory, pages)
            second_roster = [index for index, page in enumerate(pages) if page.image == "6"][1]
            partial = fixture_image("6")
            # Cards and roster anchors have arrived, but the lower halves of
            # all three allowed names have not rendered into their nameplates.
            for bounds in ((285, 206, 373, 217), (391, 345, 481, 355), (605, 206, 694, 217)):
                left, top, right, bottom = map(round, arena_bounds(*bounds))
                partial[top:bottom, left:right] = 0
            self.assertIsNotNone(arena.find("roster", VisualFrame(partial)))
            self.assertIsNone(arena.vision.target(VisualFrame(partial), hero.options.target_players))
            hero.ctx.transient_images[second_roster] = (partial, .8)

            arena.run()

            selected = [(index, when) for (index, control, _, _), when
                        in zip(hero.ctx.inputs, hero.ctx.input_times) if control == "Hello"]
            self.assertEqual(len(selected), 3)
            second_selection = next(when for index, when in selected if index == second_roster)
            self.assertGreaterEqual(second_selection - hero.ctx.entered_at[second_roster], .8 - 1e-9)
            self.assertEqual(hero.report.arena_battles, 3)
            self.assertEqual(hero.ctx.index, len(pages) - 1)

    def test_player_selection_stops_after_configured_click_retry_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, arena = self.arena(directory, [Page("6", "Hello"), Page("battle")],
                                     max_click_retries=2, transition_timeout=4,
                                     min_click_interval=.3, click_retry_interval=.5)
            hero.ctx.dropped_clicks[0] = 99
            with self.assertRaisesRegex(AutomationError, "补点 2 次"):
                arena.fight(1)
            self.assertEqual(len(hero.ctx.inputs), 3)
            self.assertEqual(hero.report.arena_battles, 0)
            self.assertEqual(hero.report.failures, 0)
            self.assertTrue(all(b - a >= .5 - 1e-9
                                for a, b in zip(hero.ctx.input_times, hero.ctx.input_times[1:])))

    def test_capture_failure_is_not_rewritten_as_a_missing_target_error(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, arena = self.arena(directory, [Page("6")])
            hero.ctx.overrides[0] = without_target_names()
            first_frame = hero.ctx.capture()
            with patch.object(hero.ctx, "capture", side_effect=[first_frame, AutomationError("ADB 连接失败")]):
                with self.assertRaisesRegex(AutomationError, "^ADB 连接失败$"):
                    arena.fight(1)
            self.assertEqual(hero.ctx.inputs, [])

    def test_real_failed_second_round_selects_available_hello_then_exits_battle(self):
        with tempfile.TemporaryDirectory() as directory:
            # This is the user's real frame after defeating Hell: Hello and
            # coco remain available in row 1, with a victory badge over Hell.
            pages = [Page("native_after_hell", "Hello_native"), Page("battle", "skip"),
                     Page("victory", "exit_battle"), Page("6")]
            hero, arena = self.arena(directory, pages)

            arena.fight(2)

            self.assertEqual([control for _, control, _, _ in hero.ctx.inputs],
                             ["Hello_native", "skip", "exit_battle"])
            self.assertEqual(hero.report.arena_battles, 1)
            self.assertEqual(hero.ctx.index, len(pages) - 1)

    def test_dynamic_real_third_round_selects_unbeaten_hell_and_returns_results(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = [Page("native_after_hello_coco", "Hell_native"), Page("battle", "skip"),
                     Page("victory", "exit_battle"), Page("7")]
            hero, arena = self.arena(directory, pages)

            arena.fight(3)

            self.assertEqual([control for _, control, _, _ in hero.ctx.inputs],
                             ["Hell_native", "skip", "exit_battle"])
            self.assertEqual(hero.report.arena_battles, 1)
            self.assertEqual(hero.ctx.index, len(pages) - 1)

    def test_dynamic_same_name_can_be_selected_in_every_one_of_eight_card_slots(self):
        for slot in range(8):
            with self.subTest(slot=slot + 1), tempfile.TemporaryDirectory() as directory:
                pages = [Page("6", f"slot_{slot + 1}"), Page("battle", "skip"),
                         Page("victory", "exit_battle"), Page("6")]
                hero, arena = self.arena(directory, pages, target_players=["Hello"])
                hero.ctx.overrides[0] = rearranged_roster({slot: "Hello", (slot + 3) % 8: "foreign"})

                arena.fight(1)

                self.assertEqual([control for _, control, _, _ in hero.ctx.inputs],
                                 [f"slot_{slot + 1}", "skip", "exit_battle"])
                self.assertEqual(hero.report.arena_battles, 1)

    def test_dynamic_new_players_and_reordering_restart_priority_each_round(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = [Page("6", "slot_8"), Page("battle", "skip"), Page("victory", "exit_battle"),
                     Page("6", "slot_3"), Page("battle", "skip"), Page("defeat", "exit_battle"),
                     Page("6", "slot_6"), Page("battle", "skip"), Page("complete", "exit_battle"), Page("7")]
            hero, arena = self.arena(directory, pages)
            hero.ctx.overrides[0] = rearranged_roster({0: "foreign", 7: "Hell"})
            # Hello joins between rounds and takes precedence over Hell/coco,
            # even though a different player appears earlier on the screen.
            hero.ctx.overrides[3] = rearranged_roster({0: "Hell", 2: "Hello", 4: "foreign", 7: "coco"})
            hero.ctx.overrides[6] = rearranged_roster({0: "coco", 2: "foreign", 3: "Hell", 5: "Hello"})
            hero.report.arena_visits = 7

            for number in range(1, 4):
                arena.fight(number)

            self.assertEqual([control for _, control, _, _ in hero.ctx.inputs
                              if control.startswith("slot_")], ["slot_8", "slot_3", "slot_6"])
            self.assertEqual(hero.report.arena_battles, 3)
            self.assertEqual(hero.ctx.index, len(pages) - 1)
            snapshots = sorted(hero.run_dir.glob("arena_visit_7_entry_0_round_*_select_1.png"))
            self.assertEqual([path.name for path in snapshots],
                             [f"arena_visit_7_entry_0_round_{number}_select_1.png" for number in range(1, 4)])
            for path, index in zip(snapshots, (0, 3, 6)):
                self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
                saved = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
                self.assertTrue(np.array_equal(saved, hero.ctx.overrides[index]))

    def test_dynamic_retry_relocates_target_when_its_previous_card_is_a_foreign_player(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = [Page("6", "slot_2"), Page("battle", "skip"), Page("victory", "exit_battle"), Page("6")]
            hero, arena = self.arena(directory, pages, target_players=["Hello"])
            hero.ctx.overrides[0] = rearranged_roster({1: "Hello", 7: "foreign"})
            hero.ctx.dropped_clicks[0] = 1
            hero.ctx.after_miss[0] = rearranged_roster({1: "foreign", 6: "Hello"})
            hero.ctx.after_miss_controls[0] = "slot_7"

            arena.fight(1)

            self.assertEqual([control for _, control, _, _ in hero.ctx.inputs],
                             ["slot_2", "slot_7", "skip", "exit_battle"])
            self.assertGreaterEqual(hero.ctx.input_times[1] - hero.ctx.input_times[0],
                                    hero.options.click_retry_interval - 1e-9)
            self.assertEqual(hero.report.arena_battles, 1)
            snapshots = sorted(hero.run_dir.glob("arena_visit_0_entry_0_round_1_select_*.png"))
            self.assertEqual([path.name for path in snapshots],
                             ["arena_visit_0_entry_0_round_1_select_1.png", "arena_visit_0_entry_0_round_1_select_2.png"])

    def test_dynamic_target_arrival_waits_without_foreign_taps_and_missing_target_times_out(self):
        for arrives in (True, False):
            with self.subTest(arrives=arrives), tempfile.TemporaryDirectory() as directory:
                pages = [Page("6", "slot_8"), Page("battle", "skip"),
                         Page("victory", "exit_battle"), Page("7")]
                hero, arena = self.arena(directory, pages)
                foreign_only = rearranged_roster({0: "foreign", 2: "foreign", 6: "foreign"})
                if arrives:
                    hero.ctx.overrides[0] = rearranged_roster({0: "foreign", 2: "foreign", 7: "Hello"})
                    hero.ctx.transient_images[0] = (foreign_only, .8)
                    arena.fight(3)
                    self.assertEqual([control for _, control, _, _ in hero.ctx.inputs],
                                     ["slot_8", "skip", "exit_battle"])
                    self.assertGreaterEqual(hero.ctx.input_times[0], .8 - 1e-9)
                    self.assertEqual(hero.report.arena_battles, 1)
                else:
                    hero.ctx.overrides[0] = foreign_only
                    with self.assertRaisesRegex(AutomationError, "target_players"):
                        arena.fight(3)
                    self.assertEqual(hero.ctx.inputs, [])
                    self.assertGreaterEqual(hero.ctx.clock.now, hero.options.transition_timeout)
                    missing = hero.run_dir / "arena_visit_0_entry_0_round_3_target_missing.png"
                    self.assertTrue(missing.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_each_round_restarts_the_entire_ordered_target_list_beyond_three_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = self.visit_pages()
            roster_indices = [index for index, page in enumerate(pages) if page.image == "6"]
            for index, player in zip(roster_indices, ("coco", "Hello", "coco")):
                pages[index] = Page("6", player)
            hero, arena = self.arena(directory, pages, target_players=[
                "MissingA", "MissingB", "MissingC", "coco", "Hello", "Hell",
            ])
            no_coco = fixture_image("6")
            left, top, right, bottom = map(round, arena_bounds(605, 199, 694, 217))
            no_coco[top:bottom, left:right] = 0
            hero.ctx.overrides[roster_indices[1]] = no_coco

            arena.run()

            selected = [control for index, control, _, _ in hero.ctx.inputs if index in roster_indices]
            self.assertEqual(selected, ["coco", "Hello", "coco"])
            self.assertEqual(hero.report.arena_battles, 3)
            self.assertEqual(hero.ctx.index, len(pages) - 1)
            self.assertEqual([control for _, control, _, _ in hero.ctx.inputs],
                             [page.control for page in pages[:-1]])

    def test_unknown_ticket_counter_stops_before_join(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = [Page("1", "town"), Page("2", "arena"), Page("3", "points"), Page("4")]
            hero, arena = self.arena(directory, pages)
            unknown = fixture_image("4")
            left, top, right, bottom = map(round, arena_bounds(215, 380, 263, 405))
            unknown[top:bottom, left:right] = 0
            hero.ctx.overrides[3] = unknown
            with self.assertRaisesRegex(AutomationError, "入场券"):
                arena.run()
            self.assertEqual(len(hero.ctx.inputs), 3)
            self.assertEqual(hero.report.arena_battles, 0)

    def test_missing_whitelist_target_stops_without_any_player_tap(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, arena = self.arena(directory, [Page("6")])
            hero.ctx.overrides[0] = without_target_names()
            with self.assertRaisesRegex(AutomationError, "target_players"):
                arena.fight(1)
            self.assertEqual(hero.ctx.inputs, [])
            self.assertGreaterEqual(hero.ctx.clock.now, hero.options.transition_timeout)

    def test_whitelist_is_checked_again_after_each_battle(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = [Page("6", "Hello"), Page("battle", "skip"), Page("defeat", "exit_battle"), Page("6")]
            hero, arena = self.arena(directory, pages)
            hero.ctx.overrides[3] = without_target_names()
            arena.fight(1)
            with self.assertRaisesRegex(AutomationError, "target_players"):
                arena.fight(2)
            self.assertEqual([control for _, control, _, _ in hero.ctx.inputs],
                             ["Hello", "skip", "exit_battle"])
            self.assertEqual(hero.report.arena_battles, 1)

    def test_target_disappearing_after_missed_click_stops_before_reusing_its_position(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, arena = self.arena(directory, [Page("6", "Hello"), Page("battle")])
            hero.ctx.dropped_clicks[0] = 1
            hero.ctx.after_miss[0] = without_target_names()
            with self.assertRaisesRegex(AutomationError, "target_players"):
                arena.fight(1)
            self.assertEqual(len(hero.ctx.inputs), 1)
            self.assertEqual(hero.report.arena_battles, 0)

    def test_loading_after_player_selection_never_receives_a_stale_player_tap(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = [Page("6", "Hello"), Page("battle", "skip"), Page("victory", "exit_battle"), Page("6")]
            hero, arena = self.arena(directory, pages)
            hero.ctx.loading_delays = {0: .8, 1: .6, 2: .6}
            arena.fight(1)
            self.assertEqual([control for _, control, _, _ in hero.ctx.inputs],
                             ["Hello", "skip", "exit_battle"])
            self.assertEqual(hero.report.arena_battles, 1)

    def test_missed_selection_relocates_the_same_player_after_cards_move(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = [Page("6", "Hello"), Page("battle", "skip"), Page("defeat", "exit_battle"), Page("6")]
            hero, arena = self.arena(directory, pages)
            # Move the whole Hello card into the empty fourth slot of row 2.
            original = cv2.imdecode(np.fromfile(str(FIXTURES / "6.png"), np.uint8), cv2.IMREAD_COLOR)
            moved = original.copy()
            moved[257:388, 600:698] = original[257:388, 387:485]
            moved[257:388, 387:485] = 0
            hero.ctx.dropped_clicks[0] = 1
            hero.ctx.after_miss[0] = cv2.resize(moved, (1920, 1080))
            hero.ctx.after_miss_controls[0] = "Hello_moved"
            arena.fight(1)
            self.assertEqual([control for _, control, _, _ in hero.ctx.inputs],
                             ["Hello", "Hello_moved", "skip", "exit_battle"])
            self.assertGreater(hero.ctx.inputs[1][2] - hero.ctx.inputs[0][2], 400)

    def test_arena_result_exits_even_when_its_title_differs_from_dungeon_results(self):
        for title in ("victory", "defeat"):
            with self.subTest(title=title), tempfile.TemporaryDirectory() as directory:
                hero, arena = self.arena(directory, [Page("6", "Hello"), Page(title, "exit_battle"), Page("6")])
                image = fixture_image(title)
                image[0:360, 600:1340] = 0
                self.assertIsNone(hero.find(title, VisualFrame(image)))
                hero.ctx.overrides[1] = image
                arena.fight(1)
                self.assertEqual([control for _, control, _, _ in hero.ctx.inputs], ["Hello", "exit_battle"])
                self.assertEqual(hero.report.arena_battles, 1)

    def test_confirmation_modal_does_not_allow_clicking_its_dimmed_background(self):
        with tempfile.TemporaryDirectory() as directory:
            pages = [Page("1", "town"), Page("2", "arena"), Page("3", "points"), Page("5")]
            hero, arena = self.arena(directory, pages)
            with self.assertRaises(AutomationError):
                arena.run()
            self.assertEqual(len(hero.ctx.inputs), 3)

    def test_wrong_page_is_not_treated_as_a_player_list(self):
        with tempfile.TemporaryDirectory() as directory:
            hero, arena = self.arena(directory, [Page("3")])
            with self.assertRaisesRegex(AutomationError, "玩家列表"):
                arena.fight(1)
            self.assertEqual(hero.ctx.inputs, [])


if __name__ == "__main__":
    unittest.main()
