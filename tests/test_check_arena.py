from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from pet_alliance.adb import AutomationError, Screenshot
from pet_alliance.check_arena import CheckArena, CheckArenaOptions
from pet_alliance.cooperation import TaskInterrupted
from pet_alliance.dragon_schedule import BEIJING


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "check_arena"
NOW = datetime(2026, 11, 5, 23, 30, tzinfo=BEIJING)


@dataclass(frozen=True)
class Page:
    image: str
    control: str | None = None


# Independent hit areas from the supplied 1374 x 772 viewport captures.
BOUNDS = {
    "COCO号": (868, 385, 1026, 433),
    "恶魔城": (602, 360, 757, 424),
    "绝望监狱": (1026, 260, 1199, 312),
    "arena_button": (855, 596, 1031, 686),
    "guardian_close": (993, 146, 1055, 206),
    "dungeon_close": (979, 139, 1052, 213),
    "challenge_guardian": (575, 590, 800, 668),
    "dismiss_claim_success": (640, 675, 735, 715),
    "adventure": (1120, 500, 1225, 565),
    "world_button": (1140, 540, 1366, 690),
    "demon": (555, 510, 677, 579),
    "town": (90, 330, 190, 420),
}


class Clock:
    def __init__(self):
        self.elapsed = 0.0

    def monotonic(self):
        return self.elapsed

    def wait(self, seconds):
        self.elapsed += seconds


class ReplayContext:
    """Replay real captures and reject input outside the current visible control."""

    def __init__(self, pages, *, size=(1374, 772), arena_list=None, dry_run=False):
        self.project_root = ROOT
        self.settings = {
            "daily_hero": {
                "poll_interval": .1, "min_click_interval": .1,
                "click_retry_interval": .4, "transition_timeout": 2,
            },
            "check_arena": {"player_name": "1111"},
        }
        if arena_list is not None:
            self.settings["check_arena"]["arena_list"] = arena_list
        self.log = logging.getLogger("check-arena-replay")
        self.dry_run = dry_run
        self.pages = pages
        self.size = size
        self.index = 0
        self.inputs = []
        self.input_times = []
        self.clock = Clock()
        self.wait = self.clock.wait
        self.dropped = {}
        self.images = {}
        for page in pages:
            if page.image == "unknown":
                image = np.zeros((772, 1374, 3), np.uint8)
            else:
                aliases = {
                    "town": ROOT / "tests/fixtures/points_arena/2.png",
                    "world": ROOT / "tests/fixtures/daily_hero/world.jpg",
                    "local": ROOT / "tests/fixtures/daily_hero/region_before.jpg",
                    "claim_success": FIXTURES / "claim_success.jpg",
                    "guardian_no_duration": FIXTURES / "guardian_no_duration.jpg",
                    "guardian_one_second": FIXTURES / "guardian_one_second.jpg",
                    "demon_daily_fresh": FIXTURES / "demon_daily_fresh.jpg",
                    "coco_self_956": FIXTURES / "coco_self_956.png",
                }
                path = aliases.get(page.image, FIXTURES / f"{page.image}.png")
                image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise AssertionError(f"Cannot decode capture {page.image}")
            image = cv2.resize(image, size)
            self.images[page.image] = Screenshot(cv2.imencode(".png", image)[1].tobytes(), *size)

    def capture(self):
        return self.images[self.pages[self.index].image]

    def tap(self, x, y):
        page = self.pages[self.index]
        if page.control is None:
            raise AssertionError(f"Unexpected input on {page.image}: {(x, y)}")
        rx, ry = x * 1374 / self.size[0], y * 772 / self.size[1]
        left, top, right, bottom = BOUNDS[page.control]
        if not (left <= rx <= right and top <= ry <= bottom):
            raise AssertionError(f"Wrong hit area for {page.control}: {(rx, ry)}")
        self.inputs.append((self.index, page.control, x, y))
        self.input_times.append(self.clock.elapsed)
        if self.dry_run:
            raise AssertionError("Preview sent an input")
        if self.dropped.get(self.index, 0):
            self.dropped[self.index] -= 1
            return
        self.index += 1
        if self.index >= len(self.pages):
            raise AssertionError("Replay advanced past its terminal screen")


def isolate_hero_preparation(checker):
    """Keep guardian replays focused on the final fresh ownership/tap checks.

    Real hero preparation, mode changes and daily proof expiry have separate
    replays in test_arena_hero. These legacy captures start at the guardian
    page, so their prerequisite seam supplies today's proof without adding
    unrelated hero screenshots to every ownership/retry case.
    """
    def prepared(name):
        checker.hero_verified_date = checker.now().date().isoformat()
        return "empty"

    checker.prepare_claim = Mock(side_effect=prepared)


class CheckArenaTests(unittest.TestCase):
    def workflow(self, directory, pages, *, navigate=False, **options):
        ctx = ReplayContext(pages, **options)
        checker = CheckArena(ctx)
        checker.run_dir.rmdir()
        checker.run_dir = Path(directory) / "run"
        checker.run_dir.mkdir()
        clock_patch = patch("pet_alliance.daily_hero.time.monotonic", ctx.clock.monotonic)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        checker.started = 0
        checker.now = lambda: NOW + timedelta(seconds=ctx.clock.elapsed)
        isolate_hero_preparation(checker)
        checker.navigation = []
        if not navigate:
            checker.world = lambda: checker.navigation.append("world")
            checker.enter_region = lambda region: checker.navigation.append(region["name"])

            def return_to_town():
                checker.navigation.append("town")
                checker.observe()

            checker.return_to_town = return_to_town
        return checker

    def self_pages(self):
        return [Page("17", "COCO号"), Page("18", "arena_button"),
                Page("20", "guardian_close"), Page("17")]

    def claim_pages(self, *, guardian="20", notice=True, destination="17"):
        return ([Page("19", "challenge_guardian")]
                + ([Page("claim_success", "dismiss_claim_success")] if notice else [])
                + [Page(guardian, "guardian_close"), Page("18", "dungeon_close"), Page(destination)])

    def test_options_have_ordered_independent_defaults(self):
        first, second = CheckArenaOptions(), CheckArenaOptions()
        self.assertEqual(first.arena_list[:2], ["COCO号", "恶魔城"])
        self.assertEqual(first.player_name, "1111")
        self.assertEqual(len(first.arena_list), 10)
        first.arena_list.pop()
        self.assertEqual(len(second.arena_list), 10)

    def test_options_reject_invalid_dungeon_lists_and_player_names(self):
        for value in ([], "COCO号", [""], [123], ["COCO号", "COCO号"]):
            with self.subTest(arena_list=value), self.assertRaises(AutomationError):
                CheckArenaOptions(arena_list=value)
        for value in ("", "  ", None, 1111):
            with self.subTest(player_name=value), self.assertRaises(AutomationError):
                CheckArenaOptions(player_name=value)

    def test_default_priority_reads_own_coco_guard_and_returns_to_town(self):
        for size in ((1374, 772), (1920, 1080), (849, 477)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                checker = self.workflow(directory, self.self_pages(), size=size)
                with patch.object(checker.time_reader, "read", return_value=1198), \
                        patch("pet_alliance.check_arena.persist_arena_state", return_value=Path(directory) / "automation.local.toml") as save:
                    report = checker.run()
                self.assertEqual(report.status, "completed")
                self.assertEqual(report.dungeon, "COCO号")
                self.assertFalse(report.claimed)
                checker.prepare_claim.assert_not_called()
                self.assertEqual(report.guard_seconds, 1198)
                self.assertTrue(report.returned_to_town)
                self.assertEqual([entry["dungeon"] for entry in report.checked], ["COCO号"])
                self.assertEqual(checker.navigation, ["world", "魔都", "town"])
                self.assertEqual([entry[1] for entry in checker.ctx.inputs], ["COCO号", "arena_button", "guardian_close"])
                self.assertEqual(save.call_count, 1)
                self.assertEqual(save.call_args.args[2], "COCO号")
                summary = json.loads((checker.run_dir / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["arena_expired_time"], report.arena_expired_time)
                self.assertTrue(summary["returned_to_town"])
                self.assertTrue((checker.run_dir / "last_frame.jpg").exists())

    def test_priority_skips_foreign_owner_without_entering_that_dungeon(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, self.self_pages(), arena_list=["恶魔城", "COCO号"])
            with patch.object(checker.time_reader, "read", return_value=1198), \
                    patch("pet_alliance.check_arena.persist_arena_state", return_value=Path(directory) / "local.toml"):
                report = checker.run()
            self.assertEqual([entry["dungeon"] for entry in report.checked], ["恶魔城", "COCO号"])
            self.assertEqual(report.checked[0]["owner"], "other")
            self.assertEqual(checker.ctx.inputs[0][1], "COCO号")

    def test_native_956_self_coco_reads_guard_without_trying_lower_priority_dungeons(self):
        pages = [Page("town", "adventure"), Page("local", "world_button"),
                 Page("world", "demon"), Page("coco_self_956", "COCO号"),
                 Page("18", "arena_button"), Page("20", "guardian_close"),
                 Page("coco_self_956", "world_button"), Page("world", "town"), Page("town")]
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, pages, navigate=True, size=(956, 538),
                                    arena_list=["COCO号", "恶魔城", "绝望监狱"])
            with patch.object(checker.time_reader, "read", return_value=1198) as read, \
                    patch("pet_alliance.check_arena.persist_arena_state",
                          return_value=Path(directory) / "local.toml") as save:
                report = checker.run()
            self.assertEqual(report.status, "completed")
            self.assertEqual(report.dungeon, "COCO号")
            self.assertFalse(report.claimed)
            self.assertEqual(report.guard_seconds, 1198)
            self.assertTrue(report.returned_to_town)
            self.assertEqual(report.checked,
                             [{"dungeon": "COCO号", "owner": "self", "detail_owner": "self"}])
            checker.prepare_claim.assert_not_called()
            read.assert_called_once()
            save.assert_called_once()
            self.assertEqual(save.call_args.args[2], "COCO号")
            self.assertEqual(checker.ctx.index, len(pages) - 1)
            self.assertEqual([entry[1] for entry in checker.ctx.inputs],
                             [page.control for page in pages[:-1]])

    def test_effective_priority_and_foreign_skip_are_logged_before_next_dungeon_tap(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, self.self_pages(), arena_list=["恶魔城", "COCO号"])
            timeline = []
            original_tap = checker.ctx.tap

            def capture_info(message, *args, **kwargs):
                timeline.append(("log", message % args if args else message))

            def capture_tap(x, y):
                timeline.append(("tap", checker.ctx.pages[checker.ctx.index].control))
                original_tap(x, y)

            checker.ctx.tap = capture_tap
            with patch.object(checker.ctx.log, "info", side_effect=capture_info), \
                    patch.object(checker.time_reader, "read", return_value=1198), \
                    patch("pet_alliance.check_arena.persist_arena_state", return_value=Path(directory) / "local.toml"):
                report = checker.run()

            self.assertEqual([entry["dungeon"] for entry in report.checked], ["恶魔城", "COCO号"])
            first_tap = timeline.index(("tap", "COCO号"))
            messages = [message for kind, message in timeline[:first_tap] if kind == "log"]
            expected = ["检查擂主 [1/2]：恶魔城", "恶魔城：擂主为其他玩家，跳过。", "检查擂主 [2/2]：COCO号"]
            for message in expected:
                self.assertIn(message, messages)
            self.assertEqual([message for message in messages if message in expected], expected)
            priority_logs = [index for index, message in enumerate(messages)
                             if "优先级" in message and "恶魔城 → COCO号" in message]
            self.assertTrue(priority_logs, "启动日志应明确打印当前生效优先级：恶魔城 → COCO号")
            self.assertLess(priority_logs[0], messages.index(expected[0]))

    def test_vacant_dungeon_is_claimed_once_and_verified_before_success(self):
        pages = [Page("17", "绝望监狱"), Page("18", "arena_button"),
                 Page("19", "challenge_guardian"), Page("20", "guardian_close"), Page("17_prison_self")]
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, pages, arena_list=["绝望监狱"])
            with patch("pet_alliance.check_arena.persist_arena_state") as save:
                report = checker.run()
            self.assertEqual(report.status, "completed")
            self.assertTrue(report.claimed)
            self.assertEqual(report.dungeon, "绝望监狱")
            self.assertTrue(report.returned_to_town)
            checker.prepare_claim.assert_called_once_with("绝望监狱")
            self.assertEqual(sum(control == "challenge_guardian" for _, control, _, _ in checker.ctx.inputs), 1)
            save.assert_not_called()

    def test_claim_success_notice_is_dismissed_before_verifying_owner(self):
        # Real 2026-11-06 failure: the claim succeeded, but this notice covered
        # the owner field until the old click loop timed out.
        for size in ((849, 477), (1280, 720), (1374, 772), (1920, 1080)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                checker = self.workflow(directory, [
                    Page("19", "challenge_guardian"),
                    Page("claim_success", "dismiss_claim_success"),
                    Page("20", "guardian_close"), Page("18", "dungeon_close"), Page("17"),
                ], size=size)
                self.assertEqual(checker.claim_empty("COCO号"), "self")
                self.assertTrue(checker.report.claimed)
                self.assertEqual([control for _, control, _, _ in checker.ctx.inputs],
                                 ["challenge_guardian", "dismiss_claim_success", "guardian_close", "dungeon_close"])

    def test_claim_and_success_notice_both_retry_only_the_visible_control(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, self.claim_pages())
            checker.ctx.dropped = {0: 1, 1: 1, 2: 1, 3: 1}
            self.assertEqual(checker.claim_empty("COCO号"), "self")
            self.assertTrue(checker.report.claimed)
            self.assertEqual([control for _, control, _, _ in checker.ctx.inputs],
                             ["challenge_guardian"] * 2 + ["dismiss_claim_success"] * 2
                             + ["guardian_close"] * 2 + ["dungeon_close"] * 2)
            for first, second in ((0, 1), (2, 3), (4, 5), (6, 7)):
                self.assertGreaterEqual(
                    checker.ctx.input_times[second] - checker.ctx.input_times[first], .4 - 1e-9)

    def test_success_notice_alone_does_not_confirm_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, [
                Page("19", "challenge_guardian"),
                Page("claim_success", "dismiss_claim_success"), Page("unknown"),
            ])
            with patch("pet_alliance.check_arena.persist_arena_state") as save, \
                    self.assertRaises(AutomationError):
                checker.claim_empty("COCO号")
            self.assertFalse(checker.report.claimed)
            self.assertEqual([control for _, control, _, _ in checker.ctx.inputs],
                             ["challenge_guardian", "dismiss_claim_success"])
            save.assert_not_called()

    def test_other_owner_after_success_notice_is_not_marked_claimed(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, self.claim_pages())
            with patch.object(checker.vision, "map_owner", return_value="other"):
                self.assertEqual(checker.claim_empty("COCO号"), "other")
            self.assertFalse(checker.report.claimed)
            self.assertEqual([control for _, control, _, _ in checker.ctx.inputs],
                             ["challenge_guardian", "dismiss_claim_success", "guardian_close", "dungeon_close"])

    def test_missing_guardian_name_and_time_are_verified_on_the_target_map_flag(self):
        for notice in (True, False):
            for size in ((849, 477), (1280, 720), (1374, 772), (1920, 1080)):
                with self.subTest(notice=notice, size=size), tempfile.TemporaryDirectory() as directory:
                    checker = self.workflow(
                        directory, self.claim_pages(guardian="guardian_no_details", notice=notice), size=size)
                    with patch.object(checker.vision, "map_owner", wraps=checker.vision.map_owner) as verify:
                        self.assertEqual(checker.claim_empty("COCO号"), "self")
                    self.assertTrue(checker.report.claimed)
                    self.assertTrue(checker.report.claimed_at)
                    self.assertEqual(checker.ctx.index, len(checker.ctx.pages) - 1)
                    self.assertEqual({call.args[1] for call in verify.call_args_list}, {"COCO号"})
                    self.assertEqual([item[1] for item in checker.ctx.inputs],
                                     [page.control for page in checker.ctx.pages[:-1]])

    def test_map_confirmation_waits_for_flag_refresh_without_more_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, self.claim_pages(guardian="guardian_no_duration"))
            results = iter((None, "empty", "self"))
            with patch.object(checker.vision, "map_owner", side_effect=lambda *args: next(results)) as verify:
                self.assertEqual(checker.claim_empty("COCO号"), "self")
            self.assertEqual(verify.call_count, 3)
            self.assertTrue(checker.report.claimed)
            self.assertEqual(len(checker.ctx.inputs), 4)

    def test_other_dungeons_own_flag_cannot_confirm_the_target(self):
        with tempfile.TemporaryDirectory() as directory:
            # The supplied map contains our COCO flag, while the target prison
            # is vacant. It must not satisfy the target's post-claim check.
            checker = self.workflow(directory, self.claim_pages())
            with self.assertRaises(AutomationError):
                checker.claim_empty("绝望监狱")
            self.assertFalse(checker.report.claimed)
            self.assertEqual(checker.report.claimed_at, "")
            self.assertEqual(len(checker.ctx.inputs), 4)

    def test_unknown_map_ownership_does_not_confirm_or_repeat_a_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, self.claim_pages())
            with patch.object(checker.vision, "map_owner", return_value=None), \
                    self.assertRaises(AutomationError):
                checker.claim_empty("COCO号")
            self.assertFalse(checker.report.claimed)
            self.assertEqual(len(checker.ctx.inputs), 4)

    def test_pause_before_map_verification_does_not_mark_a_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, self.claim_pages())
            wait = checker.ctx.wait

            def pause_after_close(seconds):
                if checker.ctx.index == 4:
                    raise TaskInterrupted("paused", "用户暂停")
                wait(seconds)

            checker.ctx.wait = pause_after_close
            with self.assertRaises(TaskInterrupted):
                checker.claim_empty("COCO号")
            self.assertFalse(checker.report.claimed)
            self.assertEqual(checker.report.claimed_at, "")
            self.assertEqual(len(checker.ctx.inputs), 4)

    def test_dropped_inputs_are_relocated_and_retried_at_configured_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, self.self_pages())
            checker.ctx.dropped = {0: 1, 1: 1, 2: 1}
            with patch.object(checker.time_reader, "read", return_value=1198), \
                    patch("pet_alliance.check_arena.persist_arena_state", return_value=Path(directory) / "local.toml"):
                report = checker.run()
            self.assertEqual(report.status, "completed")
            self.assertEqual([entry[1] for entry in checker.ctx.inputs],
                             ["COCO号", "COCO号", "arena_button", "arena_button", "guardian_close", "guardian_close"])
            for first, second in ((0, 1), (2, 3), (4, 5)):
                self.assertGreaterEqual(checker.ctx.input_times[second] - checker.ctx.input_times[first], .4 - 1e-9)

    def test_all_foreign_owners_return_to_town_then_raise_and_save_exhaustion(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, [Page("17")], arena_list=["恶魔城"])
            with patch("pet_alliance.check_arena.persist_arena_state") as save, self.assertRaises(AutomationError):
                checker.run()
            self.assertEqual(checker.report.status, "exhausted")
            self.assertTrue(checker.report.returned_to_town)
            self.assertEqual(checker.navigation[-1], "town")
            self.assertEqual(checker.ctx.inputs, [])
            save.assert_not_called()
            summary = json.loads((checker.run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "exhausted")

    def test_unknown_map_owner_never_implies_vacancy_and_sends_no_input(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, [Page("17")])
            with patch.object(checker.vision, "map_owner", return_value=None), \
                    patch("pet_alliance.check_arena.persist_arena_state") as save, self.assertRaises(AutomationError):
                checker.run()
            self.assertEqual(checker.ctx.inputs, [])
            self.assertFalse(checker.report.returned_to_town)
            self.assertNotIn("town", checker.navigation)
            save.assert_not_called()

    def test_unreadable_guard_duration_preserves_config_and_saves_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, self.self_pages())
            with patch.object(checker.time_reader, "read", return_value=None), \
                    patch("pet_alliance.check_arena.persist_arena_state") as save, self.assertRaises(AutomationError):
                checker.run()
            save.assert_not_called()
            self.assertEqual([entry[1] for entry in checker.ctx.inputs], ["COCO号", "arena_button"])
            self.assertFalse(checker.report.returned_to_town)
            self.assertTrue((checker.run_dir / "summary.json").exists())

    def test_vacant_map_is_rechecked_inside_before_any_claim(self):
        pages = [Page("17", "绝望监狱"), Page("18", "arena_button"),
                 Page("19", "guardian_close"), Page("17")]
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, pages, arena_list=["绝望监狱"])
            recognize_owner = checker.vision.guardian_owner

            def occupied_guard(frame, player):
                return "other" if recognize_owner(frame, player) is not None else None

            with patch.object(checker.vision, "guardian_owner", side_effect=occupied_guard), \
                    patch("pet_alliance.check_arena.persist_arena_state") as save, self.assertRaises(AutomationError):
                checker.run()
            self.assertFalse(checker.report.claimed)
            self.assertEqual(checker.report.status, "exhausted")
            self.assertTrue(checker.report.returned_to_town)
            self.assertEqual([entry[1] for entry in checker.ctx.inputs],
                             ["绝望监狱", "arena_button", "guardian_close"])
            checker.prepare_claim.assert_not_called()
            save.assert_not_called()

    def test_unknown_guardian_owner_is_not_claimed_or_written(self):
        pages = [Page("17", "绝望监狱"), Page("18", "arena_button"), Page("19")]
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, pages, arena_list=["绝望监狱"])
            with patch.object(checker.vision, "guardian_owner", return_value=None), \
                    patch("pet_alliance.check_arena.persist_arena_state") as save, self.assertRaises(AutomationError):
                checker.run()
            self.assertFalse(checker.report.claimed)
            self.assertFalse(checker.report.returned_to_town)
            self.assertEqual([entry[1] for entry in checker.ctx.inputs], ["绝望监狱", "arena_button"])
            save.assert_not_called()

    def test_claim_requires_own_name_confirmation_not_just_a_successful_tap(self):
        pages = [Page("17", "绝望监狱"), Page("18", "arena_button"),
                 Page("19", "challenge_guardian"), Page("unknown")]
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, pages, arena_list=["绝望监狱"])
            with patch("pet_alliance.check_arena.persist_arena_state") as save, self.assertRaises(AutomationError):
                checker.run()
            self.assertFalse(checker.report.claimed)
            self.assertFalse(checker.report.returned_to_town)
            self.assertEqual(sum(control == "challenge_guardian" for _, control, _, _ in checker.ctx.inputs), 1)
            save.assert_not_called()

    def test_zero_seconds_is_valid_guard_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, self.self_pages())
            with patch.object(checker.time_reader, "read", return_value=0), \
                    patch("pet_alliance.check_arena.persist_arena_state", return_value=Path(directory) / "local.toml") as save:
                report = checker.run()
            self.assertEqual(report.status, "completed")
            self.assertEqual(report.guard_seconds, 0)
            observed = datetime.fromisoformat(report.observed_at)
            self.assertLessEqual(abs((save.call_args.args[1] - observed - timedelta(hours=6)).total_seconds()), 1)

    def test_claim_button_retries_only_while_guardian_stays_vacant(self):
        pages = [Page("17", "绝望监狱"), Page("18", "arena_button"),
                 Page("19", "challenge_guardian"), Page("20", "guardian_close"), Page("17_prison_self")]
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, pages, arena_list=["绝望监狱"])
            checker.ctx.dropped[2] = 1
            report = checker.run()
            self.assertTrue(report.claimed)
            self.assertTrue(report.returned_to_town)
            attempts = [index for index, entry in enumerate(checker.ctx.inputs) if entry[1] == "challenge_guardian"]
            self.assertEqual(len(attempts), 2)
            self.assertGreaterEqual(checker.ctx.input_times[attempts[1]] - checker.ctx.input_times[attempts[0]], .4 - 1e-9)

    def test_foreign_owner_after_claim_attempt_is_not_clicked_again_or_marked_claimed(self):
        pages = [Page("17", "绝望监狱"), Page("18", "arena_button"),
                 Page("19", "challenge_guardian"), Page("20", "guardian_close"), Page("17")]
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, pages, arena_list=["绝望监狱"])
            recognize_owner = checker.vision.guardian_owner
            map_owner = checker.vision.map_owner

            def foreign_after_attempt(frame, player):
                owner = recognize_owner(frame, player)
                return "other" if owner is not None and checker.ctx.index == 3 else owner

            with patch.object(checker.vision, "guardian_owner", side_effect=foreign_after_attempt), \
                    patch.object(checker.vision, "map_owner", side_effect=lambda f, name, player:
                                 "other" if checker.ctx.index == 4 else map_owner(f, name, player)), \
                    patch("pet_alliance.check_arena.persist_arena_state") as save, self.assertRaises(AutomationError):
                checker.run()
            self.assertEqual(checker.report.status, "exhausted")
            self.assertFalse(checker.report.claimed)
            self.assertTrue(checker.report.returned_to_town)
            self.assertEqual(sum(control == "challenge_guardian" for _, control, _, _ in checker.ctx.inputs), 1)
            save.assert_not_called()

    def test_expiration_uses_time_of_capture_before_slow_ocr(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, self.self_pages())
            read_times = []

            def slow_read(frame):
                read_times.append(checker.now())
                checker.ctx.clock.wait(30)
                return 1198

            with patch.object(checker.time_reader, "read", side_effect=slow_read), \
                    patch("pet_alliance.check_arena.persist_arena_state", return_value=Path(directory) / "local.toml") as save:
                checker.run()
            self.assertEqual(save.call_args.args[1], read_times[0] + timedelta(seconds=21600 - 1198))

    def test_preview_never_navigates_writes_state_or_sends_input(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, [Page("17")], dry_run=True)
            with patch("pet_alliance.check_arena.persist_arena_state") as save:
                report = checker.run()
            self.assertEqual(report.status, "preview")
            self.assertEqual(checker.ctx.inputs, [])
            self.assertEqual(checker.navigation, [])
            self.assertFalse(report.returned_to_town)
            save.assert_not_called()

    def test_real_navigation_starts_in_town_and_returns_after_own_guard(self):
        pages = [Page("town", "adventure"), Page("local", "world_button"),
                 Page("world", "demon"), Page("17", "COCO号"),
                 Page("18", "arena_button"), Page("20", "guardian_close"),
                 Page("17", "world_button"), Page("world", "town"), Page("town")]
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, pages, navigate=True)
            with patch.object(checker.time_reader, "read", return_value=1198), \
                    patch("pet_alliance.check_arena.persist_arena_state", return_value=Path(directory) / "local.toml"):
                report = checker.run()
            self.assertEqual(report.status, "completed")
            self.assertTrue(report.returned_to_town)
            self.assertEqual(checker.ctx.index, len(pages) - 1)
            self.assertEqual([entry[1] for entry in checker.ctx.inputs],
                             [page.control for page in pages[:-1]])

    def test_prepare_map_accepts_daily_refreshed_demon_map_from_real_run(self):
        pages = [Page("town", "adventure"), Page("local", "world_button"),
                 Page("world", "demon"), Page("demon_daily_fresh")]
        with tempfile.TemporaryDirectory() as directory:
            checker = self.workflow(directory, pages, navigate=True)
            checker.prepare_map()
            self.assertEqual(checker.ctx.index, len(pages) - 1)
            self.assertTrue(checker.arena_find("map_anchor", checker.frame))
            self.assertEqual([entry[1] for entry in checker.ctx.inputs],
                             ["adventure", "world_button", "demon"])

    def test_fresh_demon_map_enters_empty_dungeons_in_configured_order(self):
        for priority in (["恶魔城", "COCO号"], ["COCO号", "恶魔城"]):
            with self.subTest(priority=priority), tempfile.TemporaryDirectory() as directory:
                pages = []
                for name in priority:
                    pages.extend([Page("demon_daily_fresh", name), Page("18", "arena_button"),
                                  Page("19", "guardian_close"), Page("18", "dungeon_close")])
                pages.append(Page("demon_daily_fresh"))
                checker = self.workflow(directory, pages, arena_list=priority)
                for name in checker.options.arena_list:
                    frame = checker.observe()
                    self.assertEqual(checker.vision.map_owner(frame, name, "1111"), "empty")
                    self.assertEqual(checker.inspect_dungeon(name), "empty")
                    checker.close_guardian()
                    self.assertTrue(checker.arena_find("map_anchor", checker.frame))
                self.assertEqual(checker.ctx.index, len(pages) - 1)
                self.assertEqual([entry[1] for entry in checker.ctx.inputs],
                                 [page.control for page in pages[:-1]])

    def test_hero_prerequisite_uses_fresh_demon_anchor_and_exact_entrance(self):
        from pet_alliance.arena_hero import ArenaHeroPrerequisite

        for name in ("COCO号", "恶魔城"):
            with self.subTest(dungeon=name), tempfile.TemporaryDirectory() as directory:
                pages = [Page("demon_daily_fresh", name), Page("18")]
                checker = self.workflow(directory, pages)
                helper = ArenaHeroPrerequisite(checker.ctx, now=checker.now)
                helper.run_dir.rmdir()
                helper.run_dir = Path(directory) / "hero"
                helper.run_dir.mkdir()
                helper.started = 0
                helper.dungeon_name = name
                frame = helper.observe()
                target = helper.target(frame)
                self.assertIsNotNone(target)
                helper.open_dungeon_panel(target)
                self.assertEqual(checker.ctx.index, 1)
                self.assertEqual(helper.panel_dungeon, name)
                self.assertEqual([entry[1] for entry in checker.ctx.inputs], [name])


if __name__ == "__main__":
    unittest.main()
