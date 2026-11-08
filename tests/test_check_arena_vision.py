from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np

from pet_alliance.adb import AutomationError
from pet_alliance.check_arena_vision import CheckArenaVision, _parts
from pet_alliance.vision import VisualFrame


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "check_arena"
SIZES = ((1374, 772), (849, 477), (1280, 720), (1920, 1080))


def capture(name):
    path = FIXTURES / (name if Path(name).suffix else f"{name}.png")
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def frame(name, size=(1920, 1080)):
    return VisualFrame(cv2.resize(capture(name), size))


class CheckArenaVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = CheckArenaVision(ROOT / "assets" / "check_arena")

    def test_source_captures_are_full_landscape_viewports(self):
        for name in ("17", "18", "19", "20"):
            self.assertEqual(capture(name).shape[:2], (772, 1374))

    def test_page_guards_at_all_four_supported_sizes(self):
        expected = {
            "17": {"map_anchor"},
            "18": {"dungeon_panel", "dungeon_close", "arena_button"},
            "19": {"guardian_panel", "guardian_close", "challenge_guardian"},
            "20": {"guardian_panel", "guardian_close", "challenge_guardian"},
        }
        names = set.union(*expected.values())
        for size in SIZES:
            for name, wanted in expected.items():
                with self.subTest(size=size, capture=name):
                    current = frame(name, size)
                    found = {key for key in names if self.vision.find(current, key)}
                    self.assertEqual(found, wanted)

    def test_map_ownership_and_all_ten_named_dungeons(self):
        expected = {name: "empty" for name in self.vision.dungeon_names}
        expected.update({"恶魔城": "other", "COCO号": "self"})
        self.assertEqual(len(expected), 10)
        for size in SIZES:
            current = frame("17", size)
            for name, owner in expected.items():
                with self.subTest(size=size, dungeon=name):
                    self.assertIsNotNone(self.vision.dungeon(current, name))
                    self.assertEqual(self.vision.map_owner(current, name, "1111"), owner)

    def test_real_coco_self_flag_survives_small_viewport_rasterization(self):
        for source in ("coco_self_956.png", "17"):
            # Downscaling the already-small attachment a second time can
            # obscure the entrance plaque; test the owner at its native size
            # and larger sizes, plus the old source at all supported sizes.
            sizes = (*SIZES, (956, 538)) if source == "17" else (
                (956, 538), (1280, 720), (1374, 772), (1920, 1080))
            for size in sizes:
                with self.subTest(source=source, size=size):
                    current = frame(source, size)
                    self.assertEqual(self.vision.map_owner(current, "COCO号", "1111"), "self")
                    self.assertEqual(self.vision.map_owner(current, "恶魔城", "1111"), "other")

    def test_fresh_day_map_recognizes_demon_city_and_all_ten_empty_arenas(self):
        # Actual failure after returning from the world map on 2026-11-07.
        # The old Demon Castle plaque anchor included the new orange badge.
        for size in SIZES:
            with self.subTest(size=size):
                current = frame("demon_daily_fresh.jpg", size)
                self.assertIsNotNone(self.vision.find(current, "map_anchor"))
                for name in self.vision.dungeon_names:
                    with self.subTest(dungeon=name):
                        self.assertIsNotNone(self.vision.dungeon(current, name))
                        self.assertEqual(self.vision.map_owner(current, name, "1111"), "empty")

    def test_map_identity_does_not_depend_on_demon_castle_nameplate(self):
        for source in ("17", "demon_daily_fresh.jpg"):
            image = cv2.resize(capture(source), self.vision.source_size)
            image[353:415, 593:758] = 0
            for size in SIZES:
                with self.subTest(source=source, size=size):
                    self.assertIsNotNone(self.vision.find(VisualFrame(cv2.resize(image, size)), "map_anchor"))

    def test_covered_map_anchor_keeps_dungeon_and_owner_unknown(self):
        for source in ("17", "demon_daily_fresh.jpg"):
            image = cv2.resize(capture(source), self.vision.source_size)
            image[108:167, 811:879] = 0
            for size in SIZES:
                with self.subTest(source=source, size=size):
                    current = VisualFrame(cv2.resize(image, size))
                    self.assertIsNone(self.vision.find(current, "map_anchor"))
                    for name in self.vision.dungeon_names:
                        self.assertIsNone(self.vision.dungeon(current, name))
                        self.assertIsNone(self.vision.map_owner(current, name, "1111"))

    def test_guardian_owner_requires_complete_explicit_name_or_both_dashes(self):
        for size in SIZES:
            with self.subTest(size=size):
                self.assertEqual(self.vision.guardian_owner(frame("19", size), "1111"), "empty")
                self.assertEqual(self.vision.guardian_owner(frame("20", size), "1111"), "self")
                self.assertIsNone(self.vision.guardian_owner(frame("17", size), "1111"))
                self.assertIsNone(self.vision.guardian_owner(frame("18", size), "1111"))

    def test_controls_return_actual_reference_coordinates_for_scaling(self):
        expected = {"arena_button": ("18", (940, 638)),
                    "dungeon_close": ("18", (1014, 177)),
                    "guardian_close": ("19", (1022, 173)),
                    "challenge_guardian": ("19", (685, 628))}
        for size in SIZES:
            for name, (page, point) in expected.items():
                with self.subTest(size=size, control=name):
                    current = frame(page, size)
                    match = self.vision.find(current, name)
                    self.assertIsNotNone(match)
                    x, y = current.device_point(match.center)
                    self.assertLessEqual(abs(x - round(point[0] * size[0] / 1374)), 3)
                    self.assertLessEqual(abs(y - round(point[1] * size[1] / 772)), 3)

    def test_dimmed_background_and_unrelated_game_pages_have_no_controls(self):
        names = ("map_anchor", "dungeon_panel", "arena_button", "guardian_panel",
                 "guardian_close", "challenge_guardian")
        for name in ("17", "18", "19", "20"):
            dimmed = VisualFrame((capture(name).astype(float) * .55).astype(np.uint8))
            for control in names:
                with self.subTest(page=name, control=control):
                    self.assertIsNone(self.vision.find(dimmed, control))
        unrelated = cv2.imdecode(np.fromfile(str(ROOT / "tests/fixtures/points_arena/4.png"),
                                           np.uint8), cv2.IMREAD_COLOR)
        current = VisualFrame(unrelated)
        self.assertTrue(all(self.vision.find(current, name) is None for name in names))
        # The common world-map button is not proof that this is Demon City.
        # Other real region maps must not satisfy the Demon City anchor.
        for name in ("region_before", "region_after", "region_dream", "region_otherworld"):
            path = ROOT / "tests/fixtures/daily_hero" / f"{name}.jpg"
            if not path.exists():
                continue
            current = VisualFrame(cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR))
            self.assertIsNone(self.vision.find(current, "map_anchor"))

    def test_unknown_dungeon_or_unregistered_player_cannot_be_used(self):
        with self.assertRaises(AutomationError):
            self.vision.dungeon(frame("17"), "未知副本")
        with self.assertRaises(AutomationError):
            self.vision.validate_player("11111")
        with self.assertRaises(AutomationError):
            self.vision.map_owner(frame("17"), "COCO号", "11111")

    def _changed_name(self, place, count):
        page = capture("17" if place == "map" else "20")
        if place == "map":
            left, top, right, bottom, center, text_top = 917, 425, 980, 449, 947, 429
        else:
            left, top, right, bottom, center, text_top = 548, 225, 738, 266, 590, 235
        patch = page[top:bottom, left:right]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (19, 100, 120), (42, 255, 255))
        background = np.median(patch[mask == 0], axis=0).astype(np.uint8)
        page[top:bottom, left:right] = background
        digit = _parts(self.vision.players["1111"][place])[0]
        width, height = digit.shape[1], digit.shape[0]
        total_width = count * width + (count - 1) * 5
        start = center - total_width // 2
        for index in range(count):
            x = start + index * (width + 5)
            roi = page[text_top:text_top + height, x:x + width]
            roi[digit > 0] = (25, 210, 230)
        return page

    def test_player_name_prefix_or_extra_digit_never_confirms_self(self):
        for place in ("map", "guardian"):
            for count in (3, 5):
                changed = self._changed_name(place, count)
                for size in SIZES:
                    with self.subTest(place=place, count=count, size=size):
                        current = VisualFrame(cv2.resize(changed, size))
                        owner = (self.vision.map_owner(current, "COCO号", "1111")
                                 if place == "map" else self.vision.guardian_owner(current, "1111"))
                        self.assertEqual(owner, "other")

    def test_name_punctuation_or_inserted_space_never_confirms_self(self):
        for place in ("map", "guardian"):
            for change in ("dot", "space"):
                changed = self._changed_name(place, 4)
                if place == "map":
                    center, top, right = 947, 429, 971
                else:
                    center, top, right = 590, 235, 616
                if change == "dot":
                    changed[top + 11:top + 14, right:right + 3] = (25, 210, 230)
                else:
                    # Shift the latter half of the name five source pixels,
                    # preserving all four glyphs but adding a real separator.
                    bottom = top + (15 if place == "map" else 20)
                    segment = changed[top:bottom, center:center + 26].copy()
                    changed[top:bottom, center:center + 26] = changed[top - 1, center]
                    changed[top:bottom, center + 5:center + 31] = segment
                for size in SIZES:
                    with self.subTest(place=place, change=change, size=size):
                        current = VisualFrame(cv2.resize(changed, size))
                        owner = (self.vision.map_owner(current, "COCO号", "1111")
                                 if place == "map" else self.vision.guardian_owner(current, "1111"))
                        self.assertNotEqual(owner, "self")

    def test_covered_owner_does_not_become_empty(self):
        for name, box, method in (
                ("17", (892, 423, 1007, 468), "map"),
                ("20", (548, 225, 738, 266), "guardian")):
            changed = capture(name)
            left, top, right, bottom = box
            changed[top:bottom, left:right] = 0
            current = VisualFrame(changed)
            owner = (self.vision.map_owner(current, "COCO号", "1111")
                     if method == "map" else self.vision.guardian_owner(current, "1111"))
            self.assertIsNone(owner)

    def test_empty_owner_requires_the_duration_dash_too(self):
        changed = capture("19")
        changed[287:319, 550:738] = (72, 91, 110)
        self.assertIsNone(self.vision.guardian_owner(VisualFrame(changed), "1111"))

    def test_modal_covering_a_dungeon_cannot_be_a_map_target(self):
        for page in ("18", "19", "20"):
            for name in self.vision.dungeon_names:
                with self.subTest(page=page, dungeon=name):
                    self.assertIsNone(self.vision.dungeon(frame(page), name))
                    self.assertIsNone(self.vision.map_owner(frame(page), name, "1111"))

    def test_claim_success_message_at_all_four_supported_sizes(self):
        for size in SIZES:
            with self.subTest(size=size):
                current = frame("claim_success.jpg", size)
                match = self.vision.find(current, "claim_success")
                self.assertIsNotNone(match)
                self.assertEqual(match.name, "claim_success")
                # The message covers the owner. Dark underlying controls are
                # not evidence that the modal has already been dismissed.
                self.assertIsNone(self.vision.guardian_owner(current, "1111"))
                self.assertIsNone(self.vision.find(current, "guardian_close"))
                self.assertIsNone(self.vision.find(current, "challenge_guardian"))

    def test_claim_success_requires_both_title_and_complete_body(self):
        source = cv2.resize(capture("claim_success.jpg"), (1374, 772),
                            interpolation=cv2.INTER_AREA)
        for box in ((630, 205, 740, 270), (535, 383, 660, 416),
                    (700, 383, 842, 416)):
            changed = source.copy()
            left, top, right, bottom = box
            changed[top:bottom, left:right] = (124, 135, 140)
            for size in SIZES:
                with self.subTest(box=box, size=size):
                    self.assertIsNone(self.vision.find(
                        VisualFrame(cv2.resize(changed, size)), "claim_success"))

    def test_claim_success_does_not_match_other_prompts_or_pages(self):
        paths = [FIXTURES / f"{name}.png" for name in ("17", "18", "19", "20")]
        paths += [ROOT / "tests/fixtures" / name for name in (
            "points_arena/5.png", "slay_dragon/11.png",
            "slay_dragon/native_not_started.png",
            "daily_hero_reset/reset_confirm.png",
            "daily_hero_reset/reset_confirm_10.jpg")]
        for path in paths:
            source = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
            for size in SIZES:
                with self.subTest(page=path.name, size=size):
                    self.assertIsNone(self.vision.find(
                        VisualFrame(cv2.resize(source, size)), "claim_success"))

    def test_dimmed_claim_success_is_not_an_active_message(self):
        source = (capture("claim_success.jpg").astype(float) * .55).astype(np.uint8)
        for size in SIZES:
            with self.subTest(size=size):
                self.assertIsNone(self.vision.find(
                    VisualFrame(cv2.resize(source, size)), "claim_success"))


if __name__ == "__main__":
    unittest.main()
