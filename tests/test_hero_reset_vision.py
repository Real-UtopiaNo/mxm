from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np

from pet_alliance.adb import AutomationError
from pet_alliance.hero_reset_vision import HeroResetVision
from pet_alliance.vision import VisualFrame


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
SIZES = ((849, 477), (1280, 720), (1374, 772), (1920, 1080))


def read(relative: str) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(FIXTURES / relative), np.uint8), cv2.IMREAD_COLOR)


def frame(relative: str, size=(1920, 1080)) -> VisualFrame:
    return VisualFrame(cv2.resize(read(relative), size))


class HeroResetVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = HeroResetVision(ROOT / "assets" / "daily_hero_reset")

    def test_supplied_desktop_crops_fit_the_existing_reference_coordinates(self):
        for name in ("15", "16"):
            image = read(f"daily_hero_reset/{name}.png")
            self.assertEqual(image.shape[:2], (772, 1374))
            self.assertIsNotNone(self.vision.hero_bank.find(VisualFrame(image), "hero_dragon"))
            self.assertIsNotNone(self.vision.hero_bank.find(VisualFrame(image), "panel_close"))

    def test_named_dream_dungeon_matches_full_name_at_five_sizes(self):
        for source in ("region_dream", "dream_missing"):
            for size in (*SIZES, (956, 538)):
                with self.subTest(source=source, size=size):
                    current = frame(f"daily_hero/{source}.jpg", size)
                    found = self.vision.named_entrance(current, "奇幻梦境", "龙萌幻境")
                    self.assertIsNotNone(found)
                    self.assertEqual(found.name, "龙萌幻境")
                    # These independent reference coordinates come from the
                    # visible nameplate, not from an unfinished red badge.
                    self.assertLessEqual(abs(found.center[0] - 1000), 1)
                    self.assertLessEqual(abs(found.center[1] - 734), 1)
                    expected = (round(1000 * size[0] / 1920), round(734 * size[1] / 1080))
                    actual = current.device_point(found.center)
                    self.assertTrue(all(abs(a - b) <= 1 for a, b in zip(actual, expected)))

    def test_named_dream_dungeon_does_not_need_its_red_badge(self):
        image = read("daily_hero/region_dream.jpg")
        # Remove the genuine badge to simulate a completed dungeon. The
        # complete four-character name remains unchanged in this negative.
        image[707:750, 1070:1105] = (30, 90, 65)
        for size in (*SIZES, (956, 538)):
            with self.subTest(size=size):
                current = VisualFrame(cv2.resize(image, size))
                self.assertIsNone(self.vision.hero_bank.find(current, "marker", search=(1068, 705, 1107, 752)))
                found = self.vision.named_entrance(current, "奇幻梦境", "龙萌幻境")
                self.assertIsNotNone(found)
                self.assertLessEqual(abs(found.center[0] - 1000), 1)

    def test_named_dream_dungeon_requires_every_character(self):
        original = read("daily_hero/region_dream.jpg")
        missing_first = original.copy()
        missing_first[716:752, 935:969] = 0
        missing_last = original.copy()
        missing_last[716:752, 1031:1066] = 0
        changed_last = original.copy()
        changed_last[716:752, 1031:1066] = original[716:752, 966:1001]
        for label, image in (("missing_first", missing_first), ("missing_last", missing_last),
                             ("changed_last", changed_last)):
            for size in (*SIZES, (956, 538)):
                with self.subTest(change=label, size=size):
                    self.assertIsNone(self.vision.named_entrance(
                        VisualFrame(cv2.resize(image, size)), "奇幻梦境", "龙萌幻境"))

    def test_named_dungeon_requires_an_active_local_map_even_when_name_is_present(self):
        template = self.vision.bank.images["dungeon_dream_dragon"]
        for source in ("home", "world", "normal", "hero", "battle", "victory", "reward"):
            image = frame(f"daily_hero/{source}.jpg").image.copy()
            image[716:752, 935:1066] = template
            with self.subTest(source=source):
                self.assertIsNotNone(self.vision.bank.find(VisualFrame(image), "dungeon_dream_dragon"))
                self.assertIsNone(self.vision.named_entrance(VisualFrame(image), "奇幻梦境", "龙萌幻境"))
        for source in ("15", "16", "reset_confirm"):
            image = frame(f"daily_hero_reset/{source}.png").image.copy()
            image[716:752, 935:1066] = template
            with self.subTest(source=source):
                self.assertIsNone(self.vision.named_entrance(VisualFrame(image), "奇幻梦境", "龙萌幻境"))
        image = frame("daily_hero/region_dream.jpg").image
        self.assertIsNone(self.vision.named_entrance(
            VisualFrame((image.astype(np.float32) * .5).astype(np.uint8)), "奇幻梦境", "龙萌幻境"))

    def test_named_dungeon_does_not_fall_back_to_another_green_plate(self):
        for source in ("region_before", "region_after", "cursed_missing", "region_otherworld"):
            with self.subTest(source=source):
                current = frame(f"daily_hero/{source}.jpg")
                self.assertTrue(self.vision.entrances(current))
                self.assertIsNone(self.vision.named_entrance(current, "奇幻梦境", "龙萌幻境"))

    def test_named_dungeon_configuration_requires_a_supported_exact_region_and_name(self):
        self.vision.validate_dungeon("奇幻梦境", "龙萌幻境")
        for region, dungeon in (("奇幻梦境", "龙萌"), ("奇幻梦境", "未知副本"),
                                ("魔都", "龙萌幻境"), ("未知区域", "龙萌幻境")):
            with self.subTest(region=region, dungeon=dungeon):
                with self.assertRaisesRegex(AutomationError, "完整名称视觉模板"):
                    self.vision.validate_dungeon(region, dungeon)
                with self.assertRaises(AutomationError):
                    self.vision.named_entrance(frame("daily_hero/region_dream.jpg"), region, dungeon)

    def test_enabled_and_disabled_reset_are_mutually_exclusive_at_all_sizes(self):
        for size in SIZES:
            for name, enabled in (("15", True), ("16", False)):
                with self.subTest(size=size, capture=name):
                    current = frame(f"daily_hero_reset/{name}.png", size)
                    self.assertEqual(self.vision.find(current, "reset") is not None, enabled)
                    self.assertEqual(self.vision.find(current, "reset_disabled") is not None, not enabled)
                    self.assertIsNone(self.vision.find(current, "confirm_reset"))

    def test_reset_requires_the_hero_panel_and_rejects_dimmed_background(self):
        image = frame("daily_hero_reset/15.png").image
        hidden_hero = image.copy()
        hidden_hero[170:370, 430:640] = 0
        self.assertIsNone(self.vision.find(VisualFrame(hidden_hero), "reset"))
        hidden_close = image.copy()
        hidden_close[172:330, 1335:1495] = 0
        self.assertIsNone(self.vision.find(VisualFrame(hidden_close), "reset"))
        dimmed = (image.astype(np.float32) * .5).astype(np.uint8)
        self.assertIsNone(self.vision.find(VisualFrame(dimmed), "reset"))

    def test_actual_reset_confirmation_and_price_at_all_sizes(self):
        for size in SIZES:
            with self.subTest(size=size):
                current = frame("daily_hero_reset/reset_confirm.png", size)
                self.assertIsNotNone(self.vision.find(current, "reset_dialog"))
                self.assertIsNotNone(self.vision.find(current, "confirm_reset"))
                self.assertEqual(self.vision.reset_cost(current), 50)
                self.assertIsNone(self.vision.find(current, "reset"))
                self.assertIsNone(self.vision.find(current, "reset_disabled"))
                self.assertEqual(self.vision.entrances(current), [])

    def test_actual_moru_reset_quote_is_ten_diamonds_at_all_sizes(self):
        # The scheduler stopped on this real quote because only x50 was known.
        # Source: Daily_Hero_Reset_20261106_025714_548654/last_frame.jpg.
        for size in SIZES:
            with self.subTest(size=size):
                current = frame("daily_hero_reset/reset_confirm_10.jpg", size)
                self.assertIsNotNone(self.vision.find(current, "reset_dialog"))
                self.assertIsNotNone(self.vision.find(current, "confirm_reset"))
                self.assertEqual(self.vision.reset_cost(current), 10)

    def test_actual_dream_reset_quote_and_wrapped_question_are_recognized(self):
        # Verbatim last_frame.jpg from Daily_Hero_Reset_20261108_031744_442851.
        # A third price digit pushes the full stop onto the next line and
        # reflows the question; the controls retain their original positions.
        for size in (*SIZES, (956, 538)):
            with self.subTest(size=size):
                current = frame("daily_hero_reset/reset_confirm_300_20261108.jpg", size)
                self.assertIsNotNone(self.vision.find(current, "reset_dialog"))
                button = self.vision.find(current, "confirm_reset")
                self.assertIsNotNone(button)
                self.assertEqual(self.vision.reset_cost(current), 300)
                expected = (round(803 * size[0] / 1920), round(764 * size[1] / 1080))
                actual = current.device_point(button.center)
                self.assertTrue(all(abs(a - b) <= 2 for a, b in zip(actual, expected)))
                self.assertIsNone(self.vision.find(current, "reset"))
                self.assertIsNone(self.vision.find(current, "reset_disabled"))

    def test_wrapped_reset_question_still_requires_title_action_and_entire_body(self):
        source = read("daily_hero_reset/reset_confirm_300_20261108.jpg")
        for label, bounds in (
                ("title", (907, 304, 1017, 359)),
                ("action", (686, 488, 1029, 535)),
                ("question_and_limit", (685, 539, 1206, 585)),
                ("last_line", (685, 585, 900, 635))):
            hidden = source.copy()
            left, top, right, bottom = bounds
            hidden[top:bottom, left:right] = 0
            for size in (*SIZES, (956, 538)):
                with self.subTest(missing=label, size=size):
                    current = VisualFrame(cv2.resize(hidden, size))
                    self.assertIsNone(self.vision.find(current, "reset_dialog"))
                    self.assertIsNone(self.vision.find(current, "confirm_reset"))
                    self.assertIsNone(self.vision.reset_cost(current))

    def test_three_hundred_quote_rejects_hidden_changed_and_additional_digits(self):
        source = read("daily_hero_reset/reset_confirm_300_20261108.jpg")
        hidden = source.copy()
        hidden[491:530, 1099:1189] = source[480, 1100]
        extended = source.copy()
        extended[491:530, 1189:1213] = source[491:530, 1165:1189]
        changed = source.copy()
        changed[491:530, 1165:1189] = source[491:530, 1119:1143]
        for label, image in (("hidden", hidden), ("3000", extended), ("303", changed)):
            for size in (*SIZES, (956, 538)):
                with self.subTest(price=label, size=size):
                    current = VisualFrame(cv2.resize(image, size))
                    self.assertIsNotNone(self.vision.find(current, "reset_dialog"))
                    self.assertIsNone(self.vision.reset_cost(current))

    def test_native_png_quote_matches_without_jpeg_preprocessing(self):
        # Native ADB PNG from the same stopped screen. Saving it as JPEG first
        # smoothed the glyphs and hid this failure in the earlier replay.
        for size in SIZES:
            with self.subTest(size=size):
                current = frame("daily_hero_reset/reset_confirm_10_raw.png", size)
                self.assertIsNotNone(self.vision.find(current, "reset_dialog"))
                self.assertEqual(self.vision.reset_cost(current), 10)

    def test_ten_diamond_quote_rejects_obscured_and_extra_digits(self):
        for source in ("reset_confirm_10.jpg", "reset_confirm_10_raw.png"):
            original = frame(f"daily_hero_reset/{source}").image
            hidden = original.copy()
            hidden[491:530, 1099:1165] = original[480, 1100]
            extended = original.copy()
            extended[491:530, 1165:1188] = original[491:530, 1142:1165]
            prefixed = original.copy()
            prefixed[491:530, 1142:1188] = original[491:530, 1119:1165]
            prefixed[491:530, 1119:1142] = original[491:530, 1142:1165]
            changed = original.copy()
            changed[491:530, 1142:1165] = original[491:530, 1119:1142]
            for label, image in (("obscured", hidden), ("100", extended),
                                 ("010", prefixed), ("11", changed)):
                for size in SIZES:
                    with self.subTest(source=source, price=label, size=size):
                        current = VisualFrame(cv2.resize(image, size))
                        self.assertIsNotNone(self.vision.find(current, "reset_dialog"))
                        self.assertIsNone(self.vision.reset_cost(current))

    def test_arena_prompt_is_not_a_reset_confirmation(self):
        for size in SIZES:
            with self.subTest(size=size):
                current = frame("points_arena/5.png", size)
                self.assertIsNone(self.vision.find(current, "reset_dialog"))
                self.assertIsNone(self.vision.find(current, "confirm_reset"))
                self.assertIsNone(self.vision.reset_cost(current))

    def test_reset_confirmation_requires_title_and_all_reset_text(self):
        current = frame("daily_hero_reset/reset_confirm.png")
        for name in ("reset_dialog", "reset_dialog_action", "reset_dialog_body"):
            with self.subTest(anchor=name):
                anchor = self.vision.bank.find(current, name)
                self.assertIsNotNone(anchor)
                image = current.image.copy()
                image[anchor.y:anchor.y + anchor.height,
                      anchor.x:anchor.x + anchor.width] = 0
                hidden = VisualFrame(image)
                self.assertIsNone(self.vision.find(hidden, "reset_dialog"))
                self.assertIsNone(self.vision.find(hidden, "confirm_reset"))
                self.assertIsNone(self.vision.reset_cost(hidden))

    def test_obscured_or_changed_price_is_unknown_without_losing_reset_prompt(self):
        original = frame("daily_hero_reset/reset_confirm.png").image
        hidden = original.copy()
        hidden[491:530, 1100:1166] = original[480, 1100]
        changed = original.copy()
        # Replace the 5 with the genuine 0 glyph from the same price, giving
        # a different price while retaining its font, colour and background.
        changed[491:530, 1120:1143] = original[491:530, 1143:1166]
        extended = original.copy()
        extended[491:530, 1166:1189] = original[491:530, 1143:1166]
        prefixed = original.copy()
        prefixed[491:530, 1143:1189] = original[491:530, 1120:1166]
        prefixed[491:530, 1120:1143] = original[491:530, 1143:1166]
        for label, image in (("obscured", hidden), ("00", changed),
                             ("500", extended), ("050", prefixed)):
            for size in SIZES:
                with self.subTest(price=label, size=size):
                    current = VisualFrame(cv2.resize(image, size))
                    self.assertIsNotNone(self.vision.find(current, "reset_dialog"))
                    self.assertIsNone(self.vision.reset_cost(current))

    def test_maps_include_completed_dungeons_and_chat_obscured_plates(self):
        expected = {
            "region_before": [(907, 261), (539, 441), (228, 600), (946, 633),
                              (1393, 634), (482, 906), (942, 975), (1257, 977)],
            "region_after": [(907, 261), (539, 441), (228, 600), (946, 633),
                             (1393, 634), (482, 907), (942, 975), (1257, 977)],
            "cursed_missing": [(492, 315), (1614, 340), (1125, 389), (1072, 667),
                               (409, 705), (1490, 880), (496, 954)],
        }
        for name, centers in expected.items():
            for size in SIZES:
                with self.subTest(capture=name, size=size):
                    current = frame(f"daily_hero/{name}.jpg", size)
                    found = self.vision.entrances(current)
                    self.assertEqual(len(found), len(centers))
                    for x, y in centers:
                        self.assertTrue(any(np.hypot(m.center[0] - x, m.center[1] - y) < 12
                                            for m in found), (name, x, y))

    def test_finding_entrances_does_not_require_any_red_exclamation_badge(self):
        for name, count in (("region_before", 8), ("cursed_missing", 7)):
            with self.subTest(capture=name):
                current = frame(f"daily_hero/{name}.jpg")
                image = current.image.copy()
                markers = self.vision.hero_bank.find_all(current, "marker")
                for marker in markers:
                    image[marker.y:marker.y + marker.height, marker.x:marker.x + marker.width] = (35, 80, 35)
                self.assertEqual(len(self.vision.entrances(VisualFrame(image))), count)

    def test_obscured_entrance_is_not_guessed_from_the_landmark_alone(self):
        current = frame("daily_hero/cursed_missing.jpg")
        image = current.image.copy()
        image[910:1000, 370:620] = 0
        self.assertIsNotNone(self.vision.hero_bank.find(VisualFrame(image), "cursed_landmark"))
        found = self.vision.entrances(VisualFrame(image))
        self.assertEqual(len(found), 6)
        self.assertFalse(any(np.hypot(m.center[0] - 496, m.center[1] - 954) < 30 for m in found))

    def test_golden_town_plate_is_excluded_from_map_targets(self):
        current = frame("daily_hero/region_after.jpg")
        image = current.image.copy()
        west_panel = frame("daily_hero_reset/16.png").image
        # Actual golden town plate, pasted into free map space for this test.
        town = west_panel[846:905, 223:404]
        image[440:499, 1480:1661] = town
        self.assertEqual(len(self.vision.entrances(VisualFrame(image))), 8)

    def test_other_pages_and_modal_backgrounds_have_no_map_entrances(self):
        for name in ("home", "world", "normal", "hero", "battle", "victory", "reward"):
            with self.subTest(capture=name):
                self.assertEqual(self.vision.entrances(frame(f"daily_hero/{name}.jpg")), [])
        for name in ("15", "16", "reset_confirm"):
            self.assertEqual(self.vision.entrances(frame(f"daily_hero_reset/{name}.png")), [])


if __name__ == "__main__":
    unittest.main()
