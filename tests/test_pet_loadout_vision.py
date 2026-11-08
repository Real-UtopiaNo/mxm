from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np

from pet_alliance.dragon_vision import DragonVision
from pet_alliance.vision import VisualFrame


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "pet_loadouts"
SIZES = ((1374, 772), (849, 477), (956, 538), (1280, 720), (1920, 1080))
PAGES = {
    "town": {"town_page", "town_pet_button"},
    "pets": {"pet_page", "pet_backup_title", "pet_close", "group_farm",
             "group_dragon", "group_pvp"},
    "preset_farm": {"preset_page", "preset_back", "preset_farm_title", "apply"},
    "confirm": {"confirm_page", "confirm_button", "cancel_button"},
}


def capture(name: str) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(FIXTURES / f"{name}.png"), np.uint8),
                        cv2.IMREAD_COLOR)


def frame(name: str, size=(1920, 1080)) -> VisualFrame:
    return VisualFrame(cv2.resize(capture(name), size))


class PetLoadoutVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = DragonVision(ROOT / "assets" / "pet_loadouts")

    def test_all_original_pages_are_mutually_exclusive_at_five_resolutions(self):
        for size in SIZES:
            for name, expected in PAGES.items():
                with self.subTest(page=name, size=size):
                    current = frame(name, size)
                    actual = {key for key in self.vision.images
                              if self.vision.find(current, key)}
                    self.assertEqual(actual, expected)

    def test_fresh_daily_town_from_failed_run_allows_only_town_pet_controls(self):
        source = capture("town_daily_badge_20261108")
        for size in SIZES:
            with self.subTest(size=size):
                current = VisualFrame(cv2.resize(source, size))
                found = {key for key in self.vision.images if self.vision.find(current, key)}
                self.assertEqual(found, PAGES["town"])

    def test_town_identity_does_not_depend_on_daily_badges_or_family_clouds(self):
        source = capture("town_daily_badge_20261108")
        source[330:472, 810:956] = (40, 45, 95)
        for size in SIZES:
            with self.subTest(size=size):
                current = VisualFrame(cv2.resize(source, size))
                self.assertIsNotNone(self.vision.find(current, "town_page"))
                self.assertIsNotNone(self.vision.find(current, "town_pet_button"))

    def test_latest_clouded_eye_allows_pet_controls_via_independent_town_landmark(self):
        path = ROOT / "tests" / "fixtures" / "points_arena" / "town_clouds_20261108.jpg"
        source = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        for size in SIZES:
            with self.subTest(size=size):
                current = VisualFrame(cv2.resize(source, size))
                found = {key for key in self.vision.images if self.vision.find(current, key)}
                self.assertEqual(found, PAGES["town"])

    def test_visible_pet_icon_cannot_bypass_missing_or_dim_town_identity(self):
        source = capture("town_daily_badge_20261108")
        hidden = source.copy()
        hidden[194:278, 1166:1254] = (40, 45, 95)
        hidden[330:472, 810:956] = (40, 45, 95)
        normalized = cv2.resize(hidden, (960, 540), interpolation=cv2.INTER_AREA)
        self.assertIsNotNone(self.vision._raw(normalized, "town_pet_button"))
        for image in (hidden, *((source * brightness).astype(np.uint8)
                                for brightness in (.85, .8, .65))):
            for size in SIZES:
                with self.subTest(hidden=image is hidden, size=size):
                    current = VisualFrame(cv2.resize(image, size))
                    self.assertIsNone(self.vision.find(current, "town_page"))
                    self.assertIsNone(self.vision.find(current, "town_pet_button"))

    def test_control_centers_land_on_the_named_text_or_icon(self):
        # Independently measured centers in the native device screenshot.
        expected = {
            "town_pet_button": ("town", (706, 63)),
            "pet_close": ("pets", (1839, 54)),
            "group_farm": ("pets", (359, 708)),
            "group_dragon": ("pets", (791, 708)),
            "group_pvp": ("pets", (1225, 708)),
            "preset_back": ("preset_farm", (1632, 586)),
            "apply": ("preset_farm", (1491, 916)),
            "confirm_button": ("confirm", (803, 765)),
            "cancel_button": ("confirm", (1107, 765)),
        }
        for key, (page, point) in expected.items():
            for size in SIZES:
                with self.subTest(key=key, size=size):
                    current = frame(page, size)
                    match = self.vision.find(current, key)
                    self.assertIsNotNone(match)
                    x, y = current.device_point(match.center)
                    self.assertLessEqual(abs(x - round(point[0] * size[0] / 1920)), 4)
                    self.assertLessEqual(abs(y - round(point[1] * size[1] / 1080)), 4)

    def test_dimmed_underlying_pet_controls_cannot_be_clicked(self):
        for page in ("preset_farm", "confirm"):
            for size in SIZES:
                current = frame(page, size)
                for key in ("pet_page", "pet_close", "group_farm", "group_dragon",
                            "group_pvp"):
                    with self.subTest(page=page, size=size, key=key):
                        self.assertIsNone(self.vision.find(current, key))

    def test_confirmation_blocks_the_underlying_apply_and_back_controls(self):
        for size in SIZES:
            current = frame("confirm", size)
            for key in ("preset_page", "preset_back", "apply", "preset_farm_title"):
                with self.subTest(size=size, key=key):
                    self.assertIsNone(self.vision.find(current, key))

    def test_pet_cards_can_change_without_changing_the_page_or_group_names(self):
        source = capture("pets")
        source[100:430, 110:1810] = (64, 71, 78)
        source[470:650, 91:1814] = (104, 96, 88)
        source[849:1080] = (30, 25, 20)
        for size in SIZES:
            current = VisualFrame(cv2.resize(source, size))
            for key in PAGES["pets"]:
                with self.subTest(size=size, key=key):
                    self.assertIsNotNone(self.vision.find(current, key))

    def test_group_names_can_move_to_other_letter_slots(self):
        source = capture("pets")
        farm = source[686:730, 320:400].copy()
        dragon = source[686:730, 750:830].copy()
        source[686:730, 320:400] = dragon
        source[686:730, 750:830] = farm
        for size in SIZES:
            current = VisualFrame(cv2.resize(source, size))
            for key, expected_x in (("group_farm", 789), ("group_dragon", 361)):
                with self.subTest(size=size, key=key):
                    match = self.vision.find(current, key)
                    self.assertIsNotNone(match)
                    x, _ = current.device_point(match.center)
                    self.assertLessEqual(abs(x - round(expected_x * size[0] / 1920)), 4)

    def test_preset_controls_do_not_depend_on_the_selected_group_pet_cards(self):
        source = capture("preset_farm")
        source[650:954, 420:1229] = (34, 49, 65)
        source[620:850, 1250:1600] = (65, 49, 34)
        for size in SIZES:
            current = VisualFrame(cv2.resize(source, size))
            for key in PAGES["preset_farm"]:
                with self.subTest(size=size, key=key):
                    self.assertIsNotNone(self.vision.find(current, key))

    def test_confirm_requires_the_entire_loadout_question_not_just_prompt_title(self):
        source = capture("confirm")
        source[532:588, 715:1209] = (140, 136, 116)
        for size in SIZES:
            current = VisualFrame(cv2.resize(source, size))
            for key in ("confirm_page", "confirm_button", "cancel_button"):
                with self.subTest(size=size, key=key):
                    self.assertIsNone(self.vision.find(current, key))

    def test_isolated_controls_on_an_unrelated_page_require_their_own_guard(self):
        for key, page in (("pet_close", "pets"), ("group_farm", "pets"),
                          ("group_dragon", "pets"), ("group_pvp", "pets"),
                          ("apply", "preset_farm"), ("confirm_button", "confirm"),
                          ("cancel_button", "confirm")):
            source = cv2.resize(capture(page), (960, 540), interpolation=cv2.INTER_AREA)
            unrelated = cv2.resize(capture("town"), (960, 540),
                                   interpolation=cv2.INTER_AREA)
            left, top, right, bottom = self.vision.profile["templates"][key]["source_box"]
            unrelated[top:bottom, left:right] = source[top:bottom, left:right]
            with self.subTest(key=key):
                self.assertIsNone(self.vision.find(VisualFrame(unrelated), key))

    def test_unrelated_game_pages_do_not_match_any_loadout_controls(self):
        paths = list((ROOT / "tests" / "fixtures" / "slay_dragon").glob("*.png"))
        paths += list((ROOT / "tests" / "fixtures" / "check_arena").glob("*.png"))
        for path in paths:
            source = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
            current = VisualFrame(source)
            with self.subTest(page=path.name):
                found = {key for key in self.vision.images
                         if self.vision.find(current, key)}
                expected = {"town_page", "town_pet_button"} if path.name == "12.png" else set()
                self.assertEqual(found, expected)


if __name__ == "__main__":
    unittest.main()
