from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np

from pet_alliance.hero_reset_vision import HeroResetVision
from pet_alliance.vision import VisualFrame


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
SIZES = ((849, 477), (1280, 720), (1374, 772), (1920, 1080))


def capture(name: str, size=(1920, 1080)) -> VisualFrame:
    image = cv2.imdecode(np.fromfile(str(FIXTURES / name), np.uint8), cv2.IMREAD_COLOR)
    return VisualFrame(cv2.resize(image, size))


class HeroClearVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = HeroResetVision(ROOT / "assets" / "daily_hero_reset")

    def test_five_defeated_stages_confirm_completion_before_or_after_reset_used(self):
        for name in ("15", "16"):
            for size in SIZES:
                with self.subTest(capture=name, size=size):
                    current = capture(f"daily_hero_reset/{name}.png", size)
                    self.assertEqual(self.vision.hero_state(current), "cleared")

    def test_unstarted_and_partially_cleared_hero_panels_are_incomplete(self):
        for name in ("hero", "hero_progress"):
            for size in SIZES:
                with self.subTest(capture=name, size=size):
                    current = capture(f"daily_hero/{name}.jpg", size)
                    self.assertEqual(self.vision.hero_state(current), "incomplete")

    def test_every_stage_stamp_is_required_even_when_reset_button_is_visible(self):
        # These independently measured regions cover the five stamps in the
        # supplied captures, without changing the dragon, close or reset UI.
        stamps = ((495, 535, 675, 655), (675, 485, 860, 605),
                  (860, 535, 1045, 655), (1040, 485, 1230, 605),
                  (1230, 535, 1415, 655))
        for name, button in (("15", "reset"), ("16", "reset_disabled")):
            original = capture(f"daily_hero_reset/{name}.png").image
            for stage, (left, top, right, bottom) in enumerate(stamps, start=1):
                with self.subTest(capture=name, missing_stage=stage):
                    image = original.copy()
                    image[top:bottom, left:right] = (60, 75, 85)
                    current = VisualFrame(image)
                    self.assertIsNotNone(self.vision.find(current, button))
                    self.assertIsNone(self.vision.hero_state(current))

    def test_colored_stamp_letters_are_required_not_only_their_shape(self):
        original = capture("daily_hero_reset/15.png").image
        image = original.copy()
        # Desaturate the cards alone; panel controls remain fully visible.
        cards = image[400:750, 480:1425]
        grey = cv2.cvtColor(cards, cv2.COLOR_BGR2GRAY)
        cards[:] = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
        current = VisualFrame(image)
        self.assertIsNotNone(self.vision.find(current, "reset"))
        self.assertIsNone(self.vision.hero_state(current))

    def test_completed_cards_without_an_active_hero_panel_are_unknown(self):
        original = capture("daily_hero_reset/15.png").image
        for control, box in (("hero_dragon", (430, 170, 640, 370)),
                             ("panel_close", (1335, 172, 1495, 330))):
            with self.subTest(hidden_control=control):
                image = original.copy()
                left, top, right, bottom = box
                image[top:bottom, left:right] = 0
                self.assertIsNone(self.vision.hero_state(VisualFrame(image)))
        dimmed = (original.astype(np.float32) * .5).astype(np.uint8)
        self.assertIsNone(self.vision.hero_state(VisualFrame(dimmed)))

    def test_normal_modes_modals_maps_and_battles_never_confirm_completion(self):
        others = (
            "daily_hero/normal.jpg", "daily_hero_reset/failed_west_normal.jpg",
            "daily_hero_reset/reset_confirm.png", "points_arena/5.png",
            "check_arena/19.png", "check_arena/20.png", "daily_hero/home.jpg",
            "daily_hero/world.jpg", "daily_hero/region_after.jpg",
            "daily_hero/battle.jpg", "daily_hero/victory.jpg",
            "daily_hero/complete.jpg", "daily_hero/reward.jpg",
        )
        for name in others:
            for size in ((849, 477), (1920, 1080)):
                with self.subTest(capture=name, size=size):
                    self.assertIsNone(self.vision.hero_state(capture(name, size)))
        self.assertIsNone(self.vision.hero_state(
            VisualFrame(np.zeros((1080, 1920, 3), dtype=np.uint8))))


if __name__ == "__main__":
    unittest.main()
