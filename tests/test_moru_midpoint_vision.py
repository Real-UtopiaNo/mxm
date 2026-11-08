from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np

from pet_alliance.vision import TemplateBank, VisualFrame


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "daily_hero"
SIZES = ((849, 477), (1280, 720), (1920, 1080))


def capture(name: str, size=(1920, 1080)) -> VisualFrame:
    path = FIXTURES / f"{name}.jpg"
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    return VisualFrame(cv2.resize(image, size))


class MoruMidpointVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = TemplateBank(ROOT / "assets" / "daily_hero")

    def test_real_three_enemy_exit_is_recognized_at_supported_resolutions(self):
        for size in SIZES:
            with self.subTest(size=size):
                current = capture("west_three_stage_complete", size)
                marker = self.bank.find(current, "moru_midpoint")
                self.assertIsNotNone(marker)
                self.assertGreater(marker.score, .96)
                self.assertIsNotNone(self.bank.find(current, "victory"))
                self.assertIsNotNone(self.bank.find(current, "next_disabled"))
                self.assertIsNotNone(self.bank.find(current, "exit_battle"))
                self.assertIsNone(self.bank.find(current, "next_stage"))

    def test_other_results_and_unknown_screens_do_not_match_midpoint(self):
        for name in ("complete", "victory", "defeat", "battle", "hero",
                     "hero_progress", "normal", "home", "world", "reward"):
            for size in SIZES:
                with self.subTest(capture=name, size=size):
                    self.assertIsNone(self.bank.find(capture(name, size), "moru_midpoint"))
        unknown = VisualFrame(np.zeros((1080, 1920, 3), dtype=np.uint8))
        self.assertIsNone(self.bank.find(unknown, "moru_midpoint"))

    def test_name_level_and_enemy_portrait_are_jointly_required(self):
        original = capture("west_three_stage_complete").image
        # The marker combines the enemy name, its level 3 badge, and portrait;
        # generic result buttons alone cannot identify this intermediate exit.
        for label, box in (("name", (1450, 5, 1580, 45)),
                           ("level", (1710, 7, 1815, 75)),
                           ("portrait", (1800, 7, 1910, 125))):
            with self.subTest(hidden=label):
                image = original.copy()
                left, top, right, bottom = box
                image[top:bottom, left:right] = 0
                self.assertIsNone(self.bank.find(VisualFrame(image), "moru_midpoint"))

    def test_dimmed_marker_is_rejected_by_color_guard(self):
        original = capture("west_three_stage_complete").image
        dimmed = (original.astype(np.float32) * .45).astype(np.uint8)
        self.assertIsNone(self.bank.find(VisualFrame(dimmed), "moru_midpoint"))


if __name__ == "__main__":
    unittest.main()
