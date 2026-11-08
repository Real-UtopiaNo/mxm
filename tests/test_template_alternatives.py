from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from pet_alliance.arena_vision import PointsArenaVision
from pet_alliance.dragon_vision import DragonVision
from pet_alliance.vision import VisualFrame


ROOT = Path(__file__).resolve().parent.parent


class TemplateAlternativesTests(unittest.TestCase):
    """Move real template pixels onto synthetic pages to isolate match policy."""

    def make_fixture(self, kind, *, alternative_changes=None, primary_changes=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name) / "templates"
        dragon = kind == "dragon"
        shutil.copytree(ROOT / "assets" / ("slay_dragon" if dragon else "points_arena"), directory)
        profile = json.loads((directory / "profile.json").read_text(encoding="utf-8"))

        def read(key):
            path = directory / profile["templates"][key]["file"]
            return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)

        primary = read("town_eye" if dragon else "town_page")
        alternative = read("battle_scene" if dragon else "arena")[:27, :31].copy()
        control = read("dragon_entry" if dragon else "join")
        cv2.imencode(".png", primary)[1].tofile(str(directory / "test_primary.png"))
        cv2.imencode(".png", alternative)[1].tofile(str(directory / "test_alternative.png"))
        cv2.imencode(".png", control)[1].tofile(str(directory / "test_control.png"))
        height, width = primary.shape[:2]
        alt_height, alt_width = alternative.shape[:2]
        control_height, control_width = control.shape[:2]
        alternative_entry = {
            "file": "test_alternative.png", "search": [301, 207, 301 + alt_width, 207 + alt_height],
            "threshold": .99, "max_error": 2, "source_capture": "synthetic relocation of real crop",
        }
        alternative_entry.update(alternative_changes or {})
        primary_entry = {
            "file": "test_primary.png", "search": [15, 16, 15 + width, 16 + height],
            "threshold": .97, "max_error": 2, "alternatives": [alternative_entry],
        }
        primary_entry.update(primary_changes or {})
        page, action = ("page", "control") if dragon else ("town_page", "arena")
        profile["templates"] = {
            page: primary_entry,
            action: {"file": "test_control.png", "search": [90, 80, 90 + control_width, 80 + control_height],
                     "threshold": .99, "max_error": 2, "guards": [page]},
        }
        (directory / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
        vision = (DragonVision if dragon else PointsArenaVision)(directory)
        canvas = np.zeros((profile["source_size"][1], profile["source_size"][0], 3), np.uint8)
        canvas[80:80 + control_height, 90:90 + control_width] = control
        return vision, canvas, primary, alternative, page, action

    @staticmethod
    def put(canvas, template, x, y):
        height, width = template.shape[:2]
        canvas[y:y + height, x:x + width] = template

    def test_alternative_has_own_search_and_preserves_original_match_name(self):
        for kind in ("dragon", "points"):
            with self.subTest(kind=kind):
                vision, canvas, _, alternative, page, _ = self.make_fixture(kind)
                self.put(canvas, alternative, 301, 207)
                match = vision._raw(canvas, page)
                self.assertIsNotNone(match)
                self.assertEqual((match.name, match.x, match.y), (page, 301, 207))
                self.assertEqual(len(vision.alternatives[page]), 1)
                self.assertIsInstance(vision.images[page], np.ndarray)
                if kind == "dragon":
                    self.assertEqual(vision.variants[page], [])

    def test_identical_alternative_outside_its_narrow_search_is_rejected(self):
        for kind in ("dragon", "points"):
            with self.subTest(kind=kind):
                vision, canvas, _, alternative, page, _ = self.make_fixture(kind)
                self.put(canvas, alternative, 450, 300)
                self.assertIsNone(vision._raw(canvas, page))

    def test_primary_retains_priority_when_both_landmarks_match(self):
        for kind in ("dragon", "points"):
            with self.subTest(kind=kind):
                vision, canvas, primary, alternative, page, _ = self.make_fixture(kind)
                self.put(canvas, primary, 15, 16)
                self.put(canvas, alternative, 301, 207)
                match = vision._raw(canvas, page)
                self.assertEqual((match.x, match.y), (15, 16))

    def test_alternative_dimming_is_rejected_by_its_own_color_error_limit(self):
        for kind in ("dragon", "points"):
            with self.subTest(kind=kind):
                vision, canvas, _, alternative, page, _ = self.make_fixture(
                    kind, primary_changes={"max_error": 255})
                self.put(canvas, (alternative * .65).astype(np.uint8), 301, 207)
                # Correlation stays high after dimming; only the alternative's
                # independent absolute color error must reject this page.
                self.assertIsNone(vision._raw(canvas, page))
                self.put(canvas, alternative, 301, 207)
                self.assertIsNotNone(vision._raw(canvas, page))

    def test_alternative_missing_fields_inherit_primary_settings(self):
        for kind in ("dragon", "points"):
            with self.subTest(kind=kind):
                vision, canvas, _, alternative, page, _ = self.make_fixture(kind)
                entry = vision.profile["templates"][page]
                entry["search"] = [15, 16, 100, 100]
                entry["threshold"] = 1.01
                entry["max_error"] = 1
                entry["alternatives"] = [{"file": "test_alternative.png"}]
                path = vision.directory / "profile.json"
                path.write_text(json.dumps(vision.profile), encoding="utf-8")
                vision = type(vision)(vision.directory)
                self.put(canvas, alternative, 60, 50)
                self.assertIsNone(vision._raw(canvas, page))
                entry["threshold"] = .97
                path.write_text(json.dumps({**vision.profile, "templates": {
                    **vision.profile["templates"], page: entry}}), encoding="utf-8")
                vision = type(vision)(vision.directory)
                self.assertIsNotNone(vision._raw(canvas, page))
                self.put(canvas, (alternative * .65).astype(np.uint8), 60, 50)
                self.assertIsNone(vision._raw(canvas, page))

    def test_alternative_reference_coordinates_and_existing_control_guards(self):
        for kind in ("dragon", "points"):
            with self.subTest(kind=kind):
                vision, canvas, _, alternative, page, action = self.make_fixture(
                    kind, alternative_changes={"threshold": .94, "max_error": 12})
                self.assertIsNone(vision.find(VisualFrame(canvas, vision.source_size), action))
                self.put(canvas, alternative, 301, 207)
                frame = VisualFrame(canvas)
                match = vision.find(frame, page)
                self.assertIsNotNone(match)
                self.assertEqual((match.name, match.x, match.y), (
                    page, round(301 * 1920 / vision.source_size[0]),
                    round(207 * 1080 / vision.source_size[1])))
                self.assertIsNotNone(vision.find(VisualFrame(canvas, vision.source_size), action))
                self.assertEqual(vision.alternatives[action], [])

    def test_dragon_filename_variants_still_precede_declarative_alternatives(self):
        vision, canvas, _, alternative, page, _ = self.make_fixture("dragon")
        entry = vision.profile["templates"][page]
        entry["search"] = [15, 16, 100, 100]
        entry["variants"] = ["test_alternative.png"]
        (vision.directory / "profile.json").write_text(json.dumps(vision.profile), encoding="utf-8")
        vision = DragonVision(vision.directory)
        self.assertEqual(len(vision.variants[page]), 1)
        self.put(canvas, alternative, 60, 50)
        self.put(canvas, alternative, 301, 207)
        match = vision._raw(canvas, page)
        self.assertEqual((match.x, match.y), (60, 50))


if __name__ == "__main__":
    unittest.main()
