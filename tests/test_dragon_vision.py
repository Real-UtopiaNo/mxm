from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np

from pet_alliance.adb import AutomationError
from pet_alliance.dragon_vision import DragonVision
from pet_alliance.vision import VisualFrame


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "slay_dragon"
SIZES = ((752, 423), (849, 477), (956, 538), (1280, 720),
         (1374, 772), (1920, 1080))


def capture(name: str) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(FIXTURES / f"{name}.png"), np.uint8), cv2.IMREAD_COLOR)


def frame(name: str, size=(1920, 1080)) -> VisualFrame:
    return VisualFrame(cv2.resize(capture(name), size))


class DragonVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = DragonVision(ROOT / "assets" / "slay_dragon")

    def test_all_supplied_pages_at_each_viewport_size(self):
        expected = {
            "8": {"battle_scene", "assistant_page", "stop_hosting", "quest_mark"},
            "9": {"battle_scene", "assistant_page", "start_hosting", "quest_mark"},
            "10": {"login_page", "login_enter"},
            "11": {"notice_page", "notice"},
            "12": {"town", "town_eye", "dragon_entry"},
            "13": {"battle_scene", "dragon_npc"},
            "announcement": {"announcement_page", "announcement_close"},
        }
        names = set.union(*expected.values())
        for size in SIZES:
            for name, wanted in expected.items():
                with self.subTest(capture=name, size=size):
                    found = {key for key in names if self.vision.find(frame(name, size), key)}
                    self.assertEqual(found, wanted)

    def test_original_native_adb_capture_recognizes_stop_without_start(self):
        for size in SIZES:
            with self.subTest(size=size):
                current = frame("native_stop", size)
                for name in ("battle_scene", "assistant_page", "stop_hosting", "quest_mark"):
                    self.assertIsNotNone(self.vision.find(current, name), name)
                for name in ("start_hosting", "town", "dragon_npc", "login_enter", "notice"):
                    self.assertIsNone(self.vision.find(current, name), name)

    def test_native_shifted_camera_preserves_scene_and_control_recognition(self):
        # User manually entered the scene and moved the camera.  The cave
        # pans left while the assistant remains fixed; name and marker show.
        for size in SIZES:
            with self.subTest(size=size):
                current = frame("native_shifted_stop", size)
                for name in ("battle_scene", "assistant_page", "stop_hosting",
                             "dragon_npc", "quest_mark"):
                    self.assertIsNotNone(self.vision.find(current, name), name)
                for name in ("start_hosting", "town", "dragon_entry", "login_enter", "notice"):
                    self.assertIsNone(self.vision.find(current, name), name)
                # Independently measured native nameplate center; no reuse of
                # an earlier frame's center after the scene pans.
                match = self.vision.find(current, "dragon_npc")
                if match is not None:
                    x, y = current.device_point(match.center)
                    self.assertLessEqual(abs(x - round(740 * size[0] / 1920)), 4)
                    self.assertLessEqual(abs(y - round(320 * size[1] / 1080)), 4)

    def test_real_failed_entry_scene_is_recognized_before_approaching_npc(self):
        path = FIXTURES / "native_entry_20261107.jpg"
        source = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        for size in SIZES:
            with self.subTest(size=size):
                current = VisualFrame(cv2.resize(source, size))
                self.assertIsNotNone(self.vision.find(current, "battle_scene"))
                self.assertIsNotNone(self.vision.find(current, "dragon_npc"))
                for name in ("town", "town_eye", "dragon_entry", "login_enter", "notice",
                             "start_hosting", "stop_hosting", "quest_mark"):
                    self.assertIsNone(self.vision.find(current, name), name)

    def test_real_entry_name_cannot_bypass_an_occluded_scene_guard(self):
        path = FIXTURES / "native_entry_20261107.jpg"
        source = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        source = cv2.resize(source, (752, 423), interpolation=cv2.INTER_AREA)
        # This real camera position moves the fixed crystal 15 pixels upward.
        # The full NPC name remains untouched and independently recognizable.
        source[121:167, 226:280] = (40, 45, 95)
        self.assertIsNotNone(self.vision._raw(source, "dragon_npc"))
        for size in SIZES:
            with self.subTest(size=size):
                current = VisualFrame(cv2.resize(source, size))
                self.assertIsNone(self.vision.find(current, "battle_scene"))
                self.assertIsNone(self.vision.find(current, "dragon_npc"))

    def test_real_entry_with_visible_npc_and_mark_confirms_arrival(self):
        path = FIXTURES / "native_entry_mark_20261107.jpg"
        source = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        for size in SIZES:
            with self.subTest(size=size):
                current = VisualFrame(cv2.resize(source, size))
                for name in ("battle_scene", "dragon_npc", "quest_mark"):
                    self.assertIsNotNone(self.vision.find(current, name), name)
                for name in ("town", "town_eye", "dragon_entry", "start_hosting", "stop_hosting"):
                    self.assertIsNone(self.vision.find(current, name), name)

    def test_scene_guard_accepts_either_landmark_but_rejects_both_missing(self):
        boxes = {"crystal": (224, 134, 282, 184), "upper": (380, 36, 454, 80)}
        for page, action in (("13", "dragon_npc"), ("8", "stop_hosting"), ("9", "start_hosting")):
            for missing in (("crystal",), ("upper",), ("crystal", "upper")):
                source = capture(page)
                for key in missing:
                    left, top, right, bottom = boxes[key]
                    source[top:bottom, left:right] = (40, 45, 95)
                for size in SIZES:
                    with self.subTest(page=page, missing=missing, size=size):
                        current = VisualFrame(cv2.resize(source, size))
                        if len(missing) == 1:
                            self.assertIsNotNone(self.vision.find(current, "battle_scene"))
                            self.assertIsNotNone(self.vision.find(current, action))
                        else:
                            self.assertIsNone(self.vision.find(current, "battle_scene"))
                            self.assertIsNone(self.vision.find(current, action))

    def test_match_coordinates_are_in_the_actual_device_viewport(self):
        # Independently measured landmark centers in the original 752px crop.
        # The workflow offsets the dragon entry label to its icon before tapping.
        expected = {"stop_hosting": ("8", (657, 257)),
                    "start_hosting": ("9", (657, 257)),
                    "login_enter": ("10", (371, 350)),
                    "notice": ("11", (376, 395)),
                    "announcement_close": ("announcement", (713, 66)),
                    # The tighter iris guard identifies the page only; the
                    # workflow never clicks this landmark.
                    "town_eye": ("12", (474, 92)),
                    "dragon_entry": ("12", (475, 146)),
                    "dragon_npc": ("13", (338, 125)),
                    "quest_mark": ("8", (338, 92))}
        for name, (page, point) in expected.items():
            for size in SIZES:
                with self.subTest(name=name, size=size):
                    current = frame(page, size)
                    match = self.vision.find(current, name)
                    self.assertIsNotNone(match)
                    x, y = current.device_point(match.center)
                    self.assertLessEqual(abs(x - round(point[0] * size[0] / 752)), 3)
                    self.assertLessEqual(abs(y - round(point[1] * size[1] / 423)), 3)

    def test_announcement_close_requires_its_own_page_not_an_unrelated_cross(self):
        source = cv2.resize(capture("announcement"), (752, 423),
                            interpolation=cv2.INTER_AREA)
        for page in ("8", "9", "10", "11", "12", "13"):
            unrelated = cv2.resize(capture(page), (752, 423),
                                   interpolation=cv2.INTER_AREA)
            # The exact real cross on an unrelated page must not grant
            # permission to click a superficially matching panel close.
            unrelated[48:84, 697:729] = source[48:84, 697:729]
            for size in SIZES:
                with self.subTest(page=page, size=size):
                    current = VisualFrame(cv2.resize(unrelated, size))
                    self.assertIsNone(self.vision.find(current, "announcement_page"))
                    self.assertIsNone(self.vision.find(current, "announcement_close"))

    def test_announcement_remains_recognizable_when_update_body_changes(self):
        source = capture("announcement")
        # Announcement copy, dates, left event cards, and remaining duration
        # can change without changing the stable title and close controls.
        source[125:482, 537:879] = (64, 71, 78)
        source[112:482, 60:434] = (104, 96, 88)
        source[482:504, 145:382] = (64, 71, 78)
        for size in SIZES:
            with self.subTest(size=size):
                current = VisualFrame(cv2.resize(source, size))
                self.assertIsNotNone(self.vision.find(current, "announcement_page"))
                self.assertIsNotNone(self.vision.find(current, "announcement_close"))

    def test_announcement_title_or_cross_occlusion_never_yields_a_click(self):
        source = cv2.resize(capture("announcement"), (752, 423),
                            interpolation=cv2.INTER_AREA)
        for name, (left, top, right, bottom) in {
            "title": (419, 50, 458, 94),
            "cross": (697, 48, 713, 84),
        }.items():
            covered = source.copy()
            covered[top:bottom, left:right] = (61, 52, 37)
            for size in SIZES:
                with self.subTest(obscured=name, size=size):
                    current = VisualFrame(cv2.resize(covered, size))
                    self.assertIsNone(self.vision.find(current, "announcement_close"))
                    if name == "title":
                        self.assertIsNone(self.vision.find(current, "announcement_page"))
                    else:
                        self.assertIsNotNone(self.vision.find(current, "announcement_page"))

    def test_dimmed_announcement_below_a_modal_is_not_actionable(self):
        source = capture("announcement")
        for brightness in (.8, .65, .5):
            dark = (source * brightness).astype(np.uint8)
            for size in SIZES:
                with self.subTest(brightness=brightness, size=size):
                    current = VisualFrame(cv2.resize(dark, size))
                    self.assertIsNone(self.vision.find(current, "announcement_page"))
                    self.assertIsNone(self.vision.find(current, "announcement_close"))

    def test_start_button_cannot_trigger_outside_the_dragon_assistant(self):
        town = capture("12")
        source = capture("9")
        # Move the real complete start text into its normal screen position,
        # while retaining a real but unrelated game page around it.
        town[250:265, 627:688] = source[250:265, 627:688]
        self.assertIsNone(self.vision.find(VisualFrame(town), "start_hosting"))

    def test_cloud_cover_is_absence_and_a_fresh_clear_frame_recovers(self):
        covered = capture("12")
        covered[129:166, 431:516] = 244
        self.assertIsNotNone(self.vision.find(VisualFrame(covered), "town"))
        self.assertIsNone(self.vision.find(VisualFrame(covered), "dragon_entry"))
        self.assertIsNotNone(self.vision.find(frame("12"), "dragon_entry"))

    def test_assistant_with_an_obscured_button_is_not_a_guessed_hosting_state(self):
        covered = capture("8")
        covered[247:269, 622:695] = (61, 52, 37)
        current = VisualFrame(covered)
        self.assertIsNotNone(self.vision.find(current, "assistant_page"))
        self.assertIsNone(self.vision.find(current, "start_hosting"))
        self.assertIsNone(self.vision.find(current, "stop_hosting"))

    def test_partially_occluded_name_cannot_be_clicked_using_the_remaining_suffix(self):
        covered = capture("13")
        covered[116:135, 306:341] = (40, 45, 95)
        current = VisualFrame(covered)
        self.assertIsNotNone(self.vision.find(current, "battle_scene"))
        self.assertIsNone(self.vision.find(current, "dragon_npc"))
        self.assertIsNone(self.vision.find(current, "quest_mark"))

    def test_darkened_notice_does_not_pass_the_town_guard(self):
        for size in SIZES:
            with self.subTest(size=size):
                current = frame("11", size)
                self.assertIsNotNone(self.vision.find(current, "notice"))
                self.assertIsNone(self.vision.find(current, "town"))
                self.assertIsNone(self.vision.find(current, "town_eye"))
                self.assertIsNone(self.vision.find(current, "dragon_entry"))

    def test_existing_pages_only_recognize_the_real_town_page(self):
        names = list(self.vision.profile["templates"])
        for path in sorted((ROOT / "tests" / "fixtures" / "points_arena").glob("[1-7].png")):
            image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
            for name in names:
                with self.subTest(page=path.name, name=name):
                    found = self.vision.find(VisualFrame(image), name)
                    if path.stem == "2" and name in {"town", "town_eye", "dragon_entry"}:
                        self.assertIsNotNone(found)
                    else:
                        self.assertIsNone(found)

    def test_removed_dialogue_is_not_registered_or_actionable(self):
        self.assertNotIn("not_started", self.vision.profile["templates"])
        self.assertNotIn("dragon_dialogue", self.vision.profile["templates"])
        for size in SIZES:
            with self.subTest(size=size):
                current = frame("native_not_started", size)
                # The visible button belongs to the dimmed page below the
                # dialogue and must not be treated as an actionable control.
                self.assertIsNone(self.vision.find(current, "start_hosting"))
                self.assertIsNone(self.vision.find(current, "stop_hosting"))
                self.assertIsNone(self.vision.find(current, "town_eye"))

    def test_eye_recognizes_native_town_capture_at_each_viewport_size(self):
        path = ROOT / "tests" / "fixtures" / "points_arena" / "native_town.png"
        source = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        for size in SIZES:
            with self.subTest(size=size):
                current = VisualFrame(cv2.resize(source, size))
                self.assertIsNotNone(self.vision.find(current, "town_eye"))
                self.assertIsNotNone(self.vision.find(current, "town"))

    def test_real_daily_badge_and_eye_edge_cloud_still_identify_town(self):
        path = ROOT / "tests" / "fixtures" / "pet_loadouts" / "town_daily_badge_20261108.png"
        source = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        for size in SIZES:
            with self.subTest(size=size):
                current = VisualFrame(cv2.resize(source, size))
                self.assertIsNotNone(self.vision.find(current, "town_eye"))
                self.assertIsNotNone(self.vision.find(current, "town"))

    def test_real_fully_clouded_eye_confirms_town_without_claiming_visible_eye(self):
        path = ROOT / "tests" / "fixtures" / "points_arena" / "town_clouds_20261108.jpg"
        source = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        for size in SIZES:
            with self.subTest(size=size):
                current = VisualFrame(cv2.resize(source, size))
                self.assertIsNone(self.vision.find(current, "town_eye"))
                self.assertIsNotNone(self.vision.find(current, "town"))
                self.assertIsNotNone(self.vision.find(current, "dragon_entry"))

    def test_real_clouded_eye_requires_clear_independent_town_anchor(self):
        path = ROOT / "tests" / "fixtures" / "points_arena" / "town_clouds_20261108.jpg"
        source = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        covered = source.copy()
        covered[320:520, 775:1000] = 0
        variants = [covered, *((source.astype(float) * brightness).astype(np.uint8)
                               for brightness in (.85, .8))]
        for variant in variants:
            for size in SIZES:
                with self.subTest(covered=variant is covered, size=size):
                    current = VisualFrame(cv2.resize(variant, size))
                    for name in ("town_eye", "town", "dragon_entry"):
                        self.assertIsNone(self.vision.find(current, name), name)

    def test_daily_badge_town_with_missing_anchors_or_dim_background_is_rejected(self):
        path = ROOT / "tests" / "fixtures" / "pet_loadouts" / "town_daily_badge_20261108.png"
        source = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        covered = source.copy()
        covered[166:284, 1147:1290] = 0
        covered[320:520, 775:1000] = 0
        variants = [covered, *((source.astype(float) * brightness).astype(np.uint8)
                               for brightness in (.85, .8, .55))]
        for variant in variants:
            for size in SIZES:
                with self.subTest(covered=variant is covered, size=size):
                    current = VisualFrame(cv2.resize(variant, size))
                    for name in ("town_eye", "town", "dragon_entry"):
                        self.assertIsNone(self.vision.find(current, name), name)

    def test_eye_keeps_town_visible_when_entry_text_and_family_are_clouded(self):
        source = capture("12")
        source[129:166, 431:516] = 244
        source[128:202, 304:386] = 244
        for size in SIZES:
            with self.subTest(size=size):
                current = VisualFrame(cv2.resize(source, size))
                self.assertIsNotNone(self.vision.find(current, "town_eye"))
                self.assertIsNotNone(self.vision.find(current, "town"))
                self.assertIsNone(self.vision.find(current, "dragon_entry"))

    def test_eye_occlusion_is_absence_with_independent_town_fallback(self):
        source = capture("12")
        source[68:109, 455:499] = 244
        for size in SIZES:
            with self.subTest(size=size):
                current = VisualFrame(cv2.resize(source, size))
                self.assertIsNone(self.vision.find(current, "town_eye"))
                self.assertIsNotNone(self.vision.find(current, "town"))
                self.assertIsNotNone(self.vision.find(current, "dragon_entry"))

    def test_both_town_anchors_missing_is_not_a_guessed_town(self):
        source = capture("12")
        source[68:109, 455:499] = 244
        source[128:202, 304:386] = 244
        for size in SIZES:
            with self.subTest(size=size):
                current = VisualFrame(cv2.resize(source, size))
                self.assertIsNone(self.vision.find(current, "town_eye"))
                self.assertIsNone(self.vision.find(current, "town"))
                self.assertIsNone(self.vision.find(current, "dragon_entry"))

    def test_non_landscape_capture_is_rejected(self):
        with self.assertRaises(AutomationError):
            VisualFrame(cv2.resize(capture("8"), (1080, 1920)))


if __name__ == "__main__":
    unittest.main()
