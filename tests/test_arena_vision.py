from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from pet_alliance.arena_vision import PointsArenaVision, _letters, _trim
from pet_alliance.vision import VisualFrame


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "points_arena"

# Independent test coordinates measured from the supplied roster captures.
# These describe whole cards, including their portrait, result overlay and
# nameplate, rather than borrowing the recognizer's name search windows.
ROSTER_SIZE = (849, 477)
DEVICE_SIZE = (1920, 1080)
CARD_COLUMNS = (278, 385, 491, 598)
CARD_ROWS = (115, 254)
CARD_SIZE = (102, 135)
NAMEPLATE_CENTERS_X = (330, 436, 543, 649)
NAMEPLATE_CENTERS_Y = (208, 347)
REAL_CARDS = {
    "Hello": ("native_after_hell", 1),
    "Hell": ("native_after_hello_coco", 1),
    "coco": ("native_after_hell", 2),
    "猛踢技术蛋": ("native_after_hello_coco", 0),
    "completed:Hell": ("native_after_hell", 0),
}


def image(name: str) -> np.ndarray:
    path = FIXTURES / f"{name}.png"
    if not path.exists():
        path = FIXTURES / f"{name}.jpg"
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def frame(name: str, size=None) -> VisualFrame:
    source = image(name)
    return VisualFrame(cv2.resize(source, size) if size else source)


def card_origin(slot: int) -> tuple[int, int]:
    """Map a fixed 849-pixel grid position into the native capture."""
    return (round(CARD_COLUMNS[slot % 4] * DEVICE_SIZE[0] / ROSTER_SIZE[0]),
            round(CARD_ROWS[slot // 4] * DEVICE_SIZE[1] / ROSTER_SIZE[1]))


def real_card(name: str) -> np.ndarray:
    capture, slot = REAL_CARDS[name]
    source = image(capture)
    left, top = card_origin(slot)
    width = round(CARD_SIZE[0] * DEVICE_SIZE[0] / ROSTER_SIZE[0])
    height = round(CARD_SIZE[1] * DEVICE_SIZE[1] / ROSTER_SIZE[1])
    return source[top:top + height, left:left + width].copy()


def arranged_roster(cards: dict[int, str], size=DEVICE_SIZE) -> VisualFrame:
    """Move complete independently captured cards onto a clean roster.

    Preserve the real page heading and frame, but clear the entire card grid
    before placing anything.  Thus a residual original name cannot satisfy an
    assertion about a moved card, nor can an isolated pasted name counterfeit
    an eligible portrait.
    """
    source = image("native_after_hell")
    left = round(274 * DEVICE_SIZE[0] / ROSTER_SIZE[0])
    right = round(703 * DEVICE_SIZE[0] / ROSTER_SIZE[0])
    top = round(112 * DEVICE_SIZE[1] / ROSTER_SIZE[1])
    bottom = round(394 * DEVICE_SIZE[1] / ROSTER_SIZE[1])
    # This row is empty in the real screenshot.  Retain its horizontal
    # background shading and repeat it vertically throughout the empty grid.
    background = source[850:851, left:right].copy()
    source[top:bottom, left:right] = background
    for slot, name in cards.items():
        card = real_card(name)
        x, y = card_origin(slot)
        source[y:y + card.shape[0], x:x + card.shape[1]] = card
    return VisualFrame(cv2.resize(source, size) if size != DEVICE_SIZE else source)


def registered_real_name(name: str) -> np.ndarray:
    """Build a future legacy ``players`` entry from independent real pixels."""
    source = cv2.resize(image(REAL_CARDS[name][0]), ROSTER_SIZE,
                        interpolation=cv2.INTER_AREA)
    column = REAL_CARDS[name][1] % 4
    center = NAMEPLATE_CENTERS_X[column]
    patch = source[201:215, center - 38:center + 38]
    mask = (patch.max(axis=2) < 145).astype(np.uint8) * 255
    x, y, width, height = cv2.boundingRect(mask)
    return mask[y:y + height, x:x + width]


class PointsArenaVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vision = PointsArenaVision(ROOT / "assets" / "points_arena")

    def test_pages_and_controls_in_all_seven_supplied_captures(self):
        expected = {
            "1": {"world", "town"}, "2": {"town_page", "arena"},
            "3": {"arena_page", "points", "exit"}, "4": {"lobby", "join", "close"},
            "5": {"entry_dialog", "confirm_entry"},
            "6": {"roster", "roster_close"}, "7": {"result", "confirm_result"},
        }
        names = set.union(*expected.values())
        for capture, wanted in expected.items():
            with self.subTest(capture=capture):
                found = {name for name in names if self.vision.find(frame(capture), name)}
                self.assertEqual(found, wanted)

    def test_native_adb_town_validates_desktop_capture_templates(self):
        current = frame("native_town")
        self.assertEqual((current.original_width, current.original_height), (1920, 1080))
        for name in ("town_page", "arena"):
            match = self.vision.find(current, name)
            self.assertIsNotNone(match)
            self.assertGreater(match.score, .93)

    def test_cloudy_town_from_failed_hero_run_is_recognized_at_five_sizes(self):
        for size in ((849, 477), (956, 538), (1280, 720), (1374, 772), (1920, 1080)):
            with self.subTest(size=size):
                current = frame("town_clouds_20261107", size)
                self.assertIsNotNone(self.vision.find(current, "town_page"))
                self.assertIsNotNone(self.vision.find(current, "arena"))

    def test_real_daily_badge_and_eye_edge_cloud_still_identify_town(self):
        path = ROOT / "tests" / "fixtures" / "pet_loadouts" / "town_daily_badge_20261108.png"
        source = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        for size in ((752, 423), (849, 477), (956, 538), (1280, 720),
                     (1374, 772), (1920, 1080)):
            with self.subTest(size=size):
                self.assertIsNotNone(self.vision.find(
                    VisualFrame(cv2.resize(source, size)), "town_page"))

    def test_daily_badge_town_cannot_authorize_input_with_missing_or_dim_anchors(self):
        path = ROOT / "tests" / "fixtures" / "pet_loadouts" / "town_daily_badge_20261108.png"
        source = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        covered = source.copy()
        covered[166:284, 1147:1290] = 0
        covered[320:520, 775:1000] = 0
        variants = [covered, *((source.astype(float) * brightness).astype(np.uint8)
                               for brightness in (.85, .8, .55))]
        for variant in variants:
            for size in ((752, 423), (849, 477), (956, 538), (1280, 720),
                         (1374, 772), (1920, 1080)):
                with self.subTest(covered=variant is covered, size=size):
                    current = VisualFrame(cv2.resize(variant, size))
                    self.assertIsNone(self.vision.find(current, "town_page"))
                    self.assertIsNone(self.vision.find(current, "arena"))

    def test_town_identity_does_not_depend_on_cloud_covered_building_labels(self):
        source = image("town_clouds_20261107")
        # Cover the old camp nameplate and the shield considered as another
        # anchor; neither is needed to confirm the town's fixed red eye.
        source[630:716, 220:397] = 0
        source[325:472, 790:966] = 0
        self.assertIsNotNone(self.vision.find(VisualFrame(source), "town_page"))

    def test_covered_or_dim_town_anchors_do_not_authorize_arena_input(self):
        covered = image("town_clouds_20261107")
        covered[166:284, 1147:1290] = 0
        covered[320:520, 775:1000] = 0
        dimmed = (image("town_clouds_20261107").astype(float) * .55).astype(np.uint8)
        for source in (covered, dimmed):
            with self.subTest(dimmed=source is dimmed):
                current = VisualFrame(source)
                self.assertIsNone(self.vision.find(current, "town_page"))
                self.assertIsNone(self.vision.find(current, "arena"))

    def test_real_fully_clouded_eye_town_confirms_arena_at_six_sizes(self):
        # The failed run really reached town.  Its cloud covers the entire
        # red eye while the independent shield core remains unobstructed.
        for size in ((752, 423), (849, 477), (956, 538), (1280, 720),
                     (1374, 772), (1920, 1080)):
            with self.subTest(size=size):
                current = frame("town_clouds_20261108", size)
                self.assertIsNotNone(self.vision.find(current, "town_page"))
                self.assertIsNotNone(self.vision.find(current, "arena"))

    def test_real_clouded_eye_requires_clear_independent_town_anchor(self):
        source = image("town_clouds_20261108")
        covered = source.copy()
        # Cover the entire shield and its search margin, leaving the genuine
        # cloud over the red eye untouched instead of manufacturing a clear eye.
        covered[320:520, 775:1000] = 0
        variants = [covered, *((source.astype(float) * brightness).astype(np.uint8)
                               for brightness in (.85, .8))]
        for variant in variants:
            for size in ((752, 423), (849, 477), (956, 538), (1280, 720),
                         (1374, 772), (1920, 1080)):
                with self.subTest(covered=variant is covered, size=size):
                    current = VisualFrame(cv2.resize(variant, size))
                    self.assertIsNone(self.vision.find(current, "town_page"))
                    self.assertIsNone(self.vision.find(current, "arena"))

    def test_source_and_native_resolution_return_device_coordinates(self):
        for size in ((849, 477), (1920, 1080), (1280, 720)):
            with self.subTest(size=size):
                current = frame("3", size)
                match = self.vision.find(current, "exit")
                self.assertIsNotNone(match)
                x, y = current.device_point(match.center)
                self.assertAlmostEqual(x / size[0], 115 / 849, delta=.003)
                self.assertAlmostEqual(y / size[1], 265 / 477, delta=.003)

    def test_full_allowlisted_names_at_supported_resolutions(self):
        for size in ((849, 477), (1920, 1080), (1280, 720)):
            for name in ("Hello", "Hell", "coco"):
                with self.subTest(size=size, name=name):
                    match = self.vision.target(frame("6", size), [name])
                    self.assertIsNotNone(match)
                    self.assertEqual(match.name, f"player:{name}")
        self.assertEqual(self.vision.target(frame("6"), ["coco", "Hello"]).name, "player:coco")

    def test_empty_and_unregistered_allowlists_match_nothing(self):
        for names in ([], ["Other"], ["hello"], ["Hel"], ["Hello2"]):
            self.assertIsNone(self.vision.target(frame("6"), names))

    def test_real_roster_after_hell_still_selects_highest_priority_hello(self):
        # Captured at the reported stop after the first arena battle.  The
        # remaining names occupy three first-row cards, unlike the mock roster.
        current = frame("native_after_hell")
        self.assertEqual((current.original_width, current.original_height), (1920, 1080))
        match = self.vision.target(current, ["Hello", "Hell", "coco"])
        self.assertIsNotNone(match)
        self.assertEqual(match.name, "player:Hello")

    def test_real_reordered_roster_after_hello_coco_selects_hell(self):
        match = self.vision.target(frame("native_after_hello_coco"), ["Hello", "Hell", "coco"])
        self.assertIsNotNone(match)
        self.assertEqual(match.name, "player:Hell")

    def assert_player_at_slot(self, current, names, wanted, slot):
        match = self.vision.target(current, names)
        self.assertIsNotNone(match)
        self.assertEqual(match.name, f"player:{wanted}")
        x, y = current.device_point(match.center)
        self.assertAlmostEqual(x / current.original_width,
                               NAMEPLATE_CENTERS_X[slot % 4] / ROSTER_SIZE[0],
                               delta=.004)
        self.assertAlmostEqual(y / current.original_height,
                               NAMEPLATE_CENTERS_Y[slot // 4] / ROSTER_SIZE[1],
                               delta=.004)

    def test_each_real_player_in_every_grid_slot_at_supported_resolutions(self):
        # Three independent real names x eight complete card positions x
        # three resolutions.  The only name on each page is the moved card.
        for name in ("Hello", "Hell", "coco"):
            for slot in range(8):
                for size in (ROSTER_SIZE, DEVICE_SIZE, (1280, 720)):
                    with self.subTest(name=name, slot=slot, size=size):
                        current = arranged_roster({slot: name}, size)
                        self.assert_player_at_slot(current,
                                                   ["Hello", "Hell", "coco"],
                                                   name, slot)

    def test_cleared_real_roster_contains_no_residual_opponents(self):
        for size in (ROSTER_SIZE, DEVICE_SIZE, (1280, 720)):
            with self.subTest(size=size):
                current = arranged_roster({}, size)
                self.assertIsNotNone(self.vision.find(current, "roster"))
                self.assertIsNone(self.vision.target(current, ["Hello", "Hell", "coco"]))

    def test_real_cards_reorder_across_rows_without_changing_name_priority(self):
        arrangements = (
            {0: "coco", 4: "Hell", 7: "Hello"},
            {1: "Hello", 3: "Hell", 6: "coco"},
            {2: "Hell", 5: "Hello", 7: "coco"},
        )
        for cards in arrangements:
            for size in (ROSTER_SIZE, DEVICE_SIZE, (1280, 720)):
                current = arranged_roster(cards, size)
                for names in (["Hello", "Hell", "coco"],
                              ["coco", "Hello", "Hell"],
                              ["Hell", "coco", "Hello"]):
                    with self.subTest(cards=cards, size=size, names=names):
                        wanted = names[0]
                        slot = next(slot for slot, name in cards.items() if name == wanted)
                        self.assert_player_at_slot(current, names, wanted, slot)

    def test_new_real_player_obeys_extended_priority_list_after_reordering(self):
        new_name = "猛踢技术蛋"
        glyph = registered_real_name(new_name)
        names = ["Missing A", "Missing B", "Missing C", "Missing D",
                 new_name, "coco", "Hello", "Hell"]
        with patch.dict(self.vision.players, {new_name: glyph}):
            for slot in range(8):
                for size in (ROSTER_SIZE, DEVICE_SIZE, (1280, 720)):
                    cards = {slot: new_name, (slot + 1) % 8: "coco",
                             (slot + 3) % 8: "Hello", (slot + 5) % 8: "Hell"}
                    current = arranged_roster(cards, size)
                    with self.subTest(slot=slot, size=size, priority="new"):
                        self.assert_player_at_slot(current, names, new_name, slot)
                    with self.subTest(slot=slot, size=size, priority="existing"):
                        self.assert_player_at_slot(current, ["Hello", *names], "Hello",
                                                   (slot + 3) % 8)

    def test_real_victory_card_in_every_slot_is_never_an_opponent(self):
        for slot in range(8):
            for size in (ROSTER_SIZE, DEVICE_SIZE, (1280, 720)):
                with self.subTest(slot=slot, size=size):
                    current = arranged_roster({slot: "completed:Hell"}, size)
                    self.assertIsNone(self.vision.target(current, ["Hell"]))
                    # The same result card cannot block the next eligible name
                    # simply because Hell has higher priority in the list.
                    eligible_slot = (slot + 1) % 8
                    current = arranged_roster({slot: "completed:Hell",
                                              eligible_slot: "coco"}, size)
                    self.assert_player_at_slot(current, ["Hell", "coco"],
                                               "coco", eligible_slot)

    def test_real_remaining_names_at_supported_resolutions(self):
        for size in ((849, 477), (1920, 1080), (1280, 720)):
            for names, wanted in ((["Hello", "Hell", "coco"], "Hello"),
                                  (["coco", "Hello"], "coco"),
                                  (["Missing"] * 4 + ["Hello", "coco"], "Hello")):
                with self.subTest(size=size, names=names):
                    match = self.vision.target(frame("native_after_hell", size), names)
                    self.assertIsNotNone(match)
                    self.assertEqual(match.name, f"player:{wanted}")
                    x, y = frame("native_after_hell", size).device_point(match.center)
                    self.assertAlmostEqual(x / size[0], (436 if wanted == "Hello" else 542) / 849,
                                           delta=.004)
                    self.assertAlmostEqual(y / size[1], 208 / 477, delta=.004)

    def test_completed_card_is_rejected_even_with_an_exact_registered_name(self):
        current = frame("native_after_hell")
        source = self.vision._image(current)
        left, top, width, height = self.vision.profile["name_slots"][0]
        glyph = _trim((source[top:top + height, left:left + width].max(axis=2) < 145)
                      .astype(np.uint8) * 255)
        # Exact evidence for the visible Hell word must not make its victory
        # card eligible for a second battle.
        with patch.dict(self.vision.players, {"Hell": glyph}):
            self.assertIsNone(self.vision.target(current, ["Hell"]))

    def test_unknown_or_empty_allowlist_on_real_roster_matches_nothing(self):
        for names in ([], ["Other"], ["hello"], ["Hel"], ["Hello2"]):
            self.assertIsNone(self.vision.target(frame("native_after_hell"), names))

    def test_next_remaining_name_is_selected_after_hello_is_gone(self):
        source = image("native_after_hell")
        # Remove only Hello's nameplate while leaving the completed Hell card.
        source[454:488, 899:1075] = (205, 255, 255)
        match = self.vision.target(VisualFrame(source), ["Hello", "Hell", "coco"])
        self.assertIsNotNone(match)
        self.assertEqual(match.name, "player:coco")

    def test_first_visible_priority_wins_over_card_position(self):
        for names in (["coco", "Hello", "Hell"], ["Hello", "Hell", "coco"], ["Hell", "coco", "Hello"]):
            with self.subTest(names=names):
                self.assertEqual(self.vision.target(frame("6"), names).name, f"player:{names[0]}")

    def test_extended_priority_list_can_select_a_new_registered_fourth_name(self):
        # Register the existing second card's name as a future template would.
        source = self.vision._image(frame("6"))
        left, top, width, height = self.vision.profile["name_slots"][1]
        mask = (source[top:top + height, left:left + width].max(axis=2) < 145).astype(np.uint8) * 255
        ys, xs = np.nonzero(mask)
        glyph = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        names = ["MissingA", "MissingB", "MissingC", "核桃", "coco", "Hello", "Hell"]
        with patch.dict(self.vision.players, {"核桃": glyph}):
            match = self.vision.target(frame("6"), names)
        self.assertIsNotNone(match)
        self.assertEqual(match.name, "player:核桃")

    def names_only(self, glyph: np.ndarray) -> VisualFrame:
        source = image("6")
        for left, top, width, height in self.vision.profile["name_slots"]:
            source[top:top + height, left:left + width] = (205, 255, 255)
        left = 398 + (76 - glyph.shape[1]) // 2
        top = 341
        patch = source[top:top + glyph.shape[0], left:left + glyph.shape[1]]
        patch[glyph > 100] = (35, 45, 45)
        return VisualFrame(source)

    def test_hell_is_not_a_prefix_match_for_hello_in_either_direction(self):
        self.assertIsNone(self.vision.target(self.names_only(self.vision.players["Hell"]), ["Hello"]))
        self.assertIsNone(self.vision.target(self.names_only(self.vision.players["Hello"]), ["Hell"]))

    def test_native_name_variants_keep_prefix_and_changed_letter_guards(self):
        native_hello = self.vision.player_variants["Hello"][0]
        self.assertIsNone(self.vision.target(self.names_only(native_hello), ["Hell"]))
        changed = native_hello.copy()
        changed[:, -8:] = 0
        c = _letters(self.vision.player_variants["coco"][0])[0]
        changed[-c.shape[0]:, -c.shape[1]:] = c
        self.assertIsNone(self.vision.target(self.names_only(changed), ["Hello"]))

    def native_word_only(self, word: np.ndarray) -> VisualFrame:
        # Retain a complete eligible portrait from a real native card while
        # changing only its name, and clear all other cards through the helper.
        source = arranged_roster({0: "Hell"}).image.copy()
        left, top, right, bottom = 660, 455, 832, 487
        source[top:bottom, left:right] = (205, 255, 255)
        x = (left + right - word.shape[1]) // 2
        y = bottom - word.shape[0] - 3
        patch = source[y:y + word.shape[0], x:x + word.shape[1]]
        patch[word > 100] = (30, 40, 40)
        return VisualFrame(source)

    def test_visible_near_names_and_inserted_space_are_rejected(self):
        hell = self.vision.player_variants["Hell"][-1]
        h, e, l1, l2 = _letters(hell)

        def substitute(char: str, height: int) -> np.ndarray:
            canvas = np.zeros((60, 60), np.uint8)
            cv2.putText(canvas, char, (5, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        1, 255, 2, cv2.LINE_AA)
            glyph = _trim((canvas > 100).astype(np.uint8) * 255)
            width = round(glyph.shape[1] * height / glyph.shape[0])
            return cv2.resize(glyph, (width, height), interpolation=cv2.INTER_AREA)

        def join(glyphs: list[np.ndarray]) -> np.ndarray:
            result = np.zeros((26, sum(g.shape[1] for g in glyphs) + 4 * (len(glyphs) - 1)),
                              np.uint8)
            x = 0
            for glyph in glyphs:
                result[-glyph.shape[0]:, x:x + glyph.shape[1]] = glyph
                x += glyph.shape[1] + 4
            return result

        coco = self.vision.player_variants["coco"][-1]
        c1, o1, c2, _ = _letters(coco)
        cases = {
            "Helf": join([h, e, l1, substitute("f", 26)]),
            "Hall": join([h, substitute("a", e.shape[0]), l1, l2]),
            "HeII": join([h, e, substitute("I", 26), substitute("I", 26)]),
            "He11": join([h, e, substitute("1", 26), substitute("1", 26)]),
            "cocc": join([c1, o1, c2, c1]),
        }
        hello = self.vision.player_variants["Hello"][-1]
        occupied = np.any(hello, axis=0)
        edges = np.flatnonzero(np.diff(np.r_[False, occupied, False].astype(int)))
        split = int(edges[3])
        cases["He llo"] = np.concatenate((hello[:, :split],
                                           np.zeros((hello.shape[0], 8), np.uint8),
                                           hello[:, split:]), axis=1)
        for name, word in cases.items():
            for size in ((849, 477), (1920, 1080), (1280, 720)):
                with self.subTest(name=name, size=size):
                    current = self.native_word_only(word)
                    current = VisualFrame(cv2.resize(current.image, size))
                    self.assertIsNone(self.vision.target(current, ["Hello", "Hell", "coco"]))

    def test_uppercase_coco_does_not_match_lowercase_name(self):
        # Independently rendered GDI Arial Bold 32 mask: normalized uppercase
        # C/O outlines can equal lowercase c/o, but their font heights differ.
        word = cv2.imdecode(np.fromfile(str(FIXTURES / "near_uppercase_COCO.png"), np.uint8),
                            cv2.IMREAD_GRAYSCALE)
        self.assertEqual(word.shape, (23, 92))
        for size in ((849, 477), (1920, 1080), (1280, 720)):
            with self.subTest(size=size):
                current = self.native_word_only(word)
                self.assertIsNone(self.vision.target(VisualFrame(cv2.resize(current.image, size)), ["coco"]))

    def test_new_primary_template_can_declare_native_source_resolution(self):
        profile = deepcopy(self.vision.profile)
        profile["players"] = {"Hell": {"file": "player_Hell_highres.png",
                                         "source_size": [1920, 1080]}}
        profile["player_variants"] = {}
        with patch.object(Path, "read_text", return_value=json.dumps(profile)):
            vision = PointsArenaVision(ROOT / "assets" / "points_arena")
        self.assertIsInstance(vision.players["Hell"], np.ndarray)
        self.assertEqual(vision.player_source_sizes["Hell"], (1920, 1080))
        for size in ((849, 477), (1920, 1080), (1280, 720)):
            with self.subTest(size=size):
                match = vision.target(arranged_roster({7: "Hell"}, size), ["Hell"])
                self.assertIsNotNone(match)
                self.assertEqual(match.name, "player:Hell")

    def test_result_overlay_guard_does_not_depend_on_victory_stamp_color(self):
        current = self.names_only(self.vision.players["Hello"])
        self.assertIsNotNone(self.vision.target(current, ["Hello"]))
        source = self.vision._image(current)
        left, top, _, _ = self.vision.profile["name_slots"][5]
        # Synthetic overlays exercise the generic clear-card guard.  Only the
        # victory stamp in native_after_hell is an actual captured result.
        for color in ((255, 255, 255), (190, 80, 25), (20, 20, 160)):
            with self.subTest(color=color):
                covered = source.copy()
                covered[top - 55:top - 5, left - 7:left - 3] = color
                self.assertIsNone(self.vision.target(VisualFrame(covered), ["Hello"]))

    def test_same_length_changed_letter_and_extra_suffix_are_rejected(self):
        hello = self.vision.players["Hello"].copy()
        hello[:, -8:] = 0
        c = _letters(self.vision.players["coco"])[0]
        hello[-c.shape[0]:, -c.shape[1]:] = c
        self.assertIsNone(self.vision.target(self.names_only(hello), ["Hello"]))
        extra = np.pad(self.vision.players["Hello"], ((0, 0), (0, 10)))
        extra[-8:, -8:] = _letters(self.vision.players["Hello"])[-1]
        self.assertIsNone(self.vision.target(self.names_only(extra), ["Hello"]))

    def test_modal_backgrounds_and_wrong_pages_cannot_supply_opponents(self):
        for name in ("1", "2", "3", "4", "5", "7"):
            self.assertIsNone(self.vision.target(frame(name), ["Hello", "Hell", "coco"]))
        dim = VisualFrame((image("6").astype(float) * .41).astype(np.uint8))
        self.assertIsNone(self.vision.target(dim, ["Hello", "Hell", "coco"]))

    def test_generic_confirmation_is_not_a_ticket_confirmation(self):
        source = image("5")
        source[237:260, 321:527] = (135, 141, 146)
        current = VisualFrame(source)
        self.assertIsNone(self.vision.find(current, "entry_dialog"))
        self.assertIsNone(self.vision.find(current, "confirm_entry"))

    def test_real_zero_and_recovered_one_ticket_strip(self):
        for size in ((849, 477), (1920, 1080), (1280, 720)):
            self.assertEqual(self.vision.ticket_status(frame("4", size)), 0)
            self.assertEqual(self.vision.ticket_status(frame("lobby_ticket_1", size)), 1)
        self.assertIsNone(self.vision.ticket_status(frame("5")))
        broken = image("4")
        broken[382:404, 236:262] = 0
        self.assertIsNone(self.vision.ticket_status(VisualFrame(broken)))

    def ticket_glyph(self, glyph: np.ndarray) -> VisualFrame:
        source = image("4")
        source[382:404, 219:236] = (20, 30, 39)
        width = round(glyph.shape[1] / glyph.shape[0] * 16)
        mask = cv2.resize(glyph, (width, 16), interpolation=cv2.INTER_AREA)
        patch = source[385:401, 228 - width // 2:228 - width // 2 + width]
        alpha = mask[:, :, None] / 255
        patch[:] = patch * (1 - alpha) + 255 * alpha
        return VisualFrame(source)

    def test_all_six_ticket_digit_shapes_and_unknown_shape(self):
        # Synthetic ticket strips validate the parser at the original UI scale;
        # only 0/5 and the recovered 1/5 are independently captured ticket strips.
        for digit, glyph in self.vision.digits.items():
            self.assertEqual(self.vision.ticket_status(self.ticket_glyph(glyph)), digit)
        unknown = self.vision.digits[0].copy()
        unknown[7:10, :] = 255  # two enclosed holes: an invalid eight, never zero
        self.assertIsNone(self.vision.ticket_status(self.ticket_glyph(unknown)))
        h = _letters(self.vision.players["Hello"])[0]
        self.assertIsNone(self.vision.ticket_status(self.ticket_glyph(h)))


if __name__ == "__main__":
    unittest.main()
