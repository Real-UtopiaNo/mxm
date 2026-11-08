"""Fail-closed recognition of the points-arena pages and allowed opponents.

Page controls use the supplied 849 x 477 desktop captures.  Player names are
read directly at VisualFrame's reference resolution to preserve thin strokes.
All returned coordinates use that reference system; callers must still use
VisualFrame.device_point before tapping.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .adb import AutomationError
from .vision import Match, VisualFrame


def _trim(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _letters(mask: np.ndarray) -> list[np.ndarray]:
    occupied = np.any(mask, axis=0)
    edges = np.flatnonzero(np.diff(np.r_[False, occupied, False].astype(int)))
    return [part for left, right in zip(edges[::2], edges[1::2])
            if (part := _trim(mask[:, left:right])) is not None]


def _similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Compare full glyph silhouettes; never search for a substring."""
    first = cv2.resize(first, (24, 32), interpolation=cv2.INTER_AREA) > 100
    second = cv2.resize(second, (24, 32), interpolation=cv2.INTER_AREA) > 100
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 0.0


def _holes(mask: np.ndarray) -> int:
    normalized = (cv2.resize(mask, (24, 32), interpolation=cv2.INTER_AREA) > 100).astype(np.uint8)
    contours, hierarchy = cv2.findContours(normalized, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return sum(entry[3] >= 0 and cv2.contourArea(contour) > 3
               for contour, entry in zip(contours, hierarchy[0]))


def _name_similarity(word: np.ndarray, template: np.ndarray,
                     fixed_characters: int = 0) -> float | None:
    """Compare every letter and its layout without resampling the full word.

    A one-pixel change to a narrow ``l`` at 849 pixels can move all later
    letters in the word.  Whole-word IoU then measures rasterization phase
    rather than spelling.  Preserve each letter's aspect and relative height
    on a common baseline; permit only a two-pixel contour tolerance at 48px.
    The full word's aspect, character count, and character positions remain
    independent checks, so a prefix or changed character cannot disappear.
    """
    def parts(mask: np.ndarray) -> list[np.ndarray]:
        if not fixed_characters:
            return _letters(mask)
        # Full-width CJK characters can have disconnected radicals or touch
        # the next character after scaling.  Their uniform character cells
        # remain stable even when column-wise connected groups do not.
        edges = [0]
        projection = np.count_nonzero(mask, axis=0)
        radius = max(1, round(mask.shape[0] * .12))
        for index in range(1, fixed_characters):
            ideal = mask.shape[1] * index / fixed_characters
            low = max(edges[-1] + 1, round(ideal) - radius)
            high = min(mask.shape[1] - 1, round(ideal) + radius)
            edges.append(min(range(low, high + 1),
                             key=lambda x: (projection[x], abs(x - ideal))))
        edges.append(mask.shape[1])
        return [part for left, right in zip(edges[:-1], edges[1:])
                if (part := _trim(mask[:, left:right])) is not None]

    letters, expected = parts(word), parts(template)
    if not letters or len(letters) != len(expected):
        return None
    aspect = word.shape[1] / word.shape[0]
    expected_aspect = template.shape[1] / template.shape[0]
    if abs(aspect / expected_aspect - 1) > .16:
        return None

    def layout(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        occupied = np.any(mask, axis=0)
        edges = np.flatnonzero(np.diff(np.r_[False, occupied, False].astype(int)))
        centers = (edges[::2] + edges[1::2]) / (2 * mask.shape[1])
        gaps = (edges[2::2] - edges[1:-1:2]) / mask.shape[0]
        return centers, gaps

    if not fixed_characters:
        centers, gaps = layout(word)
        expected_centers, expected_gaps = layout(template)
        if np.max(np.abs(centers - expected_centers)) > .045:
            return None
        # At the game's 26px native letter height this allows at most about
        # two pixels of phase rounding, never an inserted blank/name separator.
        if gaps.size and np.max(np.abs(gaps - expected_gaps)) > .09:
            return None

    def normalized(letter: np.ndarray, word_height: int) -> np.ndarray:
        height = max(1, round(letter.shape[0] * 48 / word_height))
        width = max(1, round(letter.shape[1] * 48 / word_height))
        result = np.zeros((52, 96), np.uint8)
        if width > result.shape[1] or height > result.shape[0]:
            return result
        left = (result.shape[1] - width) // 2
        result[-height:, left:left + width] = (
            cv2.resize(letter, (width, height), interpolation=cv2.INTER_AREA) > 100
        ).astype(np.uint8)
        return result

    kernel = np.ones((5, 5), np.uint8)
    scores = []
    for letter, expected_letter in zip(letters, expected):
        # Contour tolerance must not close a c into an o or remove the eye of
        # an e.  Topology is stable for the registered Latin names even when
        # their antialiasing and one-pixel stroke widths change.
        if not fixed_characters and _holes(letter) != _holes(expected_letter):
            return None
        if not fixed_characters and expected_letter.shape[1] / expected_letter.shape[0] >= .25:
            # Keep a strict silhouette check for broad letters.  The contour
            # tolerance alone could otherwise make an a resemble an e.  Thin
            # strokes use the aspect-preserving comparison below because a
            # one-pixel width change dominates their stretched IoU.
            if _similarity(letter, expected_letter) < .78:
                return None
        # Full-width CJK glyphs share a font height; a one-pixel difference in
        # their individual extrema should not shift a whole radical vertically.
        # Latin relative heights remain meaningful for case and spelling.
        actual = normalized(letter, letter.shape[0] if fixed_characters else word.shape[0])
        wanted = normalized(expected_letter, expected_letter.shape[0] if fixed_characters else template.shape[0])
        # Both directed coverages must pass: extra strokes and missing strokes
        # are equally disqualifying, even if the other letters are identical.
        forward = np.count_nonzero(actual & cv2.dilate(wanted, kernel)) / max(1, np.count_nonzero(actual))
        backward = np.count_nonzero(wanted & cv2.dilate(actual, kernel)) / max(1, np.count_nonzero(wanted))
        scores.append(min(forward, backward))
    if min(scores) < .93 or float(np.mean(scores)) < .97:
        return None
    return float(np.mean(scores))


class PointsArenaVision:
    _PAGES = {
        "town": "world", "arena": "town_page", "points": "arena_page",
        "exit": "arena_page", "join": "lobby", "close": "lobby",
        "confirm_entry": "entry_dialog", "roster_close": "roster",
        "confirm_result": "result",
    }

    def __init__(self, directory: Path):
        self.directory = directory
        self.profile = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
        self.source_size = tuple(self.profile["source_size"])
        self.images = {name: self._read(entry["file"], cv2.IMREAD_COLOR)
                       for name, entry in self.profile["templates"].items()}
        self.alternatives = {
            name: [(self._read(alternative["file"], cv2.IMREAD_COLOR),
                    {**entry, **alternative})
                   for alternative in entry.get("alternatives", [])]
            for name, entry in self.profile["templates"].items()
        }
        self.players = {
            name: self._read(entry["file"] if isinstance(entry, dict) else entry, cv2.IMREAD_GRAYSCALE)
            for name, entry in self.profile["players"].items()
        }
        self.player_source_sizes = {
            name: tuple(entry.get("source_size", self.source_size)) if isinstance(entry, dict) else self.source_size
            for name, entry in self.profile["players"].items()
        }
        # Native captures and desktop screenshots rasterize thin letters
        # differently at the matching scale.  Keep independently captured
        # silhouettes instead of weakening full-word or per-letter checks.
        self.player_variants = {}
        self.player_variant_source_sizes = {}
        for name, entries in self.profile.get("player_variants", {}).items():
            self.player_variants[name] = [self._read(entry["file"] if isinstance(entry, dict) else entry,
                                                   cv2.IMREAD_GRAYSCALE) for entry in entries]
            self.player_variant_source_sizes[name] = [
                tuple(entry.get("source_size", self.source_size)) if isinstance(entry, dict) else self.source_size
                for entry in entries
            ]
        self.digits = {int(name): self._read(filename, cv2.IMREAD_GRAYSCALE)
                       for name, filename in self.profile["digits"].items()}

    def _read(self, filename: str, mode: int) -> np.ndarray:
        path = self.directory / filename
        result = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), mode)
        if result is None:
            raise AutomationError(f"无法读取积分赛视觉模板：{path}")
        return result

    def _image(self, frame: VisualFrame) -> np.ndarray:
        return cv2.resize(frame.image, self.source_size, interpolation=cv2.INTER_AREA)

    def _raw(self, image: np.ndarray, name: str) -> Match | None:
        entry = self.profile["templates"][name]
        for template, config in [(self.images[name], entry), *self.alternatives[name]]:
            left, top, right, bottom = config["search"]
            source = image[max(0, top):bottom, max(0, left):right]
            height, width = template.shape[:2]
            if source.shape[0] < height or source.shape[1] < width:
                continue
            scores = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(scores)
            if score < config.get("threshold", .86):
                continue
            patch = source[y:y + height, x:x + width]
            if float(np.mean(cv2.absdiff(patch, template))) > config.get("max_error", 22):
                continue
            return Match(name, max(0, left) + x, max(0, top) + y, width, height, score)
        return None

    def _page(self, image: np.ndarray, name: str) -> Match | None:
        match = self._raw(image, name)
        if name == "roster" and self._raw(image, "roster_detail") is None:
            return None
        if name == "entry_dialog" and self._raw(image, "entry_ticket_message") is None:
            return None
        return match

    def _reference(self, frame: VisualFrame, match: Match) -> Match:
        sx = frame.reference_size[0] / self.source_size[0]
        sy = frame.reference_size[1] / self.source_size[1]
        return Match(match.name, round(match.x * sx), round(match.y * sy),
                     round(match.width * sx), round(match.height * sy), match.score)

    def find(self, frame: VisualFrame, name: str) -> Match | None:
        image = self._image(frame)
        page = self._PAGES.get(name)
        if page and self._page(image, page) is None:
            return None
        match = self._page(image, name)
        return self._reference(frame, match) if match else None

    def ticket_status(self, frame: VisualFrame) -> int | None:
        """Return a verified 0..5 ticket count, or None for an unknown/modal page."""
        image = self._image(frame)
        if self._page(image, "lobby") is None or self._raw(image, "ticket_suffix") is None:
            return None
        left, top, right, bottom = self.profile["ticket_search"]
        patch = image[top:bottom, left:right]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        white = (gray > 178) & (patch.max(axis=2).astype(int) - patch.min(axis=2) < 35)
        glyph = _trim(white.astype(np.uint8) * 255)
        if glyph is None or not 4 <= glyph.shape[1] <= 15 or not 13 <= glyph.shape[0] <= 19:
            return None
        scores = sorted(((_similarity(glyph, template), digit)
                         for digit, template in self.digits.items()
                         if _holes(glyph) == _holes(template)), reverse=True)
        if len(scores) < 2:
            return None
        best, digit = scores[0]
        # Ambiguous digits are an observation error, never an available ticket.
        if best < .72 or best - scores[1][0] < .08:
            return None
        return digit

    def target(self, frame: VisualFrame, target_players: list[str]) -> Match | None:
        """Return the first visible, registered name in the full priority list.

        Every nameplate is segmented into the complete visible word and then
        individual glyphs.  Width and glyph-count checks prevent Hell matching
        Hello, and per-glyph checks reject similar names with a changed letter.
        Unregistered names have no visual evidence and cannot match anything.
        Each call starts at the beginning, regardless of previous challenges;
        the list length is independent of the three-battle entry length.
        """
        image = self._image(frame)
        if self._page(image, "roster") is None:
            return None
        candidates = []
        sx = frame.reference_size[0] / self.source_size[0]
        sy = frame.reference_size[1] / self.source_size[1]
        for left, top, width, height in self.profile["name_slots"]:
            # Unchallenged cards have a continuous orange margin beside the
            # portrait.  Completed-battle stamps cover this margin even when
            # the name below them remains readable.  Require a clear margin,
            # so an unknown result overlay is not treated as an opponent.
            margin = image[top - 55:top - 5, left - 7:left - 3]
            hsv = cv2.cvtColor(margin, cv2.COLOR_BGR2HSV)
            orange = ((hsv[:, :, 0] > 3) & (hsv[:, :, 0] < 35)
                      & (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 160))
            if np.count_nonzero(orange) < orange.size * .9:
                continue
            x1, x2 = round(left * sx), round((left + width) * sx)
            y1, y2 = round(top * sy), round((top + height) * sy)
            # Read the name at the frame's high resolution before reducing any
            # thin strokes to one pixel.  Grid slots describe areas, never a
            # player identity; every frame scans all eight of them afresh.
            patch = frame.image[y1:y2, x1:x2]
            # Verify the empty margins instead of a whole-plate white fraction:
            # a longer registered name legitimately occupies more of its plate.
            margin_width = max(2, round(patch.shape[1] * .05))
            pale = patch.min(axis=2) > 170
            if (np.mean(pale[:, :margin_width]) < .65
                    or np.mean(pale[:, -margin_width:]) < .65):
                continue
            # The brown nameplate outline can enter a crop by one pixel after
            # scaling.  It is more saturated than the dark name text.
            spread = patch.max(axis=2).astype(int) - patch.min(axis=2)
            # A second ink threshold recovers antialiased edges in lower
            # resolution captures; every candidate still passes the same
            # spelling, topology, shape, and inter-letter spacing checks.
            for threshold in (145, 160):
                mask = ((patch.max(axis=2) < threshold) & (spread < 55)).astype(np.uint8) * 255
                # Border antialiasing may leave isolated specks in the outer
                # crop row.  Remove only those tiny edge components, never a
                # connected letter or an internal mark of the player's name.
                _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
                for index, (_, y, _, component_height, area) in enumerate(stats[1:], 1):
                    if area <= 4 and (y == 0 or y + component_height == mask.shape[0]):
                        mask[labels == index] = 0
                glyph = _trim(mask)
                if glyph is None:
                    continue
                ys, xs = np.nonzero(mask)
                # A word touching the crop edge may be truncated.  Fail closed.
                if xs.min() <= 1 or xs.max() >= patch.shape[1] - 2:
                    continue
                candidates.append((x1 + int(xs.min()), y1 + int(ys.min()), glyph))
        # Player priority is the outer loop: neither screen order nor a higher
        # match score for a later player may override an earlier visible name.
        for name in target_players:
            template = self.players.get(name)
            if template is None:
                continue
            fixed = len(name) if all("\u2e80" <= char <= "\u9fff" for char in name) else 0
            for index, variant in enumerate([template, *self.player_variants.get(name, [])]):
                source_height = (self.player_source_sizes.get(name, self.source_size)[1] if index == 0 else
                                 self.player_variant_source_sizes[name][index - 1][1])
                expected_height = variant.shape[0] * frame.reference_size[1] / source_height
                for left, top, glyph in candidates:
                    # Capital C/O and lowercase c/o have similar outlines, but
                    # different heights in the game's fixed font.  Preserve
                    # absolute font size as well as normalized letter shapes.
                    if abs(glyph.shape[0] - expected_height) > 2 * frame.reference_size[1] / 1080:
                        continue
                    score = _name_similarity(glyph, variant, fixed)
                    if score is None:
                        continue
                    return Match(f"player:{name}", left, top, glyph.shape[1], glyph.shape[0], score)
        return None
