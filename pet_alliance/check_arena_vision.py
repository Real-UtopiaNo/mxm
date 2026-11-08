"""Recognition of Demon City dungeons and their stage-guardian ownership.

The desktop captures include only the 1374 x 772 game viewport. All public
matches use VisualFrame reference coordinates and must be mapped before input.
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


def _parts(mask: np.ndarray) -> list[np.ndarray]:
    occupied = np.any(mask, axis=0)
    edges = np.flatnonzero(np.diff(np.r_[False, occupied, False].astype(int)))
    return [part for left, right in zip(edges[::2], edges[1::2])
            if (part := _trim(mask[:, left:right])) is not None]


def _same_name(actual: np.ndarray, expected: np.ndarray) -> bool:
    """Compare the entire word, retaining character count, spacing and aspect."""
    if abs((actual.shape[1] / actual.shape[0]) /
           (expected.shape[1] / expected.shape[0]) - 1) > .12:
        return False
    actual_parts, expected_parts = _parts(actual), _parts(expected)
    if len(actual_parts) != len(expected_parts):
        return False
    def spacing(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        occupied = np.any(mask, axis=0)
        edges = np.flatnonzero(np.diff(np.r_[False, occupied, False].astype(int)))
        centers = (edges[::2] + edges[1::2]) / (2 * mask.shape[1])
        gaps = (edges[2::2] - edges[1:-1:2]) / mask.shape[0]
        return centers, gaps
    centers, gaps = spacing(actual)
    expected_centers, expected_gaps = spacing(expected)
    if np.max(np.abs(centers - expected_centers)) > .055:
        return False
    if gaps.size and np.max(np.abs(gaps - expected_gaps)) > .14:
        return False
    for current, wanted in zip(actual_parts, expected_parts):
        if abs((current.shape[1] / current.shape[0]) /
               (wanted.shape[1] / wanted.shape[0]) - 1) > .30:
            return False
        first = cv2.resize(current, (24, 40), interpolation=cv2.INTER_AREA) > 100
        second = cv2.resize(wanted, (24, 40), interpolation=cv2.INTER_AREA) > 100
        # A thin glyph can occupy six pixels in one capture and seven in
        # another. After normalization, allow one horizontal pixel of edge
        # alignment per glyph, retaining the full-word checks above. Padding
        # keeps every pixel in the union even at either alignment boundary.
        first = np.pad(first, ((0, 0), (1, 1)))
        second = np.pad(second, ((0, 0), (1, 1)))
        overlap = 0.0
        for offset in (-1, 0, 1):
            aligned = np.roll(first, offset, axis=1)
            union = np.count_nonzero(aligned | second)
            if union:
                overlap = max(overlap, np.count_nonzero(aligned & second) / union)
        if overlap < .70:
            return False
    return True


class CheckArenaVision:
    def __init__(self, directory: Path):
        self.directory = directory
        self.profile = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
        self.source_size = tuple(self.profile["source_size"])
        self.images = {}
        self.players = {}
        for name, entry in self.profile["templates"].items():
            self.images[name] = self._read_image(entry["file"])
        for name, entry in self.profile["players"].items():
            self.players[name] = {
                place: self._read_image(entry[f"{place}_file"], cv2.IMREAD_GRAYSCALE)
                for place in ("map", "guardian")
            }
        self.dungeons = {entry["name"]: entry["template"]
                         for entry in self.profile["dungeons"]}

    @property
    def dungeon_names(self) -> tuple[str, ...]:
        return tuple(self.dungeons)

    def validate_player(self, name: str) -> None:
        if name not in self.players:
            raise AutomationError(f"没有该完整玩家名的视觉模板：{name}；请先采集姓名模板。")

    def _read_image(self, filename: str, flags=cv2.IMREAD_COLOR) -> np.ndarray:
        path = self.directory / filename
        image = cv2.imdecode(np.fromfile(str(path), np.uint8), flags)
        if image is None:
            raise AutomationError(f"无法读取擂主检查模板：{path}")
        return image

    def _image(self, frame: VisualFrame) -> np.ndarray:
        return cv2.resize(frame.image, self.source_size, interpolation=cv2.INTER_AREA)

    def _raw(self, image: np.ndarray, name: str) -> Match | None:
        entry = self.profile["templates"][name]
        left, top, right, bottom = entry["search"]
        source = image[top:bottom, left:right]
        template = self.images[name]
        height, width = template.shape[:2]
        if source.shape[0] < height or source.shape[1] < width:
            return None
        scores = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        if score < entry.get("threshold", .91):
            return None
        patch = source[y:y + height, x:x + width]
        if float(np.mean(cv2.absdiff(patch, template))) > entry.get("max_error", 22):
            return None
        return Match(name, left + x, top + y, width, height, score)

    def _find(self, image: np.ndarray, name: str) -> Match | None:
        entry = self.profile["templates"][name]
        if any(self._raw(image, guard) is None for guard in entry.get("guards", [])):
            return None
        return self._raw(image, name)

    def _reference(self, frame: VisualFrame, match: Match) -> Match:
        sx = frame.reference_size[0] / self.source_size[0]
        sy = frame.reference_size[1] / self.source_size[1]
        return Match(match.name, round(match.x * sx), round(match.y * sy),
                     round(match.width * sx), round(match.height * sy), match.score)

    def find(self, frame: VisualFrame, name: str) -> Match | None:
        match = self._find(self._image(frame), name)
        return self._reference(frame, match) if match else None

    def dungeon(self, frame: VisualFrame, name: str) -> Match | None:
        if name not in self.dungeons:
            raise AutomationError(f"没有该魔都副本的视觉模板：{name}")
        match = self.find(frame, self.dungeons[name])
        return Match(name, match.x, match.y, match.width, match.height, match.score) if match else None

    @staticmethod
    def _gold(patch: np.ndarray) -> np.ndarray:
        return cv2.inRange(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV),
                           (19, 100, 120), (42, 255, 255))

    @staticmethod
    def _word(mask: np.ndarray, *, remove_bottom_border: bool = False) -> np.ndarray | None:
        """Discard flag edges, but preserve every full visible name character."""
        result = np.zeros_like(mask)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            # Punctuation is part of a full name too: do not discard a dot
            # merely because it is shorter than a digit or CJK character.
            if area < 2:
                continue
            if remove_bottom_border and y + height >= mask.shape[0] - 1:
                continue
            if x == 0 or x + width >= mask.shape[1]:
                return None
            result[labels == index] = 255
        return _trim(result)

    def _owner(self, player_name: str, word: np.ndarray, place: str) -> str:
        self.validate_player(player_name)
        return "self" if _same_name(word, self.players[player_name][place]) else "other"

    def map_owner(self, frame: VisualFrame, name: str, player_name: str) -> str | None:
        """Inspect the flag under this dungeon; unknown pixels are not empty."""
        self.validate_player(player_name)
        image = self._image(frame)
        template = self.dungeons.get(name)
        if template is None:
            raise AutomationError(f"没有该魔都副本的视觉模板：{name}")
        label = self._find(image, template)
        if label is None:
            return None
        x, bottom = label.center[0], label.y + label.height
        left, right = x - 135, x + 135
        roi = image[bottom:bottom + 48, left:right]
        mask = self._gold(roi)
        # The calibrated green plaque has no yellow pixels below it until a
        # guardian flag is present. Allow a few antialiased specks, not a partial flag.
        pixels = cv2.countNonZero(mask)
        if pixels <= 12:
            inner = roi[6:27, 90:180]
            hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
            # A covered gold rim must not turn a still-visible brown flag
            # interior (or a black/flat obstruction) into a vacant dungeon.
            brown = cv2.inRange(hsv, (7, 100, 15), (39, 255, 150))
            if cv2.countNonZero(brown) > 90 or float(np.std(inner)) < 5:
                return None
            return "empty"
        if pixels < 170:
            return None
        word = self._word(mask[4:32], remove_bottom_border=True)
        if word is None:
            return None
        return self._owner(player_name, word, "map")

    def guardian_owner(self, frame: VisualFrame, player_name: str) -> str | None:
        self.validate_player(player_name)
        image = self._image(frame)
        if self._find(image, "guardian_panel") is None:
            return None
        patch = image[225:266, 548:738]
        mask = self._gold(patch)
        raw_word = _trim(mask)
        if raw_word is None:
            return None
        # Empty ownership is explicitly shown as a short dash, with a second
        # white dash in the guard-duration field. A missing/occluded name is unknown.
        if raw_word.shape[0] <= 5 and 8 <= raw_word.shape[1] <= 22:
            duration = image[287:319, 550:738]
            neutral = (duration.max(axis=2) - duration.min(axis=2) < 30)
            bright = duration.min(axis=2) > 170
            dash = _trim((neutral & bright).astype(np.uint8) * 255)
            if dash is not None and dash.shape[0] <= 5 and 8 <= dash.shape[1] <= 22:
                return "empty"
            return None
        word = self._word(mask)
        if word is None:
            return None
        return self._owner(player_name, word, "guardian")
