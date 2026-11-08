"""Conservative recognition for the dragon assistant and login recovery.

The supplied captures contain the entire 752 x 423 game viewport.  Match at
that scale and return VisualFrame reference coordinates, so a desktop crop
and native 16:9 ADB capture share one profile without device coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .adb import AutomationError
from .vision import Match, VisualFrame


class DragonVision:
    def __init__(self, directory: Path):
        self.directory = directory
        self.profile = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
        self.source_size = tuple(self.profile["source_size"])
        self.images = {}
        self.variants = {}
        self.alternatives = {}
        for name, entry in self.profile["templates"].items():
            images = [self._read(filename)
                      for filename in [entry["file"], *entry.get("variants", [])]]
            self.images[name], self.variants[name] = images[0], images[1:]
            self.alternatives[name] = [
                (self._read(alternative["file"]), {**entry, **alternative})
                for alternative in entry.get("alternatives", [])
            ]

    def _read(self, filename: str) -> np.ndarray:
        path = self.directory / filename
        template = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        if template is None:
            raise AutomationError(f"无法读取屠龙战视觉模板：{path}")
        return template

    def _raw(self, image: np.ndarray, name: str) -> Match | None:
        entry = self.profile["templates"][name]
        candidates = [(template, entry)
                      for template in [self.images[name], *self.variants[name]]]
        for template, config in [*candidates, *self.alternatives[name]]:
            left, top, right, bottom = config["search"]
            source = image[top:bottom, left:right]
            height, width = template.shape[:2]
            if source.shape[0] < height or source.shape[1] < width:
                continue
            scores = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(scores)
            if score < config.get("threshold", .90):
                continue
            patch = source[y:y + height, x:x + width]
            if float(np.mean(cv2.absdiff(patch, template))) > config.get("max_error", 16):
                continue
            return Match(name, left + x, top + y, width, height, score)
        return None

    def _landmark(self, image: np.ndarray, name: str) -> Match | None:
        match = self._raw(image, name)
        if match is None and name == "town":
            # The eye remains usable when a cloud hides the entry label or
            # the independent family-building landmark.
            eye = self._raw(image, "town_eye")
            if eye is not None:
                return Match(name, eye.x, eye.y, eye.width, eye.height, eye.score)
        return match

    def find(self, frame: VisualFrame, name: str) -> Match | None:
        image = cv2.resize(frame.image, self.source_size, interpolation=cv2.INTER_AREA)
        entry = self.profile["templates"][name]
        for guard in entry.get("guards", []):
            if self._landmark(image, guard) is None:
                return None
        match = self._landmark(image, name)
        if match is None:
            return None
        sx = frame.reference_size[0] / self.source_size[0]
        sy = frame.reference_size[1] / self.source_size[1]
        return Match(match.name, round(match.x * sx), round(match.y * sy),
                     round(match.width * sx), round(match.height * sy), match.score)
