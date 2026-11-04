"""Reusable image matching, independent of any game's workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .adb import AutomationError, Screenshot

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise AutomationError(
        "视觉依赖未安装。请使用 .\\.venv\\Scripts\\python.exe，"
        "或运行 python -m pip install -r requirements-vision.txt。"
    ) from exc


@dataclass(frozen=True)
class Match:
    name: str
    x: int
    y: int
    width: int
    height: int
    score: float

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2


class VisualFrame:
    def __init__(self, image: np.ndarray, reference_size: tuple[int, int] = (1920, 1080)):
        if image is None or image.size == 0:
            raise AutomationError("无法解码截图。")
        self.original_height, self.original_width = image.shape[:2]
        self.reference_size = reference_size
        width, height = reference_size
        if abs(self.original_width / self.original_height - width / height) > .025:
            raise AutomationError("当前画面不是 16:9 横屏；请恢复横屏后运行视觉脚本。")
        self.image = cv2.resize(image, reference_size, interpolation=cv2.INTER_LINEAR)

    @classmethod
    def from_screenshot(cls, screenshot: Screenshot) -> VisualFrame:
        return cls(cv2.imdecode(np.frombuffer(screenshot.data, dtype=np.uint8), cv2.IMREAD_COLOR))

    def device_point(self, point: tuple[int, int]) -> tuple[int, int]:
        width, height = self.reference_size
        x = min(self.original_width - 1, max(0, round(point[0] * self.original_width / width)))
        y = min(self.original_height - 1, max(0, round(point[1] * self.original_height / height)))
        return x, y

    def count_hsv_pixels(self, search: tuple | list, lower: tuple | list, upper: tuple | list) -> int:
        left, top, right, bottom = search
        left, top = max(0, int(left)), max(0, int(top))
        right, bottom = min(self.image.shape[1], int(right)), min(self.image.shape[0], int(bottom))
        patch = self.image[top:bottom, left:right]
        if patch.size == 0:
            return 0
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        return cv2.countNonZero(cv2.inRange(hsv, tuple(lower), tuple(upper)))

    def color_components(self, name: str, search: tuple | list, hsv_ranges: list) -> list[Match]:
        left, top, right, bottom = search
        left, top = max(0, int(left)), max(0, int(top))
        right, bottom = min(self.image.shape[1], int(right)), min(self.image.shape[0], int(bottom))
        patch = self.image[top:bottom, left:right]
        if patch.size == 0:
            return []
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        mask = np.zeros(patch.shape[:2], np.uint8)
        for lower, upper in hsv_ranges:
            mask |= cv2.inRange(hsv, tuple(lower), tuple(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            fill = cv2.contourArea(contour) / (width * height)
            found.append(Match(name, left + x, top + y, width, height, fill))
        return found

    def save(self, path: Path, matches: list[Match] = ()) -> Path:
        output = self.image.copy()
        for match in matches:
            cv2.rectangle(output, (match.x, match.y), (match.x + match.width, match.y + match.height), (0, 255, 0), 2)
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise AutomationError("无法保存视觉诊断截图。")
        encoded.tofile(str(path))
        return path


class TemplateBank:
    def __init__(self, directory: Path):
        self.directory = directory
        self.profile = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
        self.images = {}
        for name, entry in self.profile["templates"].items():
            path = directory / entry["file"]
            image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise AutomationError(f"无法读取视觉模板：{path}")
            self.images[name] = image

    def find(self, frame: VisualFrame, name: str, search: tuple | list | None = None) -> Match | None:
        matches = self.find_all(frame, name, search, limit=1)
        return matches[0] if matches else None

    def find_all(
        self, frame: VisualFrame, name: str, search: tuple | list | None = None, limit: int = 40,
    ) -> list[Match]:
        entry = self.profile["templates"][name]
        template = self.images[name]
        height, width = template.shape[:2]
        left, top, right, bottom = search if search is not None else entry["search"]
        left, top = max(0, int(left)), max(0, int(top))
        right, bottom = min(frame.image.shape[1], int(right)), min(frame.image.shape[0], int(bottom))
        source = frame.image[top:bottom, left:right]
        if source.shape[0] < height or source.shape[1] < width:
            return []
        scores = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
        found = []
        # Correlation alone also matches a grey or dimmed button. Compare color
        # pixels as well so disabled controls and modal backgrounds are rejected.
        for _ in range(limit * 8):
            _, score, _, location = cv2.minMaxLoc(scores)
            if score < entry.get("threshold", .86):
                break
            x, y = location
            patch = source[y:y + height, x:x + width]
            error = float(np.mean(cv2.absdiff(patch, template)))
            if error <= entry.get("max_error", 30):
                found.append(Match(name, left + x, top + y, width, height, score))
                if len(found) >= limit:
                    break
            # Suppress nearby maxima to avoid returning the same icon twice.
            scores[max(0, y - height // 2):y + height // 2 + 1,
                   max(0, x - width // 2):x + width // 2 + 1] = -1
        return sorted(found, key=lambda match: (match.y, match.x))
