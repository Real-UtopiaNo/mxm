"""Read a guardian's elapsed duration with Windows OCR and calculate its end."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unicodedata

from .dragon_schedule import BEIJING
from .vision import VisualFrame, cv2


GUARD_SECONDS = 6 * 60 * 60
# The value to the right of 守护时间, in VisualFrame's 1920 x 1080 coordinates.
# There is room for hours as well as minutes and seconds; the label is excluded.
GUARD_TIME_ROI = (765, 395, 1125, 455)


def parse_guard_duration(text: str) -> int | None:
    """Parse the complete duration, rejecting incomplete or ambiguous OCR text."""
    if not isinstance(text, str):
        return None
    value = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    value = re.sub(r"^守护时间[:：]?", "", value)
    if not value:
        return None
    colon = re.fullmatch(r"([0-9]{1,3}):([0-9]{1,2}):([0-9]{1,2})", value)
    if colon:
        hours, minutes, seconds = map(int, colon.groups())
        return hours * 3600 + minutes * 60 + seconds if minutes < 60 and seconds < 60 else None
    units = re.fullmatch(
        r"(?:(?P<hours>[0-9]{1,3})(?:小时|时))?"
        r"(?:(?P<minutes>[0-9]{1,4})(?:分钟|分))?"
        r"(?:(?P<seconds>[0-9]{1,5})(?:秒钟|秒))?", value,
    )
    if units is None or not any(units.groups()):
        return None
    hours, minutes, seconds = (int(part or 0) for part in units.groups())
    if units["hours"] is not None and minutes >= 60:
        return None
    if (units["hours"] is not None or units["minutes"] is not None) and seconds >= 60:
        return None
    return hours * 3600 + minutes * 60 + seconds


def guard_expiration(now: datetime, elapsed_seconds: int) -> datetime:
    """Return the actual six-hour end in Beijing time, including past ends."""
    if isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, int) or elapsed_seconds < 0:
        raise ValueError("守护时长必须是非负整数秒。")
    current = now.replace(tzinfo=BEIJING) if now.tzinfo is None else now.astimezone(BEIJING)
    return current + timedelta(seconds=GUARD_SECONDS - elapsed_seconds)


class GuardianTimeReader:
    """Use the installed Chinese Windows OCR engine, without a network service."""

    def __init__(self) -> None:
        self.last_text = ""
        self.last_error = ""

    def read_text(
        self, frame: VisualFrame, roi: tuple[int, int, int, int], *, high_contrast: bool = False,
    ) -> str | None:
        """Read one bounded crop in the frame's reference coordinates."""
        self.last_text = ""
        self.last_error = ""
        left, top, right, bottom = roi
        left, top = max(0, left), max(0, top)
        right, bottom = min(frame.image.shape[1], right), min(frame.image.shape[0], bottom)
        if right <= left or bottom <= top:
            self.last_error = "OCR 区域超出截图。"
            return None
        patch = frame.image[top:bottom, left:right]
        if patch.size == 0:
            self.last_error = "守护时间截图为空。"
            return None
        if high_contrast:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            _, patch = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            patch = cv2.copyMakeBorder(patch, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        # Enlarging the bounded value crop improves the small game font.
        patch = cv2.resize(patch, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        ok, encoded = cv2.imencode(".png", patch)
        if not ok:
            self.last_error = "无法编码守护时间截图。"
            return None
        powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
        helper = Path(__file__).with_name("windows_ocr.ps1")
        try:
            with tempfile.TemporaryDirectory(prefix="mxm_guard_time_") as directory:
                image_path = Path(directory) / "guard_time.png"
                image_path.write_bytes(encoded.tobytes())
                result = subprocess.run(
                    [str(powershell), "-NoLogo", "-NoProfile", "-NonInteractive",
                     "-ExecutionPolicy", "Bypass", "-File", str(helper), "-ImagePath", str(image_path)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
                )
                if result.returncode:
                    self.last_error = result.stderr.decode("utf-8", errors="replace").strip()
                    return None
                response = json.loads(result.stdout.decode("utf-8-sig"))
                self.last_text = response.get("text", "")
                if not isinstance(self.last_text, str):
                    self.last_text = ""
                    self.last_error = "Windows OCR 返回的文字格式无效。"
                    return None
        except (OSError, subprocess.TimeoutExpired, ValueError, AttributeError) as exc:
            self.last_error = str(exc)
            return None
        return self.last_text

    def read(self, frame: VisualFrame) -> int | None:
        self.read_text(frame, GUARD_TIME_ROI)
        if self.last_error:
            return None
        elapsed = parse_guard_duration(self.last_text)
        if elapsed is None:
            # A second, black-on-white reading can recover small leading zeros.
            # Never infer a missing digit from an otherwise incomplete result.
            self.read_text(frame, GUARD_TIME_ROI, high_contrast=True)
            if self.last_error:
                return None
            elapsed = parse_guard_duration(self.last_text)
        if elapsed is None:
            self.last_error = f"无法解析守护时间：{self.last_text!r}"
        return elapsed
