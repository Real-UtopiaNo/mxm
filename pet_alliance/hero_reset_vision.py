"""Recognition for completed hero dungeons, with no dependency on red badges."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .adb import AutomationError
from .vision import Match, TemplateBank, VisualFrame


class HeroResetVision:
    def __init__(self, directory: Path):
        self.bank = TemplateBank(directory)
        self.hero_bank = TemplateBank(directory.parent / "daily_hero")
        self.dungeons = {(entry["region"], entry["name"]): entry["template"]
                         for entry in self.bank.profile.get("dungeons", [])}

    def validate_dungeon(self, region_name: str, dungeon_name: str) -> None:
        """Reject unsupported exact names before navigation or paid inputs."""
        if (region_name, dungeon_name) not in self.dungeons:
            raise AutomationError(
                f"没有副本“{region_name}-{dungeon_name}”的完整名称视觉模板；请先采集该副本名牌。"
            )

    def named_entrance(self, frame: VisualFrame, region_name: str, dungeon_name: str) -> Match | None:
        """Locate one configured full name on a freshly verified local map.

        The caller first selects its region on the world map. A missing name
        never falls back to a nearby green plaque or an unfinished red badge.
        """
        self.validate_dungeon(region_name, dungeon_name)
        if (not self.hero_bank.find(frame, "world_button")
                or self.hero_bank.find(frame, "world_anchor")
                or self.hero_bank.find(frame, "panel_close")):
            return None
        match = self.bank.find(frame, self.dungeons[(region_name, dungeon_name)])
        return Match(dungeon_name, match.x, match.y, match.width, match.height, match.score) if match else None

    def find(self, frame: VisualFrame, name: str) -> Match | None:
        if name in {"reset", "reset_disabled"}:
            if not self.hero_bank.find(frame, "hero_dragon") or not self.hero_bank.find(frame, "panel_close"):
                return None
        elif name in {"reset_dialog", "confirm_reset"}:
            # A common "提示" title is also used by the arena. Require the
            # actual reset question and its daily limit, independently of the
            # price so an unknown price can be reported without confirming it.
            # Three price digits wrap the full stop to the next line, changing
            # the whole question layout without moving the dialog controls.
            if not all(self.bank.find(frame, anchor) for anchor in (
                "reset_dialog", "reset_dialog_action",
            )):
                return None
            bodies = self.bank.profile.get("reset_dialog_bodies", ["reset_dialog_body"])
            if not any(self.bank.find(frame, body) for body in bodies):
                return None
        return self.bank.find(frame, name)

    def hero_state(self, frame: VisualFrame) -> str | None:
        """Classify a visible hero panel without inferring progress from reset.

        The same red defeat stamp appears on every completed stage, regardless
        of its monster. All five independent stage slots must show that stamp
        before this method returns ``cleared``. A grey reset button only says
        that the reset allowance was used; it is not evidence of completion.
        """
        if (not self.hero_bank.find(frame, "hero_dragon")
                or not self.hero_bank.find(frame, "panel_close")):
            return None
        if self.find(frame, "reset_dialog"):
            return None
        slots = self.bank.profile.get("hero_clear_slots", [])
        if len(slots) != 5:
            return None
        if all(self.bank.find(frame, "cleared_stamp", search=slot) for slot in slots):
            return "cleared"
        if self.hero_bank.find(frame, "challenge"):
            return "incomplete"
        return None

    def reset_cost(self, frame: VisualFrame) -> int | None:
        """Read a verified price from a reset prompt; never infer an amount."""
        if self.find(frame, "reset_dialog") is None:
            return None
        prices = set()
        for entry in self.bank.profile["reset_costs"]:
            match = self.bank.find(frame, entry["template"])
            if match is None:
                continue
            x1, y1, x2, y2 = entry["suffix_search"]
            suffix = (match.x + x1, match.y + y1, match.x + x2, match.y + y2)
            # Matching x10/x50/x300 must reject x100/x500/x3000 prefixes. The
            # supplied complete price is followed by a white full stop and
            # empty background, with no more yellow number glyphs.
            if frame.count_hsv_pixels(suffix, (20, 100, 140), (40, 255, 255)) > 8:
                continue
            prices.add(entry["diamonds"])
        return next(iter(prices)) if len(prices) == 1 else None

    def entrances(self, frame: VisualFrame) -> list[Match]:
        """Return fresh, clickable green dungeon nameplates on the local map."""
        if (not self.hero_bank.find(frame, "world_button")
                or self.hero_bank.find(frame, "world_anchor")
                or self.hero_bank.find(frame, "panel_close")):
            return []
        return self._entrances(frame)

    def _entrances(self, frame: VisualFrame) -> list[Match]:
        hsv = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (38, 55, 35), (95, 255, 190))
        found = []

        def fraction(box):
            left, top, right, bottom = box
            patch = green[max(0, top):min(green.shape[0], bottom),
                          max(0, left):min(green.shape[1], right)]
            return cv2.countNonZero(patch) / patch.size if patch.size else 0

        def add(match):
            if not any(np.hypot(match.center[0] - other.center[0],
                                match.center[1] - other.center[1]) < 50 for other in found):
                found.append(match)

        # The fold and the green body remain identifiable after stars replace
        # the unfinished badge. Text and red/yellow notifications are ignored.
        for fold in self.bank.find_all(frame, "entrance_left"):
            x, y = fold.x, fold.y
            if fraction((x + 19, y - 21, x + 177, y + 6)) > .40:
                add(Match("dungeon_entrance", x - 5, y - 25, 212, 69, fold.score))

        # Chat may cover a fold, while enough of the green scroll survives.
        # Conversely some map scenery joins a plate's body; the fold path
        # above covers those cases. Restrict to the map, excluding fixed UI.
        left, top, right, bottom = 115, 200, 1710, 1020
        mask = cv2.morphologyEx(green[top:bottom, left:right], cv2.MORPH_CLOSE,
                               np.ones((3, 12), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            fill = cv2.contourArea(contour) / (width * height)
            if 190 <= width <= 230 and 60 <= height <= 85 and fill > .45:
                add(Match("dungeon_entrance", x + left, y + top, width, height, fill))

        # Freyes is behind several chat lines and blends into green scenery.
        # Reuse the unique neighbouring landmark, requiring surviving green
        # plate pixels rather than the red badge (which completed runs lack).
        for entry in self.bank.profile.get("obscured_entrances", []):
            anchor = self.hero_bank.find(frame, entry["anchor"])
            if anchor is None:
                continue
            x1, y1, x2, y2 = entry["offset"]
            box = (anchor.x + x1, anchor.y + y1, anchor.x + x2, anchor.y + y2)
            if fraction(box) > .25:
                add(Match("dungeon_entrance", box[0], box[1], x2 - x1, y2 - y1, anchor.score))
        return sorted(found, key=lambda match: (match.y, match.x))
