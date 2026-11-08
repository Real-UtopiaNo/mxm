from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import time

from .adb import AutomationError
from .cooperation import TaskInterrupted, checkpoint
from .runtime import ScriptContext
from .run_logs import RunLogOwner
from .vision import Match, TemplateBank, VisualFrame


class ClickTimeoutError(AutomationError):
    """The observation deadline expired without confirming a click's result."""


@dataclass(frozen=True)
class HeroOptions:
    max_retries: int = 10
    poll_interval: float = .35
    min_click_interval: float = 1.0
    click_retry_interval: float = 2.0
    max_click_retries: int = 3
    transition_timeout: float = 30
    max_runtime: float = 7200
    max_dungeons: int = 0
    points_arena_enabled: bool = True
    target_players: list[str] = field(default_factory=lambda: ["Hello", "Hell", "coco"])

    def __post_init__(self):
        if type(self.points_arena_enabled) is not bool:
            raise AutomationError("daily_hero.points_arena_enabled 必须是布尔值（true 或 false）。")
        if (not isinstance(self.target_players, list) or not self.target_players
                or any(not isinstance(name, str) or not name.strip() for name in self.target_players)):
            raise AutomationError("daily_hero.target_players 必须是包含非空玩家名的字符串列表。")
        for name in ("max_retries", "max_dungeons", "max_click_retries"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise AutomationError(f"daily_hero.{name} 必须是非负整数。")
        for name in ("poll_interval", "min_click_interval", "click_retry_interval", "transition_timeout", "max_runtime"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise AutomationError(f"daily_hero.{name} 必须是有限的正数。")


@dataclass
class RetryBudget:
    max_retries: int
    failures: int = 0

    def failed(self) -> bool:
        """True allows another attempt: initial attempt plus max_retries."""
        self.failures += 1
        return self.failures <= self.max_retries


@dataclass
class HeroReport:
    status: str = "running"
    completed: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    victories: int = 0
    failures: int = 0
    arena_visits: int = 0
    arena_battles: int = 0
    arena_no_tickets: int = 0
    points_arena_pending: bool = False
    reason: str = ""


class DailyHero(RunLogOwner):
    completion_reason = "未发现尚需处理且带感叹号的推荐等级区域。"
    dungeon_entry_offset = (-65, 10)

    def __init__(self, ctx: ScriptContext, *, options: HeroOptions | None = None,
                 run_name: str = "Daily_Hero"):
        self.ctx = ctx
        self.options = options if options is not None else HeroOptions(**ctx.settings.get("daily_hero", {}))
        self.run_name = run_name
        self.bank = TemplateBank(ctx.project_root / "assets" / "daily_hero")
        self._init_run_log(ctx, run_name)
        self.report = HeroReport()
        self.started = time.monotonic()
        self.frame: VisualFrame | None = None
        self.actions = 0
        self.last_click_at: float | None = None

    def event(self, kind: str, **details) -> None:
        event = {"time": datetime.now().astimezone().isoformat(), "event": kind, **details}
        with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def observe(self) -> VisualFrame:
        if time.monotonic() - self.started > self.options.max_runtime:
            raise AutomationError(f"{self.run_name} 已达到最长运行时间。")
        self.frame = VisualFrame.from_screenshot(self.ctx.capture())
        return self.frame

    def find(self, name: str, frame: VisualFrame | None = None) -> Match | None:
        return self.bank.find(frame or self.frame, name)

    def town_find(self, name: str, frame: VisualFrame | None = None) -> Match | None:
        if not hasattr(self, "town_vision"):
            from .arena_vision import PointsArenaVision

            self.town_vision = PointsArenaVision(self.ctx.project_root / "assets" / "points_arena")
        return self.town_vision.find(frame or self.frame, name)

    def return_to_town(self) -> None:
        frame = self.observe()
        if self.town_find("town_page", frame):
            return
        self.world()
        self.click_until(
            lambda f: self.town_find("town", f) if self.is_world(f) else None,
            lambda f: self.town_find("town_page", f), "运行完成 → 返回天星镇", timeout=300,
        )

    def tap(self, match: Match, description: str, offset: tuple[int, int] = (0, 0)) -> None:
        x, y = match.center
        point = self.frame.device_point((x + offset[0], y + offset[1]))
        self.ctx.log.info("%s [匹配 %.3f]", description, match.score)
        self.event("tap", description=description, template=match.name, score=round(match.score, 4), point=point)
        self.ctx.tap(*point)
        self.last_click_at = time.monotonic()
        self.actions += 1
        if self.actions > 4000:
            raise AutomationError("动作数量异常，已停止并保存现场。")

    def click_until(self, locate, arrived, description: str, *, offset: tuple[int, int] = (0, 0),
                    timeout: float | None = None, retry_by_control: bool = False):
        """Click once, observe the destination, and retry only a visible source control."""
        deadline = time.monotonic() + (self.options.transition_timeout if timeout is None else timeout)
        attempts = 0
        control_attempts: dict[str, int] = {}
        next_attempt_at = time.monotonic()
        while True:
            frame = self.observe()
            result = arrived(frame)
            if result:
                if attempts:
                    self.event("click_confirmed", description=description, attempts=attempts)
                return result
            now = time.monotonic()
            if now >= deadline:
                raise ClickTimeoutError(f"点击“{description}”后未确认目标页面，已停止并保存现场。")
            # Re-locate on every attempt. Once the source disappears (e.g. a
            # loading screen), only observe; never tap the previous coordinates.
            if now >= next_attempt_at and (
                self.last_click_at is None or now - self.last_click_at >= self.options.min_click_interval
            ):
                control = locate(frame)
                if control is not None:
                    count = control_attempts.get(control.name, 0) if retry_by_control else attempts
                    if count > self.options.max_click_retries:
                        raise AutomationError(
                            f"“{description}”补点 {self.options.max_click_retries} 次后仍未成功，已停止。"
                        )
                    if count:
                        self.ctx.log.info("页面尚未切换，重新定位后补点：%s（%d/%d）",
                                          description, count, self.options.max_click_retries)
                        self.event("click_retry", description=description, retry=count)
                    self.tap(control, description, offset)
                    control_attempts[control.name] = count + 1
                    attempts += 1
                    next_attempt_at = self.last_click_at + self.options.click_retry_interval
            self.ctx.wait(self.options.poll_interval)

    def wait_for(self, predicate, description: str):
        deadline = time.monotonic() + self.options.transition_timeout
        while True:
            frame = self.observe()
            result = predicate(frame)
            if result:
                return result
            if time.monotonic() >= deadline:
                raise AutomationError(f"等待“{description}”超时；已停止，避免在未知页面继续点击。")
            self.ctx.wait(self.options.poll_interval)

    def is_world(self, frame: VisualFrame) -> bool:
        return self.find("world_anchor", frame) is not None

    def world_ready(self, frame: VisualFrame) -> bool:
        return self.is_world(frame) and all(
            self.find(region["template"], frame) is not None
            for region in self.bank.profile["regions"]
        )

    def is_map(self, frame: VisualFrame) -> bool:
        # The map also exists behind the dungeon modal; require no close control.
        return (
            not self.is_world(frame) and self.find("panel_close", frame) is None
            and self.find("world_button", frame) is not None
        )

    def adventure_map(self) -> bool:
        """Reach the local map; return False when the world map is already open."""
        frame = self.observe()
        if self.is_world(frame):
            self.wait_for(self.world_ready, "大地图区域标签显示完整")
            return False
        if self.find("panel_close", frame):
            self.click_until(lambda f: self.find("panel_close", f), self.is_map, "关闭副本面板")
        elif not self.is_map(frame):
            # Clouds in town can hide the label for longer than the ordinary
            # transition timeout. Keep observing and re-locating; never require
            # the first frame to contain Adventure or click its old position.
            self.ctx.log.info("等待识别天星镇“冒险”入口，最多 5 分钟。")
            self.click_until(
                lambda f: self.find("adventure", f), self.is_map, "天星镇 → 冒险", timeout=300,
            )
        return True

    def world(self) -> None:
        if not self.adventure_map():
            return
        self.click_until(
            lambda f: self.find("world_button", f) if self.is_map(f) else None,
            self.world_ready, "打开大地图",
        )

    def regions(self, frame: VisualFrame) -> list[tuple[dict, Match]]:
        found = []
        for region in sorted(self.bank.profile["regions"], key=lambda region: region["min_level"]):
            label = self.find(region["template"], frame)
            if label is None:
                continue
            left, top, right, bottom = region["marker_search_offset"]
            search = (label.x + left, label.y + top, label.x + right, label.y + bottom)
            if self.bank.find(frame, "world_marker", search):
                found.append((region, label))
        return found

    def enter_region(self, region: dict) -> None:
        frame = self.observe()
        current = self.current_region(frame) if self.is_world(frame) else None
        if current and current[0]["key"] == region["key"]:
            # Selecting the current region again does nothing in the game.
            self.click_until(
                lambda f: self.find("world_close", f) if self.is_world(f) else None,
                self.is_map, f"关闭大地图，返回当前区域：{region['name']}",
            )
            return
        # Label templates include the recommended level, which is not always
        # clickable. Keep each region's hit point relative to its visual match.
        self.click_until(
            lambda f: self.find(region["template"], f) if self.is_world(f) else None,
            self.is_map, f"进入区域：{region['name']}",
            offset=tuple(region.get("click_offset", (0, 0))),
        )

    def current_region(self, frame: VisualFrame) -> tuple[dict, Match] | None:
        # The current-region diamond can cover its exclamation mark. When no
        # flagged regions remain, inspect that region's dungeon flags once.
        pointer = self.find("world_pointer", frame)
        if pointer is None:
            # Its glow/background varies between regions. The red diamond's
            # color and tall silhouette remain visible when template matching fails.
            components = frame.color_components(
                "world_pointer", self.bank.profile["templates"]["world_pointer"]["search"],
                [((0, 130, 140), (10, 255, 255)), ((170, 130, 140), (180, 255, 255))],
            )
            diamonds = [m for m in components if 20 < m.width < 65 and 40 < m.height < 95
                        and 1.3 < m.height / m.width < 3 and m.score * m.width * m.height > 500]
            if len(diamonds) != 1:
                return None
            pointer = diamonds[0]
        candidates = []
        px, py = pointer.center
        for region in self.bank.profile["regions"]:
            label = self.find(region["template"], frame)
            if label:
                x, y = label.center
                if abs(x - px) < 110 and 0 < y - py < 220:
                    candidates.append((math.dist((px, py), (x, y)), region, label))
        if not candidates:
            return None
        _, region, label = min(candidates, key=lambda entry: entry[0])
        return region, label

    def markers(self, frame: VisualFrame) -> list[Match]:
        found = self.bank.find_all(frame, "marker")
        # Chat text can cover a badge's white exclamation. Require a unique
        # nearby landmark and enough remaining badge color in its small ROI.
        for entry in self.bank.profile.get("obscured_markers", []):
            anchor = self.find(entry["anchor"], frame)
            if anchor is None:
                continue
            left, top, right, bottom = entry["badge_search_offset"]
            search = (anchor.x + left, anchor.y + top, anchor.x + right, anchor.y + bottom)
            pixels = frame.count_hsv_pixels(search, entry["hsv_lower"], entry["hsv_upper"])
            if pixels >= entry["min_pixels"]:
                marker = Match(entry["name"], search[0], search[1], right - left, bottom - top, anchor.score)
                if not any(self.near(marker, other) for other in found):
                    found.append(marker)
        return sorted(found, key=lambda match: (match.y, match.x))

    @staticmethod
    def near(first: Match, second: Match) -> bool:
        return math.dist(first.center, second.center) < 35

    def map_destination(self, frame: VisualFrame):
        return (
            self.is_map(frame) or self.find("panel_close", frame)
            or (self.find("reward_header", frame) and self.find("claim_reward", frame))
        )

    def return_to_map(self) -> None:
        for _ in range(3):
            self.wait_for(
                self.map_destination,
                "区域地图、副本面板或通关奖励",
            )
            if not self.find("reward_header"):
                break
            self.click_until(
                lambda f: self.find("claim_reward", f) if self.find("reward_header", f) else None,
                lambda f: not self.find("reward_header", f) and self.map_destination(f),
                "领取英雄级通关奖励",
            )
        else:
            raise AutomationError("通关奖励弹窗重复出现，已停止。")
        if self.find("panel_close"):
            self.click_until(lambda f: self.find("panel_close", f), self.is_map, "关闭副本面板，返回区域地图")

    def open_dungeon_panel(self, target: Match, resume: bool = False) -> None:
        # After a defeat, Exit may return to the existing hero panel. Continue
        # there to preserve cleared stages; some versions return to the map.
        frame = self.observe()
        if not (resume and self.find("panel_close", frame)):
            self.wait_for(self.is_map, "返回区域地图")
            markers = self.markers(self.frame)
            current = next((marker for marker in markers if self.near(marker, target)), None)
            if current is None:
                raise AutomationError("目标副本的未完成标记已消失，无法确认重试目标。")
            # The exclamation mark is attached to the right end of its nameplate.
            self.click_until(
                lambda f: next((m for m in self.markers(f) if self.near(m, target)), None) if self.is_map(f) else None,
                lambda f: self.find("panel_close", f),
                "进入带感叹号的副本", offset=self.dungeon_entry_offset,
            )

    def open_dungeon(self, target: Match, resume: bool = False) -> None:
        self.open_dungeon_panel(target, resume)
        frame = self.observe()
        if self.normal_mode_control(frame):
            self.click_until(
                self.normal_mode_control,
                lambda f: self.find("hero_dragon", f) and self.find("challenge", f),
                "切换英雄级模式",
            )
        self.wait_for(
            lambda f: self.find("hero_dragon", f) and self.find("challenge", f),
            "英雄级模式和挑战按钮",
        )
        self.click_until(
            lambda f: self.find("challenge", f) if self.find("hero_dragon", f) else None,
            self.battle_state, "英雄级挑战",
        )

    def normal_mode_control(self, frame: VisualFrame) -> Match | None:
        # The dragon's glow varies, so its color tolerance is slightly wider.
        # A visible close control confirms the panel is active, not dimmed
        # behind another modal. Check it again before every retry.
        return self.find("normal_dragon", frame) if self.find("panel_close", frame) else None

    def continue_dungeon(self, target: Match) -> None:
        """Resume this fight's remaining stages without purchasing a reset."""
        if self.is_map(self.observe()):
            self.open_segment_panel(target)
        DailyHero.open_dungeon(self, target, resume=True)

    def segment_has_remaining_stages(self, region: dict, target: Match,
                                     *, midpoint: bool = False) -> bool:
        """Only Moru's recognized midpoint may enter the special continuation."""
        if not midpoint:
            return False
        if region.get("key") not in {"west", "__current__"} and region.get("name") != "西部大陆":
            return False
        self.wait_for(self.map_destination, "退出战斗后的英雄面板、奖励或区域地图")
        # An actual dungeon reward is sufficient evidence of the final clear.
        if self.find("reward_header", self.frame) and self.find("claim_reward", self.frame):
            return False
        frame = self.frame
        if self.is_map(frame):
            self.open_segment_panel(target)
            frame = self.observe()
        if self.normal_mode_control(frame):
            self.click_until(
                self.normal_mode_control,
                lambda f: self.find("hero_dragon", f) and self.find("panel_close", f),
                "检查退出战斗后是否仍有英雄关卡",
            )
            frame = self.frame
        if self.find("hero_dragon", frame) and self.find("panel_close", frame):
            if self.find("challenge", frame):
                return True
            from .hero_reset_vision import HeroResetVision

            if not hasattr(self, "_completion_vision"):
                self._completion_vision = HeroResetVision(self.ctx.project_root / "assets" / "daily_hero_reset")
            if self._completion_vision.hero_state(frame) == "cleared":
                return False
            raise AutomationError("退出战斗后无法确认英雄关卡已全部通过，未计入副本完成数。")
        return False

    def open_segment_panel(self, target: Match) -> None:
        """Reopen the same map entrance, including when its badge disappeared."""
        frame = self.observe()
        if any(self.near(marker, target) for marker in self.markers(frame)):
            self.open_dungeon_panel(target, resume=True)
            return
        from .hero_reset_vision import HeroResetVision

        vision = HeroResetVision(self.ctx.project_root / "assets" / "daily_hero_reset")
        expected = (target.center[0] + self.dungeon_entry_offset[0],
                    target.center[1] + self.dungeon_entry_offset[1])

        def entrance(current):
            candidates = [m for m in vision.entrances(current) if math.dist(m.center, expected) < 45]
            if len(candidates) > 1:
                raise AutomationError("中场续战无法唯一确认原副本名牌，未点击其他副本。")
            return candidates[0] if candidates else None

        self.click_until(entrance, lambda f: self.find("panel_close", f), "重新进入原副本，检查剩余英雄关卡")

    def fight_dungeon(self, region: dict, target: Match) -> bool:
        budget = RetryBudget(self.options.max_retries)
        description = self.dungeon_description(region, target)
        resume = False
        continuation = False
        segments = 0
        while True:
            if continuation:
                self.continue_dungeon(target)
            else:
                self.open_dungeon(target, resume)
            while True:
                result = self.wait_for(self.battle_state, "战斗、胜利或失败界面")
                state, _ = result
                if state == "battle":
                    self.click_until(
                        lambda f: self.find("skip", f) if not self.result_state(f) else None,
                        self.result_state, "立即跳过战斗",
                    )
                    continue
                if state == "victory":
                    self.report.victories += 1
                    self.event("victory", **description)
                    # A slow transition may leave the previous result on screen.
                    # Do not count it again or send another Next tap.
                    self.click_until(
                        lambda f: self.find("next_stage", f) if self.find("victory", f) else None,
                        lambda f: self.battle_state(f) if not self.find("victory", f) else None,
                        "胜利 → 下一关",
                    )
                    continue
                if state == "complete":
                    self.report.victories += 1
                    result_frame = self.frame
                    midpoint = bool(self.find("moru_midpoint", result_frame))
                    segments += 1
                    self.click_until(
                        lambda f: self.find("exit_battle", f) if self.find("victory", f) else None,
                        self.map_destination, "当前连续战斗结束 → 退出战斗",
                    )
                    if self.segment_has_remaining_stages(region, target, midpoint=midpoint):
                        result_frame.save(self.run_dir / f"segment_exit_{len(self.report.completed) + 1}_{segments}.jpg")
                        self.event("hero_segment_exit", **description, segment=segments)
                        self.ctx.log.info("当前副本还有英雄关卡，再次挑战；暂不计入副本完成数。")
                        continuation = True
                        break
                    result_frame.save(self.run_dir / f"completed_{len(self.report.completed) + 1}.jpg")
                    self.return_to_map()
                    self.report.completed.append({**description, "failures": budget.failures})
                    self.event("dungeon_complete", **description, failures=budget.failures)
                    return True
                # A defeat screen has a replay button, which must never be used
                # as a retry. Exit and continue from the hero panel or map.
                self.report.failures += 1
                retry = budget.failed()
                self.frame.save(self.run_dir / f"failure_{self.report.failures}.jpg")
                self.event("defeat", **description, failures=budget.failures, retry=retry)
                self.click_until(
                    lambda f: self.find("exit_battle", f) if self.find("defeat", f) else None,
                    self.map_destination, "战斗失败 → 退出战斗",
                )
                if not retry:
                    self.return_to_map()
                    self.report.skipped.append({**description, "failures": budget.failures})
                    self.ctx.log.info("已用完 %d 次重试，放弃当前副本。", self.options.max_retries)
                    return False
                resume = True
                self.ctx.log.info("当前副本累计失败 %d 次，进行第 %d/%d 次重试。",
                                  budget.failures, budget.failures, self.options.max_retries)
                break

    def battle_state(self, frame: VisualFrame) -> tuple[str, Match] | None:
        # Results take precedence over the battle UI visible behind the overlay.
        if self.find("defeat", frame):
            exit_button = self.find("exit_battle", frame)
            return ("defeat", exit_button) if exit_button else None
        if self.find("victory", frame):
            # This ends the current continuous battle, not necessarily the
            # dungeon. Moru exits after stage three and resumes at stage four.
            if self.find("next_disabled", frame):
                exit_button = self.find("exit_battle", frame)
                return ("complete", exit_button) if exit_button else None
            next_button = self.find("next_stage", frame)
            return ("victory", next_button) if next_button else None
        if skip := self.find("skip", frame):
            return "battle", skip
        return None

    def result_state(self, frame: VisualFrame) -> tuple[str, Match] | None:
        state = self.battle_state(frame)
        return state if state and state[0] != "battle" else None

    def preview(self) -> None:
        frame = self.observe()
        if self.is_world(frame):
            candidates = self.regions(frame)
            self.ctx.log.info("[预演] 大地图上识别到未完成区域：%s", ", ".join(r["name"] for r, _ in candidates) or "无")
            if candidates:
                region, label = candidates[0]
                current = self.current_region(frame)
                if current and current[0]["key"] == region["key"]:
                    close = self.find("world_close", frame)
                    if close:
                        self.tap(close, "计划关闭大地图，检查当前区域")
                else:
                    self.tap(label, "计划进入首个未完成区域", tuple(region.get("click_offset", (0, 0))))
        elif adventure := self.find("adventure", frame):
            self.tap(adventure, "计划进入冒险地图")
        elif self.is_map(frame):
            markers = self.markers(frame)
            self.ctx.log.info("[预演] 当前地图识别到 %d 个未完成副本。", len(markers))
        else:
            raise AutomationError("未识别到天星镇或地图，请检查诊断截图。")
        self.frame.save(self.run_dir / "preview.jpg")
        self.report.status = "preview"

    def run_points_arena(self, region: dict) -> None:
        if not self.options.points_arena_enabled:
            self.report.points_arena_pending = False
            return
        from .points_arena import PointsArena

        before = self.observe()
        self.resolve_region(region)
        self.report.arena_visits += 1
        PointsArena(self).run()
        # Adventure normally reopens the last local map. Continue there without
        # another world-map detour, retaining the caller's visited markers.
        if self.adventure_map() and self.same_map(before, self.frame):
            self.event("arena_resumed_local_map", region=region["name"])
            return
        # A different map cannot inherit this region's visited coordinates.
        self.world()
        self.enter_region(region)

    @staticmethod
    def same_map(first: VisualFrame, second: VisualFrame) -> bool:
        import cv2

        # Compare the map's scenery above chat, excluding account and toolbar
        # overlays. Small marker changes and animations may differ after battle.
        before = cv2.resize(first.image[190:820, 160:1660], (320, 134), interpolation=cv2.INTER_AREA)
        after = cv2.resize(second.image[190:820, 160:1660], (320, 134), interpolation=cv2.INTER_AREA)
        correlation = float(cv2.matchTemplate(before, after, cv2.TM_CCOEFF_NORMED)[0, 0])
        error = sum(cv2.mean(cv2.absdiff(before, after))[:3]) / 3
        return correlation >= .9 and error <= 25

    def resolve_region(self, region: dict) -> None:
        if region["key"] != "__current__":
            return
        # Local maps do not display their region name. Defer identification
        # until the world map is needed for another region or an arena visit.
        self.world()
        current, _ = self.wait_for(self.current_region, "确认刚处理的当前区域")
        region.update(current)
        for entry in self.report.completed + self.report.skipped:
            if entry["region"] == "当前区域":
                entry["region"] = region["name"]
        self.event("current_region_identified", region=region["name"], key=region["key"])

    def starting_region(self) -> dict | None:
        return {"key": "__current__", "name": "当前区域"} if self.adventure_map() else None

    def region_work_key(self, region: dict) -> str:
        """Identify scan work independently of the map used for navigation."""
        return region["key"]

    def dungeon_targets(self, region: dict, frame: VisualFrame) -> list[Match]:
        return self.markers(frame)

    def dungeon_description(self, region: dict, target: Match) -> dict:
        return {"region": region["name"], "marker": target.center}

    def region_candidates(self, frame: VisualFrame, processed: set[str]) -> list[tuple[dict, Match]]:
        candidates = [(r, m) for r, m in self.regions(frame) if r["key"] not in processed]
        if not candidates:
            current = self.current_region(frame)
            if current and current[0]["key"] not in processed:
                self.ctx.log.info("当前区域的菱形标志可能遮住感叹号，检查其副本标记。")
                candidates = [current]
        return candidates

    def finish(self) -> None:
        """Finish successful navigation before recording the final report."""

    def run(self) -> HeroReport:
        self.ctx.log.info("%s 开始；诊断目录：%s", self.run_name, self.run_dir)
        try:
            if self.ctx.dry_run:
                self.preview()
                return self.report
            resume = getattr(self.ctx, "resume_data", {})
            completed_since_arena = resume.get("completed_since_arena", 0) if isinstance(resume, dict) else 0
            if type(completed_since_arena) is not int or not 0 <= completed_since_arena <= 2:
                completed_since_arena = 0
            self.report.points_arena_pending = resume.get("points_arena_pending") is True if isinstance(resume, dict) else False
            if not self.options.points_arena_enabled:
                # Disabled runs discard both an interrupted visit and its
                # partial count. Re-enabling starts with the next clear.
                completed_since_arena = 0
                self.report.points_arena_pending = False
            checkpoint(self.ctx, "hero_start", report=asdict(self.report),
                       completed_since_arena=completed_since_arena)
            region = self.starting_region()
            if self.report.points_arena_pending:
                # A pause can arrive after the third clear but before/during
                # its arena visit. Finish that visit before another dungeon.
                self.world()
                region = next(r for r in self.bank.profile["regions"] if r["key"] == "demon")
                self.enter_region(region)
                self.run_points_arena(region)
                self.report.points_arena_pending = False
                completed_since_arena = 0
                checkpoint(self.ctx, "hero_points_resumed", report=asdict(self.report),
                           completed_since_arena=completed_since_arena)
                # Reapply the script's usual starting-map policy. Reset mode
                # must still choose its configured regions after this detour.
                region = self.starting_region()
            processed = set()
            while True:
                if region is None:
                    self.world()
                    frame = self.observe()
                    candidates = self.region_candidates(frame, processed)
                    if not candidates:
                        self.finish()
                        self.report.status = "completed"
                        self.report.reason = self.completion_reason
                        break
                    region, _ = candidates[0]
                    self.enter_region(region)
                    processed.add(self.region_work_key(region))
                checkpoint(self.ctx, "hero_region", report=asdict(self.report),
                           completed_since_arena=completed_since_arena, region=dict(region))
                visited: list[Match] = []
                while True:
                    frame = self.observe()
                    targets = [m for m in self.dungeon_targets(region, frame)
                               if not any(self.near(m, old) for old in visited)]
                    if not targets:
                        break
                    target = targets[0]
                    completed = self.fight_dungeon(region, target)
                    visited.append(target)
                    if completed and self.options.points_arena_enabled:
                        completed_since_arena += 1
                        if completed_since_arena == 3:
                            self.report.points_arena_pending = True
                            checkpoint(self.ctx, "hero_points_pending", report=asdict(self.report),
                                       completed_since_arena=2)
                            self.run_points_arena(region)
                            self.report.points_arena_pending = False
                            completed_since_arena = 0
                    checkpoint(self.ctx, "hero_dungeon", report=asdict(self.report),
                               completed_since_arena=completed_since_arena,
                               region=dict(region), completed=completed)
                    attempted = len(self.report.completed) + len(self.report.skipped)
                    if self.options.max_dungeons and attempted >= self.options.max_dungeons:
                        self.finish()
                        self.report.status = "limited"
                        self.report.reason = f"已达到本次最多处理 {self.options.max_dungeons} 个副本的限制。"
                        return self.report
                self.world()
                self.resolve_region(region)
                processed.add(self.region_work_key(region))
                checkpoint(self.ctx, "hero_region_done", report=asdict(self.report),
                           completed_since_arena=completed_since_arena, region=dict(region))
                region = None
            return self.report
        except TaskInterrupted as exc:
            self.report.status = exc.status
            self.report.reason = exc.reason
            raise
        except KeyboardInterrupt:
            self.report.status = "interrupted"
            self.report.reason = "用户按 Ctrl+C 停止。"
            raise
        except Exception as exc:
            self.report.status = "stopped"
            self.report.reason = str(exc)
            raise
        finally:
            if self.frame is not None:
                self.frame.save(self.run_dir / "last_frame.jpg")
            (self.run_dir / "summary.json").write_text(
                json.dumps(asdict(self.report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            self.ctx.log.info("%s：%s；完成 %d 个，跳过 %d 个，失败 %d 次。",
                              self.run_name, self.report.status, len(self.report.completed),
                              len(self.report.skipped), self.report.failures)
