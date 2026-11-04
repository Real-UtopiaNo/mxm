from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import time

from .adb import AutomationError
from .runtime import ScriptContext
from .vision import Match, TemplateBank, VisualFrame


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

    def __post_init__(self):
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
    reason: str = ""


class DailyHero:
    def __init__(self, ctx: ScriptContext):
        self.ctx = ctx
        self.options = HeroOptions(**ctx.settings.get("daily_hero", {}))
        self.bank = TemplateBank(ctx.project_root / "assets" / "daily_hero")
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = ctx.project_root / "artifacts" / "runs" / f"Daily_Hero_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
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
            raise AutomationError("Daily_Hero 已达到最长运行时间。")
        self.frame = VisualFrame.from_screenshot(self.ctx.capture())
        return self.frame

    def find(self, name: str, frame: VisualFrame | None = None) -> Match | None:
        return self.bank.find(frame or self.frame, name)

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

    def click_until(self, locate, arrived, description: str, *, offset: tuple[int, int] = (0, 0)):
        """Click once, observe the destination, and retry only a visible source control."""
        deadline = time.monotonic() + self.options.transition_timeout
        attempts = 0
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
                raise AutomationError(f"点击“{description}”后未确认目标页面，已停止并保存现场。")
            # Re-locate on every attempt. Once the source disappears (e.g. a
            # loading screen), only observe; never tap the previous coordinates.
            if now >= next_attempt_at and (
                self.last_click_at is None or now - self.last_click_at >= self.options.min_click_interval
            ):
                control = locate(frame)
                if control is not None:
                    if attempts > self.options.max_click_retries:
                        raise AutomationError(
                            f"“{description}”补点 {self.options.max_click_retries} 次后仍未成功，已停止。"
                        )
                    if attempts:
                        self.ctx.log.info("页面尚未切换，重新定位后补点：%s（%d/%d）",
                                          description, attempts, self.options.max_click_retries)
                        self.event("click_retry", description=description, retry=attempts)
                    self.tap(control, description, offset)
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

    def world(self) -> None:
        frame = self.observe()
        if self.is_world(frame):
            self.wait_for(self.world_ready, "大地图区域标签显示完整")
            return
        if self.find("panel_close", frame):
            self.click_until(lambda f: self.find("panel_close", f), self.is_map, "关闭副本面板")
        elif self.find("adventure", frame):
            self.click_until(lambda f: self.find("adventure", f), self.is_map, "天星镇 → 冒险")
        elif not self.is_map(frame):
            raise AutomationError("请从天星镇主页面开始；也支持从区域地图或大地图继续。")
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
                "进入带感叹号的副本", offset=(-65, 10),
            )

    def open_dungeon(self, target: Match, resume: bool = False) -> None:
        self.open_dungeon_panel(target, resume)
        frame = self.observe()
        if self.find("normal_dragon", frame):
            self.click_until(
                lambda f: self.find("normal_dragon", f),
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

    def fight_dungeon(self, region: dict, target: Match) -> bool:
        budget = RetryBudget(self.options.max_retries)
        description = {"region": region["name"], "marker": target.center}
        resume = False
        while True:
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
                    self.frame.save(self.run_dir / f"completed_{len(self.report.completed) + 1}.jpg")
                    self.click_until(
                        lambda f: self.find("exit_battle", f) if self.find("victory", f) else None,
                        self.map_destination, "副本英雄级已完成 → 退出战斗",
                    )
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

    def run(self) -> HeroReport:
        self.ctx.log.info("Daily_Hero 开始；诊断目录：%s", self.run_dir)
        try:
            if self.ctx.dry_run:
                self.preview()
                return self.report
            self.world()
            processed = set()
            while True:
                frame = self.observe()
                candidates = [(r, m) for r, m in self.regions(frame) if r["key"] not in processed]
                if not candidates:
                    current = self.current_region(frame)
                    if current and current[0]["key"] not in processed:
                        self.ctx.log.info("当前区域的菱形标志可能遮住感叹号，检查其副本标记。")
                        candidates = [current]
                if not candidates:
                    self.report.status = "completed"
                    self.report.reason = "未发现尚需处理且带感叹号的推荐等级区域。"
                    break
                region, _ = candidates[0]
                self.enter_region(region)
                processed.add(region["key"])
                visited: list[Match] = []
                while True:
                    frame = self.observe()
                    targets = [m for m in self.markers(frame) if not any(self.near(m, old) for old in visited)]
                    if not targets:
                        break
                    target = targets[0]
                    self.fight_dungeon(region, target)
                    visited.append(target)
                    attempted = len(self.report.completed) + len(self.report.skipped)
                    if self.options.max_dungeons and attempted >= self.options.max_dungeons:
                        self.report.status = "limited"
                        self.report.reason = f"已达到本次最多处理 {self.options.max_dungeons} 个副本的限制。"
                        return self.report
                self.world()
            return self.report
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
            self.ctx.log.info("Daily_Hero：%s；完成 %d 个，跳过 %d 个，失败 %d 次。",
                              self.report.status, len(self.report.completed), len(self.report.skipped), self.report.failures)
