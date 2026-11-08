"""Reset selected regions first, then individually named hero dungeons."""

from __future__ import annotations

from dataclasses import dataclass, field

from .adb import AutomationError
from .arena_vision import PointsArenaVision
from .daily_hero import DailyHero, HeroOptions, HeroReport
from .hero_reset_vision import HeroResetVision
from .runtime import ScriptContext
from .vision import Match, VisualFrame


@dataclass(frozen=True)
class HeroResetOptions(HeroOptions):
    regions: list[str] = field(default_factory=lambda: ["西部大陆", "环形领地", "诅咒荒野"])
    dungeons: list[str] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if (not isinstance(self.regions, list)
                or any(not isinstance(name, str) or not name.strip() for name in self.regions)):
            raise AutomationError("daily_hero_reset.regions 必须是包含非空区域名的字符串列表。")
        if len(set(self.regions)) != len(self.regions):
            raise AutomationError("daily_hero_reset.regions 不能包含重复区域。")
        if not isinstance(self.dungeons, list):
            raise AutomationError("daily_hero_reset.dungeons 必须是“区域-副本”格式的字符串列表。")
        parsed = []
        for value in self.dungeons:
            if (not isinstance(value, str) or "-" not in value
                    or any(not part.strip() for part in value.split("-", 1))):
                raise AutomationError("daily_hero_reset.dungeons 必须是“区域-副本”格式的字符串列表。")
            parsed.append(tuple(part.strip() for part in value.split("-", 1)))
        if len(set(parsed)) != len(parsed):
            raise AutomationError("daily_hero_reset.dungeons 不能包含重复副本。")
        object.__setattr__(self, "dungeons", [f"{region}-{dungeon}" for region, dungeon in parsed])
        if not self.regions and not self.dungeons:
            raise AutomationError("daily_hero_reset.regions 和 dungeons 至少配置一项。")


@dataclass
class ResetReport(HeroReport):
    resets: int = 0
    diamonds_spent: int = 0
    diamond_spending: list[dict] = field(default_factory=list)
    returned_to_town: bool = False


class ResetUnavailable(AutomationError):
    def __init__(self):
        super().__init__("当前副本已重置过（重置按钮为灰色）。")


class DailyHeroReset(DailyHero):
    completion_reason = "已处理所有选定区域和指定英雄副本，并返回天星镇。"
    dungeon_entry_offset = (0, 0)

    def __init__(self, ctx: ScriptContext):
        values = {**ctx.settings.get("daily_hero", {}), **ctx.settings.get("daily_hero_reset", {})}
        # Points participation is one global hero setting; other reset options
        # may still override their ordinary hero defaults.
        values["points_arena_enabled"] = ctx.settings.get("daily_hero", {}).get("points_arena_enabled", True)
        options = HeroResetOptions(**values)
        super().__init__(ctx, options=options, run_name="Daily_Hero_Reset")
        available = {region["name"]: region for region in self.bank.profile["regions"]}
        unknown = [name for name in options.regions if name not in available]
        if unknown:
            raise AutomationError(f"daily_hero_reset.regions 含未知区域：{', '.join(unknown)}。")
        self.selected_regions = [available[name] for name in options.regions]
        self.reset_vision = HeroResetVision(ctx.project_root / "assets" / "daily_hero_reset")
        self.selected_dungeons = []
        for value in options.dungeons:
            region_name, dungeon_name = (part.strip() for part in value.split("-", 1))
            if region_name not in available:
                raise AutomationError(f"daily_hero_reset.dungeons 含未知区域：{region_name}。")
            self.reset_vision.validate_dungeon(region_name, dungeon_name)
            self.selected_dungeons.append({**available[region_name], "_reset_dungeon": dungeon_name})
        self.town_vision = PointsArenaVision(ctx.project_root / "assets" / "points_arena")
        self.report = ResetReport()
        self._current_reset_region: str | None = None
        self._current_reset_dungeon: str | None = None

    def reset_find(self, name: str, frame: VisualFrame | None = None) -> Match | None:
        return self.reset_vision.find(frame or self.frame, name)

    def finish(self) -> None:
        self.ctx.log.info("本次副本处理完成，返回天星镇。")
        self.return_to_town()
        self.report.returned_to_town = True
        self.event("reset_returned_to_town")

    def starting_region(self) -> None:
        # Local maps have no region label. Establish the whitelist on the world
        # map before opening any dungeon, including the initial local map.
        self.world()
        return None

    def region_candidates(self, frame: VisualFrame, processed: set[str]) -> list[tuple[dict, Match]]:
        candidates = []
        for region in [*self.selected_regions, *self.selected_dungeons]:
            if self.region_work_key(region) in processed:
                continue
            label = self.find(region["template"], frame)
            if label is None:
                raise AutomationError(f"大地图未识别到选定区域“{region['name']}”，已停止。")
            candidates.append((region, label))
        return candidates

    def region_work_key(self, region: dict) -> str:
        name = region.get("_reset_dungeon")
        return f"{region['key']}:{name}" if name is not None else region["key"]

    def dungeon_targets(self, region: dict, frame: VisualFrame) -> list[Match]:
        name = region.get("_reset_dungeon")
        if name is None:
            return self.markers(frame)
        target = self.reset_vision.named_entrance(frame, region["name"], name)
        if target is None:
            raise AutomationError(f"当前区域未识别到指定副本“{region['name']}-{name}”，已停止，未点击其他副本。")
        return [target]

    def dungeon_description(self, region: dict, target: Match) -> dict:
        description = super().dungeon_description(region, target)
        if region.get("_reset_dungeon") is not None:
            description["dungeon"] = region["_reset_dungeon"]
        return description

    def markers(self, frame: VisualFrame) -> list[Match]:
        if self._current_reset_dungeon is not None:
            target = self.reset_vision.named_entrance(
                frame, self._current_reset_region, self._current_reset_dungeon,
            )
            return [target] if target is not None else []
        # Completed dungeons no longer have exclamation badges. Locate their
        # green nameplates instead; town's gold nameplate is excluded.
        entrances = self.reset_vision.entrances(frame)
        if not entrances:
            raise AutomationError("当前区域未识别到副本名牌，已停止并保存现场。")
        return entrances

    def open_dungeon_panel(self, target: Match, resume: bool = False) -> None:
        frame = self.observe()
        if resume and self.find("panel_close", frame):
            return
        self.wait_for(self.is_map, "返回区域地图")
        self.click_until(
            lambda f: next((m for m in self.markers(f) if self.near(m, target)), None) if self.is_map(f) else None,
            lambda f: self.find("panel_close", f),
            "进入待重置的副本",
        )

    def hero_panel(self, frame: VisualFrame):
        return self.find("hero_dragon", frame) and self.find("panel_close", frame) and (
            self.find("challenge", frame) or self.reset_find("reset", frame)
            or self.reset_find("reset_disabled", frame)
        )

    def challenge_ready(self, frame: VisualFrame):
        return (self.find("hero_dragon", frame) and self.find("challenge", frame)
                and not self.reset_find("reset_dialog", frame))

    def open_dungeon(self, target: Match, resume: bool = False) -> None:
        self.open_dungeon_panel(target, resume)
        frame = self.observe()
        if self.normal_mode_control(frame):
            self.click_until(
                self.normal_mode_control, self.hero_panel, "切换英雄级模式",
            )
        self.wait_for(self.hero_panel, "英雄级模式和挑战或重置按钮")
        # Grey means already reset. It is a skip on both the initial visit
        # and any return after a failed battle, never a reason to challenge.
        if self.reset_find("reset_disabled"):
            raise ResetUnavailable()
        if not resume:
            if self.reset_find("reset"):
                self.click_until(
                    lambda f: self.reset_find("reset", f),
                    lambda f: self.reset_find("reset_dialog", f) and self.reset_find("confirm_reset", f),
                    "重置已完成的英雄副本",
                )
                # Keep the exact recognition input even when its price is
                # unreadable. JPEG smoothing can hide a native PNG failure.
                self.frame.save(self.run_dir / f"reset_{self.report.resets + 1}_quote.png")
                cost = self.reset_vision.reset_cost(self.frame)
                if type(cost) is not int or cost < 0:
                    raise AutomationError("无法识别重置所需的钻石数量，已停止，未点击确定。")
                self.frame.save(self.run_dir / f"reset_{self.report.resets + 1}_quote.jpg")
                self.ctx.log.info("本次重置需要 %d 钻石。", cost)

                def confirmation(frame):
                    button = self.reset_find("confirm_reset", frame)
                    if button and self.reset_vision.reset_cost(frame) != cost:
                        raise AutomationError("重置弹窗的钻石费用发生变化或无法确认，已停止。")
                    return button

                # Confirmation is accepted only after deliberately opening the
                # reset dialog, never while navigating or retrying a battle.
                self.click_until(
                    confirmation,
                    self.challenge_ready, "确定重置英雄副本",
                )
                # A retry sends another input, not another purchase. Record
                # the charge once, after the playable panel confirms success.
                self.report.resets += 1
                self.report.diamonds_spent += cost
                spending = {"region": self._current_reset_region or "未知区域",
                            "marker": target.center, "diamonds": cost}
                if self._current_reset_dungeon is not None:
                    spending["dungeon"] = self._current_reset_dungeon
                self.report.diamond_spending.append(spending)
                self.event("dungeon_reset", **spending, diamonds_spent=self.report.diamonds_spent)
                self.ctx.log.info("重置成功，消耗 %d 钻石；本次运行累计消耗 %d 钻石。",
                                  cost, self.report.diamonds_spent)
        # Failed battles must retain cleared stages. A resume never clicks
        # either reset or its confirmation, even if the panel is unexpected.
        self.wait_for(self.challenge_ready, "英雄级模式和挑战按钮")
        self.click_until(
            lambda f: self.find("challenge", f) if self.challenge_ready(f) else None,
            self.battle_state, "英雄级挑战",
        )

    def fight_dungeon(self, region: dict, target: Match) -> bool:
        previous_region = self._current_reset_region
        previous_dungeon = self._current_reset_dungeon
        self._current_reset_region = region["name"]
        self._current_reset_dungeon = region.get("_reset_dungeon")
        if self._current_reset_dungeon is not None:
            self.ctx.log.info("指定副本重置：%s - %s。", region["name"], self._current_reset_dungeon)
        failures_before = self.report.failures
        try:
            return super().fight_dungeon(region, target)
        except ResetUnavailable:
            self.return_to_map()
            entry = {**self.dungeon_description(region, target),
                     "failures": self.report.failures - failures_before,
                     "reason": "已重置过（重置按钮为灰色）", "reason_code": "already_reset"}
            self.report.skipped.append(entry)
            self.event("dungeon_reset_unavailable", **entry)
            self.ctx.log.info("%s：当前副本已重置过（重置按钮为灰色），关闭面板并跳过。", region["name"])
            return False
        finally:
            self._current_reset_region = previous_region
            self._current_reset_dungeon = previous_dungeon

    def run(self) -> ResetReport:
        try:
            return super().run()
        finally:
            self.ctx.log.info("本次运行已确认重置 %d 次，累计消耗 %d 钻石；统计文件：%s",
                              self.report.resets, self.report.diamonds_spent, self.run_dir / "summary.json")

    def preview(self) -> None:
        frame = self.observe()
        matches = []
        self.ctx.log.info("[预演] 仅处理区域：%s", "、".join(self.options.regions))
        self.ctx.log.info("[预演] 区域处理后指定副本：%s", "、".join(self.options.dungeons) or "无")
        if self.is_world(frame):
            candidates = self.region_candidates(frame, set())
            self.ctx.log.info("[预演] 已识别选定区域：%s", "、".join(r["name"] for r, _ in candidates))
            matches = [m for _, m in candidates]
        elif self.is_map(frame):
            matches = self.markers(frame)
            self.ctx.log.info("[预演] 当前地图识别到 %d 个副本入口；正式运行先通过大地图核对区域。", len(matches))
        elif self.find("panel_close", frame):
            state = ("可重置" if self.reset_find("reset", frame) else
                     "已重置过，正式运行时跳过" if self.reset_find("reset_disabled", frame) else "待切换或可挑战")
            self.ctx.log.info("[预演] 当前副本面板：%s；正式运行先关闭面板并核对区域。", state)
        elif self.find("adventure", frame):
            self.ctx.log.info("[预演] 已识别冒险入口；正式运行进入大地图核对区域。")
        else:
            raise AutomationError("未识别到天星镇、地图或副本面板，请检查诊断截图。")
        frame.save(self.run_dir / "preview.jpg", matches)
        self.report.status = "preview"
