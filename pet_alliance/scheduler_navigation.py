"""Return to a visually confirmed town before switching scheduler tasks."""

from __future__ import annotations

from .adb import AutomationError
from .arena_vision import PointsArenaVision
from .check_arena_vision import CheckArenaVision
from .daily_hero import DailyHero
from .dragon_vision import DragonVision
from .slay_dragon import DragonOptions
from .vision import VisualFrame


class SchedulerNavigator(DailyHero):
    """Navigation only: never challenge a guardian or claim a reward."""

    def __init__(self, ctx):
        super().__init__(ctx, run_name="Scheduler_Navigation")
        self.points = PointsArenaVision(ctx.project_root / "assets" / "points_arena")
        self.guardians = CheckArenaVision(ctx.project_root / "assets" / "check_arena")
        self.dragon = DragonVision(ctx.project_root / "assets" / "slay_dragon")
        self.exits = DragonVision(ctx.project_root / "assets" / "scheduler")
        shared = ctx.settings.get("daily_hero", {})
        values = {name: shared[name] for name in (
            "poll_interval", "min_click_interval", "click_retry_interval", "max_click_retries",
        ) if name in shared}
        values.update(ctx.settings.get("slay_dragon", {}))
        self.login_options = DragonOptions(**values)

    def login_notice_control(self, frame: VisualFrame):
        return self.dragon.find(frame, "announcement_close") or self.dragon.find(frame, "notice")

    def is_town(self, frame: VisualFrame) -> bool:
        # Notices cover the town and must be dismissed before it can be used.
        if self.login_notice_control(frame) or self.dragon.find(frame, "announcement_page"):
            return False
        return bool(self.points.find(frame, "town_page")
                    or self.dragon.find(frame, "town_eye")
                    or self.dragon.find(frame, "town"))

    def dragon_exit(self, frame: VisualFrame):
        if self.dragon.find(frame, "battle_scene") is None:
            return None
        # Both the destination label and the glowing ring must be visible.
        # The returned match is the ring itself, never the label above it.
        return self.exits.find(frame, "dragon_town_portal")

    def close_guardian(self) -> None:
        self.click_until(
            lambda f: self.guardians.find(f, "guardian_close"),
            lambda f: not self.guardians.find(f, "guardian_panel") and (
                self.is_map(f) or self.find("panel_close", f)
                or self.guardians.find(f, "dungeon_close")),
            "调度切换：关闭关卡守护神页面",
        )

    def restart_to_town(self) -> None:
        """Always restart once, then return only after a fresh town confirmation."""
        self.ctx.log.info("调度切换：重启游戏并登录天星镇。")
        self.event("scheduler_navigation_restart", package=self.login_options.package)
        try:
            # Do not navigate or inspect the pre-restart page: midnight refresh
            # must reload the game even when that page already shows the town.
            self.ctx.restart_app(self.login_options.package)
            self.click_until(
                lambda f: None,
                lambda f: self.dragon.find(f, "login_enter")
                or self.login_notice_control(f) or self.dragon.find(f, "announcement_page") or self.is_town(f),
                "重启后等待登录页面", timeout=self.login_options.login_timeout,
            )
            self.click_until(
                lambda f: self.login_notice_control(f) or self.dragon.find(f, "login_enter"),
                self.is_town, "调度切换：登录并处理通知，返回天星镇",
                timeout=self.login_options.town_timeout, retry_by_control=True,
            )
            self.event("scheduler_town_confirmed")
        except BaseException:
            if self.frame is not None:
                self.frame.save(self.run_dir / "navigation_last_frame.jpg")
            raise

    def recover_to_town(self) -> None:
        """Restart/login at any time of day, without the dragon activity gate."""
        self.ctx.log.info("当前页面无法安全回城，重启游戏并登录天星镇。")
        self.restart_to_town()

    def ensure_town(self) -> None:
        """Return only after a fresh frame confirms town; propagate interruption."""
        recovered = False
        try:
            for _ in range(12):
                frame = self.observe()
                if self.is_town(frame):
                    self.event("scheduler_town_confirmed")
                    return
                if self.ctx.dry_run:
                    self.ctx.log.info("[预演] 当前页面尚未回到天星镇，需要导航后再启动任务。")
                    self.event("scheduler_navigation_preview")
                    return
                if self.dragon_exit(frame):
                    self.click_until(self.dragon_exit, self.is_town,
                                     "屠龙场景 → 点击天星镇下方发光环回城",
                                     timeout=self.login_options.town_timeout)
                elif self.login_notice_control(frame) or self.dragon.find(frame, "announcement_page"):
                    self.click_until(self.login_notice_control, self.is_town,
                                     "调度切换：关闭登录通知",
                                     timeout=self.login_options.town_timeout, retry_by_control=True)
                elif self.guardians.find(frame, "guardian_panel"):
                    self.close_guardian()
                elif self.points.find(frame, "roster"):
                    self.click_until(
                        lambda f: self.points.find(f, "roster_close"),
                        lambda f: self.points.find(f, "lobby"),
                        "调度切换：关闭积分赛名单",
                    )
                elif self.points.find(frame, "lobby"):
                    self.click_until(lambda f: self.points.find(f, "close"),
                                     lambda f: self.points.find(f, "arena_page"),
                                     "调度切换：关闭积分战面板")
                elif self.points.find(frame, "arena_page"):
                    self.click_until(lambda f: self.points.find(f, "exit"), self.is_town,
                                     "调度切换：点击竞技场出口物体回城", timeout=300)
                elif self.guardians.find(frame, "dungeon_close") or self.find("panel_close", frame):
                    self.click_until(
                        lambda f: self.guardians.find(f, "dungeon_close") or self.find("panel_close", f),
                        self.is_map, "调度切换：关闭副本面板",
                    )
                elif self.is_world(frame):
                    self.click_until(
                        lambda f: self.points.find(f, "town") if self.is_world(f) else None,
                        self.is_town, "调度切换：大地图 → 天星镇", timeout=300,
                    )
                elif self.is_map(frame):
                    self.world()
                elif self.find("exit_battle", frame):
                    self.click_until(lambda f: self.find("exit_battle", f),
                                     lambda f: self.is_map(f) or self.find("panel_close", f)
                                     or self.points.find(f, "roster") or self.points.find(f, "result"),
                                     "调度切换：退出已结束的战斗")
                elif not recovered:
                    recovered = True
                    self.recover_to_town()
                else:
                    raise AutomationError("重启登录后仍无法确认天星镇，已停止调度切换。")
            raise AutomationError("回到天星镇的页面切换次数异常，已停止调度切换。")
        except BaseException:
            if self.frame is not None:
                self.frame.save(self.run_dir / "navigation_last_frame.jpg")
            raise
