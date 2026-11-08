"""One points-arena visit, using DailyHero's observed click/confirm loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .adb import AutomationError
from .arena_vision import PointsArenaVision
from .daily_hero import ClickTimeoutError

if TYPE_CHECKING:
    from .daily_hero import DailyHero
    from .vision import Match, VisualFrame


class PointsArena:
    def __init__(self, hero: DailyHero):
        self.hero = hero
        self.vision = PointsArenaVision(hero.ctx.project_root / "assets" / "points_arena")
        self.entry = 0
        self.round = 0
        self.selection_attempts = 0
        self.selection_evidence: list[tuple[int, VisualFrame, str]] = []

    def save_roster(self, frame: VisualFrame, name: str) -> str:
        """Keep the exact matching frame losslessly, without another ADB capture."""
        import cv2

        path = self.hero.run_dir / f"arena_visit_{self.hero.report.arena_visits}_entry_{self.entry}_{name}.png"
        ok, encoded = cv2.imencode(".png", frame.image)
        if not ok:
            raise AutomationError("无法保存积分赛名单诊断截图。")
        encoded.tofile(str(path))
        return path.name

    def find(self, name: str, frame: VisualFrame) -> Match | None:
        return self.vision.find(frame, name)

    def transition(self, control: str, source: str, destination: str, description: str, *,
                   timeout: float | None = None) -> None:
        self.hero.click_until(
            lambda f: self.find(control, f) if self.find(source, f) else None,
            lambda f: self.find(destination, f), description, timeout=timeout,
        )

    def battle_result(self, frame: VisualFrame) -> Match | None:
        # The shared Exit control is the arena's result action. Its title may
        # differ from dungeon results, and it never requires clicking Next.
        return self.hero.find("exit_battle", frame)

    def battle_started(self, frame: VisualFrame) -> Match | None:
        return self.battle_result(frame) or self.hero.find("skip", frame)

    def select_target(self, frame: VisualFrame) -> Match | None:
        """Recheck the entire priority list from its first item on every call."""
        if not self.find("roster", frame):
            return None
        target = self.vision.target(frame, self.hero.options.target_players)
        if target is not None and self.round:
            self.selection_attempts += 1
            # Keep this frame in memory. Encoding it before the tap would add
            # delay while opponents may be joining and changing card positions.
            self.selection_evidence.append((self.selection_attempts, frame, target.name))
        return target

    def fight(self, number: int) -> None:
        self.round = number
        self.selection_attempts = 0
        self.selection_evidence.clear()
        self.hero.wait_for(lambda f: self.find("roster", f), "自由竞技场玩家列表")
        # The page heading can render before its nameplates. No verified target
        # means observe, never tap a previous card or abort a partial frame.
        # Selection uses the same timed re-location/retry loop as every button.
        try:
            self.hero.click_until(self.select_target, self.battle_started, f"积分赛第 {number}/3 场：选择白名单玩家")
        except ClickTimeoutError as exc:
            frame = self.hero.frame
            if (frame is not None and self.find("roster", frame)
                    and self.vision.target(frame, self.hero.options.target_players) is None):
                evidence = self.save_roster(frame, f"round_{number}_target_missing")
                self.hero.event("arena_target_missing", entry=self.entry, round=number,
                                target_players=self.hero.options.target_players, file=evidence)
                raise AutomationError(
                    "积分赛观察等待后仍未找到 target_players 中可挑战的完整玩家名（"
                    + "、".join(self.hero.options.target_players)
                    + "）；已停止，不会点击其他玩家。"
                ) from exc
            raise
        finally:
            for attempt, roster, template in self.selection_evidence:
                evidence = self.save_roster(roster, f"round_{number}_select_{attempt}")
                self.hero.event("arena_target_snapshot", entry=self.entry, round=number,
                                template=template, file=evidence)
            self.selection_evidence.clear()
        if not self.battle_result(self.hero.frame):
            self.hero.click_until(
                lambda f: self.hero.find("skip", f) if not self.battle_result(f) else None,
                self.battle_result, "积分赛：跳过战斗",
            )
        self.hero.click_until(
            self.battle_result,
            lambda f: self.find("roster", f) or self.find("result", f),
            "积分赛：退出战斗",
        )
        self.hero.report.arena_battles += 1
        self.hero.event("arena_battle_complete", entry=self.entry, round=number)
        expected = "result" if number == 3 else "roster"
        if not self.find(expected, self.hero.frame):
            raise AutomationError(f"积分赛第 {number} 场退出后的页面不符合预期，已停止。")

    def run(self) -> None:
        self.hero.ctx.log.info("已完成三个英雄副本，进入积分赛。")
        self.hero.event("arena_start", completed_dungeons=len(self.hero.report.completed))
        self.hero.world()
        self.hero.click_until(
            lambda f: self.find("town", f) if self.hero.is_world(f) else None,
            lambda f: self.find("town_page", f), "大地图 → 天星镇", timeout=300,
        )
        self.hero.ctx.log.info("等待识别天星镇“竞技场”入口，最多 5 分钟。")
        self.transition("arena", "town_page", "arena_page", "天星镇 → 竞技场", timeout=300)
        self.transition("points", "arena_page", "lobby", "竞技场 → 积分战")
        while True:
            # Read the lobby again after every settlement, never reuse the
            # pre-entry count or assume the ticket was already deducted.
            tickets = self.vision.ticket_status(self.hero.observe())
            if tickets is None:
                raise AutomationError("无法确认积分赛入场券数量，已停止，避免误用入场券。")
            self.hero.event("arena_tickets", remaining=tickets, completed_entries=self.entry)
            if tickets == 0:
                self.hero.report.arena_no_tickets += 1
                self.hero.ctx.log.info("积分赛入场券为 0/5，返回英雄副本。")
                self.hero.event("arena_no_tickets", completed_entries=self.entry)
                break
            self.entry += 1
            self.hero.ctx.log.info("积分赛第 %d 轮报名，当前入场券 %d/5。", self.entry, tickets)
            self.hero.event("arena_entry_start", entry=self.entry, tickets=tickets)
            self.transition("join", "lobby", "confirm_entry", "参加自由竞技场")
            self.transition("confirm_entry", "confirm_entry", "roster", "确认使用一张竞技入场券")
            # Each ticket buys three battles. Every battle independently
            # searches the current roster from the top of the priority list.
            for number in range(1, 4):
                self.fight(number)
            self.hero.click_until(
                lambda f: self.find("confirm_result", f) if self.find("result", f) else None,
                lambda f: self.find("lobby", f) or self.find("roster", f),
                "确认积分赛三场结果",
            )
            if self.find("roster", self.hero.frame):
                self.transition("roster_close", "roster", "lobby", "关闭自由竞技场玩家列表")
            self.hero.event("arena_entry_complete", entry=self.entry)
        self.transition("close", "lobby", "arena_page", "关闭积分战面板")
        self.transition("exit", "arena_page", "town_page", "点击出口物体，返回天星镇", timeout=300)
        self.hero.event("arena_returned_to_town")
