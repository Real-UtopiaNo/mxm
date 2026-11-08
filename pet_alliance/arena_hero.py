"""Verify today's hero clear before claiming one exact Demon City arena."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json

from .adb import AutomationError
from .check_arena_vision import CheckArenaVision
from .cooperation import TaskInterrupted
from .daily_hero import DailyHero, HeroReport
from .dragon_schedule import beijing_now
from .hero_reset_vision import HeroResetVision
from .vision import Match, VisualFrame


PROGRESS_LISTS = ("completed", "skipped")
PROGRESS_COUNTS = ("victories", "failures", "arena_visits", "arena_battles", "arena_no_tickets")


def empty_hero_progress() -> dict:
    return {**{key: [] for key in PROGRESS_LISTS},
            **{key: 0 for key in PROGRESS_COUNTS}, "completed_since_arena": 0,
            "points_arena_pending": False}


class ArenaHeroPrerequisite(DailyHero):
    """Inspect or clear a named dungeon without ever purchasing a reset.

    The caller must already be on the Demon City map. Entrance coordinates are
    always obtained from this dungeon's full name, including on battle retry.
    A completed battle is checked again through all five hero clear stamps.
    """

    def __init__(self, ctx, *, now=None):
        super().__init__(ctx, run_name="Arena_Hero_Prerequisite")
        if now is not None:
            self.now = now
        self.arena_vision = CheckArenaVision(ctx.project_root / "assets" / "check_arena")
        self.reset_vision = HeroResetVision(ctx.project_root / "assets" / "daily_hero_reset")
        self.demon_region = next(region for region in self.bank.profile["regions"] if region["key"] == "demon")
        self.dungeon_name: str | None = None
        self.panel_dungeon: str | None = None
        self.verified_date: str | None = None
        self.progress_by_day: dict[str, dict] = {}
        self._accounted = asdict(HeroReport())
        self.counter_date = self.day()
        resume = getattr(ctx, "hero_resume_data", {})
        resume = resume if isinstance(resume, dict) else {}
        count = resume.get("completed_since_arena", 0)
        self.completed_since_arena = count if (getattr(ctx, "hero_resume_date", None) == self.counter_date
                                                 and type(count) is int and 0 <= count <= 2) else 0
        self.report.points_arena_pending = (getattr(ctx, "hero_resume_date", None) == self.counter_date
                                            and resume.get("points_arena_pending") is True)
        if not self.options.points_arena_enabled:
            self.completed_since_arena = 0
            self.report.points_arena_pending = False

    @staticmethod
    def now() -> datetime:
        return beijing_now()

    def day(self) -> str:
        return self.now().date().isoformat()

    def sync_progress(self) -> None:
        day = self.day()
        if day != self.counter_date or not self.options.points_arena_enabled:
            self.counter_date = day
            self.completed_since_arena = 0
            self.report.points_arena_pending = False
        current = asdict(self.report)
        bucket = self.progress_by_day.setdefault(day, empty_hero_progress())
        for key in PROGRESS_LISTS:
            bucket[key].extend(current[key][len(self._accounted[key]):])
        for key in PROGRESS_COUNTS:
            bucket[key] += current[key] - self._accounted[key]
        bucket["completed_since_arena"] = self.completed_since_arena
        bucket["points_arena_pending"] = self.report.points_arena_pending
        self._accounted = current
        self.ctx.hero_resume_date = day
        self.ctx.hero_resume_data = {"completed_since_arena": self.completed_since_arena,
                                    "points_arena_pending": self.report.points_arena_pending}

    def save_summary(self) -> None:
        if self.frame is not None:
            self.frame.save(self.run_dir / "last_frame.jpg")
        (self.run_dir / "summary.json").write_text(json.dumps({
            **asdict(self.report), "dungeon": self.dungeon_name, "verified_date": self.verified_date,
            "hero_progress_by_day": self.progress_by_day,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def resume_pending_points(self) -> None:
        """Finish a previously interrupted third-dungeon points visit."""
        try:
            self.sync_progress()
            if self.report.points_arena_pending:
                self.run_points_arena(self.demon_region)
                self.report.points_arena_pending = False
                self.completed_since_arena = 0
        except TaskInterrupted as exc:
            self.report.status = exc.status
            self.report.reason = exc.reason
            raise
        except Exception as exc:
            self.report.status = "stopped"
            self.report.reason = str(exc)
            raise
        finally:
            self.sync_progress()
            self.save_summary()

    def target(self, frame: VisualFrame) -> Match | None:
        if self.dungeon_name is None or not self.arena_vision.find(frame, "map_anchor"):
            return None
        return self.arena_vision.dungeon(frame, self.dungeon_name)

    def open_dungeon_panel(self, target: Match, resume: bool = False) -> None:
        frame = self.observe()
        if self.find("panel_close", frame) and self.panel_dungeon == self.dungeon_name:
            return
        self.wait_for(lambda f: self.arena_vision.find(f, "map_anchor"), "魔都区域地图")
        self.click_until(self.target, lambda f: self.find("panel_close", f),
                         f"检查占擂前置：进入{self.dungeon_name}")
        self.panel_dungeon = self.dungeon_name

    def open_hero_panel(self, target: Match, resume: bool = False) -> str:
        self.open_dungeon_panel(target, resume)
        frame = self.observe()
        if self.normal_mode_control(frame):
            self.click_until(self.normal_mode_control, self.reset_vision.hero_state,
                             f"检查{self.dungeon_name}的英雄级通关状态")
        return self.wait_for(self.reset_vision.hero_state, "英雄级五关通关状态")

    def open_dungeon(self, target: Match, resume: bool = False) -> None:
        state = self.open_hero_panel(target, resume)
        if state != "incomplete":
            raise AutomationError("战斗重试前英雄级状态已改变，停止挑战并重新检查。")
        self.click_until(
            lambda f: self.find("challenge", f) if self.reset_vision.hero_state(f) == "incomplete" else None,
            self.battle_state, f"占擂前置：挑战{self.dungeon_name}英雄级",
        )

    def close_panel(self) -> None:
        self.return_to_map()
        self.panel_dungeon = None
        self.wait_for(lambda f: self.arena_vision.find(f, "map_anchor"), "返回魔都区域地图")

    def switch_battle_loadout(self, group: str) -> None:
        """Return through town and relocate Demon City after changing pets."""
        from .pet_loadouts import switch_pet_loadout

        if self.last_click_at is not None:
            self.ctx.pet_loadout_last_click_at = self.last_click_at
        switch_pet_loadout(self.ctx, group)
        self.last_click_at = getattr(self.ctx, "pet_loadout_last_click_at", self.last_click_at)
        self.panel_dungeon = None
        self.world()
        self.enter_region(self.demon_region)
        self.wait_for(lambda f: self.arena_vision.find(f, "map_anchor"), "换组后返回魔都区域地图")

    def ensure_cleared(self, name: str, allow_battle: bool) -> bool:
        self.dungeon_name = name
        if name not in self.arena_vision.dungeon_names:
            raise AutomationError(f"没有该魔都副本的视觉模板：{name}")
        battled_date = None
        try:
            self.resume_pending_points()
            # A midnight between panel entry and verification invalidates the
            # observation. Reopen it rather than using yesterday's stamps.
            for _ in range(4):
                opened_date = self.day()
                target = self.wait_for(self.target, f"精确识别魔都副本：{name}")
                state = self.open_hero_panel(target)
                if self.day() != opened_date:
                    self.close_panel()
                    continue
                if state == "cleared":
                    self.close_panel()
                    if self.day() != opened_date:
                        continue
                    self.verified_date = opened_date
                    self.report.status = "completed"
                    self.report.reason = f"已核验{name}当天英雄级五关全部通过。"
                    return True
                if not allow_battle or battled_date == opened_date:
                    self.close_panel()
                    self.report.status = "hero_required"
                    self.report.reason = f"{name}当天英雄级五关通关状态未获确认，跳过占擂。"
                    return False
                self.ctx.log.info("%s当天英雄级尚未通过，先完成该副本后再占擂。", name)
                if getattr(self.ctx, "active_pet_loadout", None) != "刷图":
                    self.close_panel()
                    self.switch_battle_loadout("刷图")
                    # Changing groups returns through town. Never reuse the
                    # old entrance or hero-state observation after that trip.
                    target = self.wait_for(self.target, f"换组后精确识别魔都副本：{name}")
                    state = self.open_hero_panel(target)
                    if state != "incomplete" or self.day() != opened_date:
                        self.close_panel()
                        self.switch_battle_loadout("PVP")
                        continue
                completed = self.fight_dungeon(self.demon_region, target)
                self.panel_dungeon = None
                if completed:
                    battled_date = self.day()
                if completed and self.options.points_arena_enabled:
                    if self.day() != self.counter_date:
                        self.counter_date = self.day()
                        self.completed_since_arena = 0
                    self.completed_since_arena += 1
                    if self.completed_since_arena == 3:
                        self.completed_since_arena = 2
                        self.report.points_arena_pending = True
                self.sync_progress()
                # This is normal control flow, not cleanup: a pause or failed
                # fight operation must propagate without issuing extra inputs.
                self.switch_battle_loadout("PVP")
                if self.report.points_arena_pending:
                    self.run_points_arena(self.demon_region)
                    self.report.points_arena_pending = False
                    self.completed_since_arena = 0
                    self.sync_progress()
                if not completed:
                    self.report.status = "hero_required"
                    self.report.reason = f"{name}英雄级未能通关，跳过占擂。"
                    return False
                # Always reopen the exact dungeon and inspect the stamps; a
                # victory screen alone is never a claim authorization.
            raise AutomationError("占擂前置核验反复跨越北京时间零点，已停止。")
        except TaskInterrupted as exc:
            self.report.status = exc.status
            self.report.reason = exc.reason
            raise
        except Exception as exc:
            self.report.status = "stopped"
            self.report.reason = str(exc)
            raise
        finally:
            self.sync_progress()
            self.save_summary()
