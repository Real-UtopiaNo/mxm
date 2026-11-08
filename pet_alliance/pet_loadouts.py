"""Apply a named, already saved pet formation before a game task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from .adb import AutomationError
from .cooperation import TaskInterrupted
from .daily_hero import DailyHero
from .dragon_vision import DragonVision


GROUP_CONTROLS = {"刷图": "group_farm", "屠龙": "group_dragon", "PVP": "group_pvp"}


@dataclass
class LoadoutReport:
    group: str
    status: str = "running"
    reason: str = ""


class PetLoadoutSwitcher(DailyHero):
    def __init__(self, ctx, group: str):
        if group not in GROUP_CONTROLS:
            raise AutomationError(f"没有注册的宠物组：{group}。")
        super().__init__(ctx, run_name="Pet_Loadout")
        self.last_click_at = getattr(ctx, "pet_loadout_last_click_at", None)
        self.group = group
        self.report = LoadoutReport(group)
        self.vision = DragonVision(ctx.project_root / "assets" / "pet_loadouts")

    def pet_find(self, name, frame=None):
        return self.vision.find(frame or self.frame, name)

    def pets_ready(self, frame):
        return self.pet_find("pet_page", frame) and self.pet_find("pet_backup_title", frame)

    def run(self):
        self.ctx.active_pet_loadout = None
        self.ctx.log.info("切换宠物组：%s；记录目录：%s", self.group, self.run_dir)
        try:
            if self.ctx.dry_run:
                self.report.status = "preview"
                return self.report
            frame = self.observe()
            # A cancelled attempt can leave a pet dialog open. Discard that
            # previous selection rather than confirming an unspecified group.
            if self.pet_find("confirm_page", frame):
                self.click_until(
                    lambda f: self.pet_find("cancel_button", f),
                    lambda f: self.pet_find("preset_page", f) or self.pets_ready(f),
                    "取消上次尚未确认的宠物方案",
                )
                frame = self.frame
            if self.pet_find("preset_page", frame):
                self.click_until(lambda f: self.pet_find("preset_back", f), self.pets_ready,
                                 "返回宠物页面，重新选择目标组")
                frame = self.frame
            if not self.pets_ready(frame):
                if not self.pet_find("town_page", frame):
                    from .scheduler_navigation import SchedulerNavigator

                    navigator = SchedulerNavigator(self.ctx)
                    navigator.last_click_at = self.last_click_at
                    navigator.ensure_town()
                    self.last_click_at = navigator.last_click_at
                self.click_until(
                    lambda f: self.pet_find("town_pet_button", f), self.pets_ready,
                    "天星镇 → 宠物页面", timeout=300,
                )
            self.click_until(
                lambda f: self.pet_find(GROUP_CONTROLS[self.group], f) if self.pets_ready(f) else None,
                lambda f: self.pet_find("preset_page", f),
                f"选择宠物组：{self.group}",
            )
            self.click_until(lambda f: self.pet_find("apply", f),
                             lambda f: self.pet_find("confirm_page", f),
                             f"一键使用宠物组：{self.group}")
            self.click_until(
                lambda f: self.pet_find("confirm_button", f),
                lambda f: self.pets_ready(f) or self.pet_find("preset_page", f),
                f"确定使用宠物组：{self.group}",
            )
            # Some transitions restore the preview first. Never press Apply
            # again after confirming; only leave the already applied preview.
            if self.pet_find("preset_page", self.frame):
                self.click_until(lambda f: self.pet_find("preset_back", f), self.pets_ready,
                                 "关闭已使用的宠物方案预览")
            self.click_until(lambda f: self.pet_find("pet_close", f) if self.pets_ready(f) else None,
                             lambda f: self.pet_find("town_page", f),
                             "关闭宠物页面，返回天星镇", timeout=300)
            self.ctx.active_pet_loadout = self.group
            self.ctx.pet_loadout_last_click_at = self.last_click_at
            self.report.status = "completed"
            self.event("pet_loadout_applied", group=self.group)
            self.ctx.log.info("已使用宠物组“%s”，并返回天星镇。", self.group)
            return self.report
        except TaskInterrupted as exc:
            self.report.status, self.report.reason = exc.status, exc.reason
            raise
        except KeyboardInterrupt:
            self.report.status, self.report.reason = "interrupted", "用户停止换组。"
            raise
        except Exception as exc:
            self.report.status, self.report.reason = "stopped", str(exc)
            raise
        finally:
            if self.frame is not None:
                self.frame.save(self.run_dir / "last_frame.png")
            (self.run_dir / "summary.json").write_text(
                json.dumps(asdict(self.report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def switch_pet_loadout(ctx, group: str) -> None:
    PetLoadoutSwitcher(ctx, group).run()


def run_with_loadout(job, group: str):
    # Preserve script preview behavior: the task still inspects its current
    # page, while no preparation navigation or pet input is attempted.
    if not job.ctx.dry_run and not getattr(job, "defer_loadout_preparation", False):
        try:
            switch_pet_loadout(job.ctx, group)
            job.last_click_at = getattr(job.ctx, "pet_loadout_last_click_at", None)
        except (Exception, KeyboardInterrupt) as exc:
            report = getattr(job, "report", None)
            if report is not None:
                report.status = (exc.status if isinstance(exc, TaskInterrupted) else
                                 "interrupted" if isinstance(exc, KeyboardInterrupt) else "stopped")
                report.reason = f"准备宠物组“{group}”时停止：{exc}"
                (job.run_dir / "summary.json").write_text(
                    json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise
    return job.run()
