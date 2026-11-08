"""Watch dragon hosting and recover the game through visually confirmed pages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
import re
import time

from .adb import AutomationError
from .cooperation import TaskInterrupted
from .dragon_schedule import DEFAULT_START_TIMES, beijing_now, is_dragon_active, next_dragon_start, parse_start_times
from .dragon_vision import DragonVision
from .runtime import ScriptContext
from .run_logs import RunLogOwner
from .vision import Match, VisualFrame


class _OutsideDragonTime(Exception):
    """Unwind an in-progress game flow before waiting for another session."""

    def __init__(self, current: datetime):
        super().__init__("当前不在屠龙时段。")
        self.current = current


@dataclass(frozen=True)
class DragonOptions:
    package: str = "com.monster.mxm1old"
    monitor_interval: float = 5
    poll_interval: float = .35
    min_click_interval: float = 1
    click_retry_interval: float = 2
    max_click_retries: int = 3
    login_timeout: float = 90
    town_timeout: float = 90
    cloud_timeout: float = 300
    transition_timeout: float = 30
    max_checks: int = 0
    max_runtime: float = 0
    start_times: list[str] = field(default_factory=lambda: list(DEFAULT_START_TIMES))

    def __post_init__(self):
        parse_start_times(self.start_times)
        if not isinstance(self.package, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+", self.package):
            raise AutomationError("slay_dragon.package 必须是有效的 Android 游戏包名。")
        for name in ("max_click_retries", "max_checks"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise AutomationError(f"slay_dragon.{name} 必须是非负整数。")
        for name in ("monitor_interval", "poll_interval", "min_click_interval", "click_retry_interval",
                     "login_timeout", "town_timeout", "cloud_timeout", "transition_timeout", "max_runtime"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < 0 or (name != "max_runtime" and value == 0)):
                raise AutomationError(f"slay_dragon.{name} 必须是有限的{'非负' if name == 'max_runtime' else '正'}数。")


@dataclass
class DragonReport:
    status: str = "running"
    checks: int = 0
    restarts: int = 0
    start_clicks: int = 0
    scheduled_waits: int = 0
    next_start: str = ""
    reason: str = ""


class SlayDragon(RunLogOwner):
    # Its opening-time decision must precede preparation game input.
    defer_loadout_preparation = True

    def __init__(self, ctx: ScriptContext):
        self.ctx = ctx
        shared = ctx.settings.get("daily_hero", {})
        values = {name: shared[name] for name in (
            "poll_interval", "min_click_interval", "click_retry_interval", "max_click_retries",
        ) if name in shared}
        values.update(ctx.settings.get("slay_dragon", {}))
        self.options = DragonOptions(**values)
        self.vision = DragonVision(ctx.project_root / "assets" / "slay_dragon")
        self._init_run_log(ctx, "Slay_Dragon")
        self.report = DragonReport()
        self.started = time.monotonic()
        self.last_click_at: float | None = None
        self.frame: VisualFrame | None = None

    def event(self, kind: str, **details) -> None:
        with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": datetime.now().astimezone().isoformat(),
                                     "event": kind, **details}, ensure_ascii=False) + "\n")

    def check_runtime(self) -> None:
        if self.options.max_runtime and time.monotonic() - self.started >= self.options.max_runtime:
            raise AutomationError("slay_dragon 已达到最长运行时间。")

    def ensure_active(self) -> None:
        self.check_runtime()
        if self.ctx.dry_run:
            return
        current = beijing_now()
        if not is_dragon_active(current, self.options.start_times):
            raise _OutsideDragonTime(current)

    def observe(self) -> VisualFrame:
        self.ensure_active()
        self.frame = VisualFrame.from_screenshot(self.ctx.capture())
        self.ensure_active()
        return self.frame

    def find(self, name: str, frame: VisualFrame | None = None) -> Match | None:
        return self.vision.find(frame or self.frame, name)

    def tap(self, match: Match, description: str, *, offset: tuple[int, int] = (0, 0)) -> None:
        self.ensure_active()
        x, y = match.center
        point = self.frame.device_point((x + offset[0], y + offset[1]))
        self.ctx.log.info("%s [匹配 %.3f]", description, match.score)
        self.event("tap", description=description, template=match.name,
                   score=round(match.score, 4), point=point)
        self.ctx.tap(*point)
        self.last_click_at = time.monotonic()

    def hosting_control(self, frame: VisualFrame) -> Match | None:
        return self.find("stop_hosting", frame) or self.find("start_hosting", frame)

    def login_notice_control(self, frame: VisualFrame) -> Match | None:
        return self.find("announcement_close", frame) or self.find("notice", frame)

    def town_control(self, frame: VisualFrame) -> Match | None:
        return (self.login_notice_control(frame) or self.find("announcement_page", frame)
                or self.find("town_eye", frame) or self.find("town", frame))

    def uncovered_town(self, frame: VisualFrame) -> Match | None:
        if self.login_notice_control(frame) or self.find("announcement_page", frame):
            return None
        return self.find("town_eye", frame) or self.find("town", frame)

    def prepare_dragon_loadout(self, *, force: bool = False, deadline: float | None = None) -> None:
        if not force and getattr(self.ctx, "active_pet_loadout", None) == "屠龙":
            return
        from .pet_loadouts import switch_pet_loadout

        self.ensure_active()
        original_control = getattr(self.ctx, "control", None)
        had_control = hasattr(self.ctx, "control")

        def check_loadout_input(operation):
            # ScriptContext checks this before every capture, input and wait.
            # Keep both the dragon schedule and the UI pause gate active while
            # the shared preparation workflow is running on our behalf.
            self.ensure_active()
            if deadline is not None and time.monotonic() >= deadline:
                raise AutomationError(f"处理登录通知并进入屠龙战超时（共享 {self.options.cloud_timeout:g} 秒），已停止并保存现场。")
            if original_control is not None:
                original_control(operation)

        self.ctx.control = check_loadout_input
        if self.last_click_at is not None:
            self.ctx.pet_loadout_last_click_at = self.last_click_at
        try:
            switch_pet_loadout(self.ctx, "屠龙")
        finally:
            if had_control:
                self.ctx.control = original_control
            else:
                del self.ctx.control
        self.last_click_at = getattr(self.ctx, "pet_loadout_last_click_at", self.last_click_at)
        self.ensure_active()

    def wait_for(self, predicate, description: str, timeout: float):
        deadline = time.monotonic() + timeout
        while True:
            frame = self.observe()
            if result := predicate(frame):
                return result
            if time.monotonic() >= deadline:
                self.ensure_active()
                raise AutomationError(f"等待“{description}”超时（{timeout:g} 秒），已停止并保存现场。")
            self.ctx.wait(self.options.poll_interval)

    def click_until(self, locate, arrived, description: str, *, timeout: float | None = None,
                    repeat: bool = False,
                    offset: tuple[int, int] | Callable[[Match], tuple[int, int]] = (0, 0)):
        timeout = self.options.transition_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        attempts: dict[str, int] = {}
        retry_at: dict[str, float] = {}
        while True:
            frame = self.observe()
            if result := arrived(frame):
                return result
            now = time.monotonic()
            if now >= deadline:
                self.ensure_active()
                raise AutomationError(f"等待“{description}”超时（{timeout:g} 秒），已停止并保存现场。")
            if self.last_click_at is None or now - self.last_click_at >= self.options.min_click_interval:
                control = locate(frame)
                if control is not None and now >= retry_at.get(control.name, 0):
                    self.ensure_active()
                    count = attempts.get(control.name, 0)
                    if not repeat and count > self.options.max_click_retries:
                        raise AutomationError(f"“{description}”补点 {self.options.max_click_retries} 次后仍未成功，已停止。")
                    if count:
                        self.event("click_retry", description=description, template=control.name, retry=count)
                    click_offset = offset(control) if callable(offset) else offset
                    self.tap(control, description, offset=click_offset)
                    attempts[control.name] = count + 1
                    retry_at[control.name] = self.last_click_at + self.options.click_retry_interval
            self.ctx.wait(self.options.poll_interval)

    def wait_for_start(self, *, current: datetime | None = None) -> None:
        """Wait on the Beijing system clock without interacting with the game."""
        self.check_runtime()
        current = beijing_now() if current is None else current
        target = next_dragon_start(current, self.options.start_times)
        self.report.scheduled_waits += 1
        self.report.next_start = target.isoformat()
        self.ctx.log.info("当前不在屠龙时段；北京时间 %s，等待 %s 到点重启。",
                          current.strftime("%Y-%m-%d %H:%M:%S"), target.strftime("%Y-%m-%d %H:%M:%S"))
        self.event("dragon_wait_start", current=current.isoformat(), next_start=target.isoformat())
        while True:
            self.check_runtime()
            remaining = (target - beijing_now()).total_seconds()
            if remaining <= 0:
                self.event("dragon_wait_complete", next_start=target.isoformat())
                return
            # Re-read the wall clock after each short sleep so system time
            # adjustments are respected. Ctrl+C and the runtime limit remain active.
            delay = min(remaining, 30)
            if self.options.max_runtime:
                delay = min(delay, self.options.max_runtime - (time.monotonic() - self.started))
            if delay > 0:
                self.ctx.wait(delay)

    def recover(self) -> None:
        self.ensure_active()
        self.ctx.log.info("重启游戏并恢复屠龙托管。")
        self.event("recovery_start")
        # Login restores a game scene, not proof that this run's loadout is
        # still selected. Reconfirm it only after all covering notices close.
        self.ctx.active_pet_loadout = None
        self.ctx.restart_app(self.options.package)
        self.report.restarts += 1
        self.event("game_restarted", package=self.options.package)
        self.wait_for(lambda f: self.find("login_enter", f), "登录页的进入按钮", self.options.login_timeout)
        # The login notice covers the town. Seeing it also confirms arrival;
        # dismiss it in the same timed loop that handles the dragon entrance.
        self.click_until(
            lambda f: self.find("login_enter", f),
            self.town_control,
            "进入游戏并等待天星镇", timeout=self.options.town_timeout,
        )
        self.enter_dragon()
        self.event("recovery_complete")

    def enter_dragon(self) -> None:
        """Resume at step 3 from town, sharing its notice/entrance deadline."""
        self.ctx.log.info("循环检测登录通知和“屠龙战”入口，共享 %g 秒超时。", self.options.cloud_timeout)
        deadline = time.monotonic() + self.options.cloud_timeout
        if getattr(self.ctx, "active_pet_loadout", None) != "屠龙":
            self.click_until(
                self.login_notice_control, self.uncovered_town,
                "处理登录通知，准备切换屠龙宠物组", timeout=self.options.cloud_timeout,
            )
            self.prepare_dragon_loadout(deadline=deadline)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AutomationError(f"处理登录通知并进入屠龙战超时（共享 {self.options.cloud_timeout:g} 秒），已停止并保存现场。")
        self.click_until(
            lambda f: self.login_notice_control(f) or self.find("dragon_entry", f),
            lambda f: self.hosting_control(f) or self.find("dragon_npc", f) or self.find("quest_mark", f),
            # Notices use their safe lower edge. Only the entrance label is
            # offset to the icon, on each fresh match including retries.
            "处理登录通知并进入屠龙战", timeout=remaining,
            offset=lambda control: (0, -130) if control.name == "dragon_entry" else (0, 0),
        )
        if not self.hosting_control(self.frame):
            self.click_until(
                lambda f: self.find("dragon_npc", f),
                lambda f: self.hosting_control(f) or (
                    self.find("quest_mark", f) if not self.find("dragon_npc", f) else None
                ),
                "靠近斯波纳克，等待黄色感叹号", repeat=True,
            )
            self.click_until(
                lambda f: self.find("quest_mark", f),
                self.hosting_control,
                "点击黄色感叹号，打开屠龙助手",
            )
        self.event("dragon_ready")

    def step(self) -> None:
        self.report.checks += 1
        frame = self.observe()
        if self.find("stop_hosting", frame):
            self.event("hosting_active", check=self.report.checks)
        elif start := self.find("start_hosting", frame):
            # Deliberately one tap, without a confirmation/retry loop. The next
            # five-second monitoring check will observe whether it took effect.
            self.tap(start, "点击开始托管（本轮仅点击一次）")
            self.report.start_clicks += 1
            self.event("hosting_start_clicked", check=self.report.checks)
        elif self.town_control(frame):
            self.event("town_detected", check=self.report.checks)
            self.enter_dragon()
        else:
            self.recover()

    def preview(self, *, current: datetime | None = None) -> None:
        current = beijing_now() if current is None else current
        frame = self.observe()
        if not is_dragon_active(current, self.options.start_times):
            target = next_dragon_start(current, self.options.start_times)
            planned = f"当前不在屠龙时段：正式运行时等待北京时间 {target:%Y-%m-%d %H:%M:%S} 后重启。"
        elif self.find("stop_hosting", frame):
            planned = "“停止托管”按钮可见：保持当前托管，5 秒后再检查。"
        elif self.find("start_hosting", frame):
            planned = "开始托管按钮可见：正式运行时单击一次，5 秒后再检查。"
        elif self.town_control(frame):
            planned = "天星镇可见：正式运行时从第3步进入屠龙战。"
        else:
            planned = "未识别托管按钮：正式运行时重启游戏并恢复屠龙流程。"
        self.ctx.log.info("[预演] %s", planned)
        self.event("preview", planned=planned)
        self.report.status = "preview"

    def run(self) -> DragonReport:
        self.ctx.log.info("slay_dragon 开始；诊断目录：%s", self.run_dir)
        try:
            # Decide the startup branch before any capture or game input.
            current = beijing_now()
            active = is_dragon_active(current, self.options.start_times)
            self.event("startup_schedule", current=current.isoformat(), active=active)
            if self.ctx.dry_run:
                self.preview(current=current)
                return self.report
            needs_restart = not active
            startup_loadout_pending = active
            wait_current = current
            while True:
                try:
                    if needs_restart:
                        self.wait_for_start(current=wait_current)
                        self.recover()
                        needs_restart = False
                        startup_loadout_pending = False
                    self.ensure_active()
                    if startup_loadout_pending:
                        self.prepare_dragon_loadout(force=True)
                        startup_loadout_pending = False
                    self.step()
                except _OutsideDragonTime as closed:
                    # Discard the interrupted flow and its short deadlines.
                    # At the next opening, rebuild the whole flow via restart.
                    self.event("session_closed", current=closed.current.isoformat())
                    wait_current = closed.current
                    needs_restart = True
                    startup_loadout_pending = False
                    continue
                if self.options.max_checks and self.report.checks >= self.options.max_checks:
                    self.report.status = "limited"
                    self.report.reason = f"已完成 {self.options.max_checks} 次监测。"
                    return self.report
                self.ctx.wait(self.options.monitor_interval)
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
