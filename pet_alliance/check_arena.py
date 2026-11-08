"""Check Demon City guardians in priority order and record confirmed ownership."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
import json

from .adb import AutomationError
from .arena_state import persist_arena_state
from .arena_hero import ArenaHeroPrerequisite, PROGRESS_COUNTS, PROGRESS_LISTS, empty_hero_progress
from .check_arena_vision import CheckArenaVision
from .cooperation import TaskInterrupted
from .daily_hero import DailyHero, HeroOptions
from .dragon_schedule import beijing_now
from .guardian_time import GuardianTimeReader, guard_expiration
from .runtime import ScriptContext
from .vision import Match, VisualFrame


DEFAULT_ARENAS = (
    "COCO号", "恶魔城", "鬼木断崖", "瘴气科研", "病态虫巢", "绝望监狱",
    "冥火特快", "死亡墓地", "永夜镇", "荒弃圣堂",
)


@dataclass(frozen=True)
class CheckArenaOptions(HeroOptions):
    arena_list: list[str] = field(default_factory=lambda: list(DEFAULT_ARENAS))
    player_name: str = "1111"

    def __post_init__(self):
        super().__post_init__()
        if (not isinstance(self.arena_list, list) or not self.arena_list
                or any(not isinstance(name, str) or not name.strip() for name in self.arena_list)):
            raise AutomationError("check_arena.arena_list 必须是包含非空副本名的字符串列表。")
        if len(set(self.arena_list)) != len(self.arena_list):
            raise AutomationError("check_arena.arena_list 不能包含重复副本。")
        if not isinstance(self.player_name, str) or not self.player_name.strip():
            raise AutomationError("check_arena.player_name 必须是非空的完整玩家名。")


@dataclass
class CheckArenaReport:
    status: str = "running"
    checked: list[dict] = field(default_factory=list)
    dungeon: str = ""
    claimed: bool = False
    claimed_at: str = ""
    guard_seconds: int | None = None
    observed_at: str = ""
    arena_expired_time: str = ""
    config_file: str = ""
    returned_to_town: bool = False
    hero_checks: list[dict] = field(default_factory=list)
    hero_progress: dict = field(default_factory=dict)
    hero_progress_date: str = ""
    hero_progress_by_day: dict = field(default_factory=dict)
    reason: str = ""


class _HeroProofExpired(AutomationError):
    pass


class CheckArena(DailyHero):
    # Standalone arena inspection can verify a completed hero panel, but only
    # the scheduler may run prerequisite battles on the user's behalf.
    allow_hero_battle = False

    def __init__(self, ctx: ScriptContext):
        values = {**ctx.settings.get("daily_hero", {}), **ctx.settings.get("check_arena", {})}
        values["points_arena_enabled"] = ctx.settings.get("daily_hero", {}).get("points_arena_enabled", True)
        # Persisted observations are outputs, never proof of current ownership.
        values.pop("arena_expired_time", None)
        values.pop("arena_dungeon", None)
        options = CheckArenaOptions(**values)
        super().__init__(ctx, options=options, run_name="Check_Arena")
        self.vision = CheckArenaVision(ctx.project_root / "assets" / "check_arena")
        unknown = set(options.arena_list) - set(self.vision.dungeon_names)
        if unknown:
            raise AutomationError("check_arena.arena_list 含未知副本：" + "、".join(sorted(unknown)))
        self.vision.validate_player(options.player_name)
        self.time_reader = GuardianTimeReader()
        self.report = CheckArenaReport()
        self.frame_time: datetime | None = None
        self.hero_verified_date: str | None = None
        self._claim_proof_date: str | None = None
        self._claim_attempt_at: datetime | None = None

    @staticmethod
    def now() -> datetime:
        return beijing_now()

    def observe(self) -> VisualFrame:
        frame = super().observe()
        self.frame_time = self.now()
        return frame

    def arena_find(self, name: str, frame: VisualFrame | None = None):
        return self.vision.find(frame or self.frame, name)

    def guardian_owner(self, frame: VisualFrame):
        return self.vision.guardian_owner(frame, self.options.player_name)

    def prepare_map(self) -> None:
        self.world()
        region = next(r for r in self.bank.profile["regions"] if r["key"] == "demon")
        self.enter_region(region)
        self.wait_for(lambda f: self.arena_find("map_anchor", f), "魔都区域地图")

    def inspect_dungeon(self, name: str) -> str:
        self.click_until(
            lambda f: self.vision.dungeon(f, name) if self.arena_find("map_anchor", f) else None,
            lambda f: (self.arena_find("dungeon_panel", f)
                       or (self.find("panel_close", f) and self.find("hero_dragon", f))),
            f"进入副本：{name}",
        )
        frame = self.observe()
        if self.find("panel_close", frame) and self.find("hero_dragon", frame):
            self.click_until(
                lambda f: self.find("hero_dragon", f) if self.find("panel_close", f) else None,
                lambda f: self.arena_find("dungeon_panel", f) and self.arena_find("arena_button", f),
                f"切回{name}普通模式以检查擂主",
            )
        return self.click_until(
            lambda f: self.arena_find("arena_button", f) if self.arena_find("dungeon_panel", f) else None,
            self.guardian_owner, f"打开{name}的擂主页面",
        )

    def ensure_hero_prerequisite(self, name: str) -> bool:
        self.hero_verified_date = None
        helper = ArenaHeroPrerequisite(self.ctx, now=self.now)
        helper.last_click_at = self.last_click_at
        check = {"dungeon": name, "allow_battle": self.allow_hero_battle,
                 "run_dir": str(helper.run_dir)}
        self.report.hero_checks.append(check)
        try:
            cleared = helper.ensure_cleared(name, allow_battle=self.allow_hero_battle)
            self.hero_verified_date = helper.verified_date
            check.update(cleared=cleared, verified_date=helper.verified_date)
            return cleared
        finally:
            self.collect_hero_progress(helper, check)

    def collect_hero_progress(self, helper: ArenaHeroPrerequisite, check: dict) -> None:
        self.frame = helper.frame or self.frame
        self.last_click_at = helper.last_click_at
        check["status"] = helper.report.status
        check["reason"] = helper.report.reason
        for day, progress in helper.progress_by_day.items():
            bucket = self.report.hero_progress_by_day.setdefault(day, empty_hero_progress())
            for key in PROGRESS_LISTS:
                bucket[key].extend(progress[key])
            for key in PROGRESS_COUNTS:
                bucket[key] += progress[key]
            bucket["completed_since_arena"] = progress["completed_since_arena"]
            bucket["points_arena_pending"] = progress["points_arena_pending"]
        today = self.now().date().isoformat()
        self.report.hero_progress_date = today
        self.report.hero_progress = self.report.hero_progress_by_day.get(today, empty_hero_progress())

    def resume_hero_points(self) -> None:
        if not self.options.points_arena_enabled:
            # Keep the arena ownership task enabled while clearing only the
            # hero script's deferred points visit in shared scheduler progress.
            today = self.now().date().isoformat()
            self.ctx.hero_resume_date = today
            self.ctx.hero_resume_data = {"completed_since_arena": 0, "points_arena_pending": False}
            bucket = self.report.hero_progress_by_day.setdefault(today, empty_hero_progress())
            bucket["completed_since_arena"] = 0
            bucket["points_arena_pending"] = False
            self.report.hero_progress_date = today
            self.report.hero_progress = bucket
            return
        resume = getattr(self.ctx, "hero_resume_data", {})
        if (not isinstance(resume, dict) or resume.get("points_arena_pending") is not True
                or getattr(self.ctx, "hero_resume_date", None) != self.now().date().isoformat()):
            return
        helper = ArenaHeroPrerequisite(self.ctx, now=self.now)
        helper.last_click_at = self.last_click_at
        check = {"kind": "resume_points_arena", "run_dir": str(helper.run_dir)}
        self.report.hero_checks.append(check)
        try:
            helper.resume_pending_points()
        finally:
            self.collect_hero_progress(helper, check)

    def prepare_claim(self, name: str) -> str:
        self.close_guardian()
        if not self.ensure_hero_prerequisite(name):
            return "hero_required"
        if self.hero_verified_date is None:
            raise AutomationError("英雄级核验缺少当日通关凭证，未点击占擂。")
        return self.inspect_dungeon(name)

    def tap(self, match, description, offset=(0, 0)) -> None:
        claim_time = None
        if match.name == "challenge_guardian":
            if self._claim_proof_date is None:
                raise AutomationError("缺少英雄级当日通关凭证，未点击占擂。")
            if self.now().date().isoformat() != self._claim_proof_date:
                raise _HeroProofExpired("英雄级通关凭证已跨零点，需要重新核验。")
            claim_time = self.now()
        super().tap(match, description, offset)
        if claim_time is not None and self._claim_attempt_at is None:
            # First sent input is a conservative start bound: a later retry
            # may follow a successful claim whose UI has not refreshed yet.
            self._claim_attempt_at = claim_time

    def claim_empty(self, name: str) -> str:
        # Revalidate ownership after hero inspection or battle: other players
        # can occupy this arena while the prerequisite is being completed.
        for _ in range(4):
            owner = self.prepare_claim(name)
            if owner != "empty":
                return owner
            proof = self.hero_verified_date
            if proof is None:
                raise AutomationError("缺少英雄级当日通关凭证，未点击占擂。")
            if self.now().date().isoformat() != proof:
                continue
            actions_before = self.actions
            self._claim_attempt_at = None

            def locate(frame):
                if self.now().date().isoformat() != proof:
                    raise _HeroProofExpired("英雄级通关凭证已跨零点，需要重新核验。")
                if self.arena_find("claim_success", frame):
                    return None
                return self.arena_find("challenge_guardian", frame) if self.guardian_owner(frame) == "empty" else None

            def claim_response(frame):
                if self.arena_find("claim_success", frame):
                    return "notice"
                owner = self.guardian_owner(frame)
                if owner in {"self", "other"}:
                    return owner
                if (self.actions > actions_before and owner is None
                        and self.arena_find("guardian_panel", frame)):
                    return "unrefreshed"
                return None

            self._claim_proof_date = proof
            try:
                owner = self.click_until(
                    locate, claim_response,
                    f"占领无人擂台：{name}",
                )
            except _HeroProofExpired:
                continue
            finally:
                self._claim_proof_date = None
            challenge_sent = self.actions > actions_before
            if not challenge_sent and owner != "notice":
                return owner
            if owner == "notice":
                self.event("guardian_claim_notice", dungeon=name)

                def dismiss(frame):
                    notice = self.arena_find("claim_success", frame)
                    # Blank space below the notice and guardian controls,
                    # above the screen's "点击空白处关闭" label.
                    return Match("dismiss_claim_success", 950, 960, 20, 20,
                                 notice.score) if notice else None

                owner = self.click_until(
                    dismiss,
                    lambda f: self.arena_find("guardian_panel", f)
                    if not self.arena_find("claim_success", f) else None,
                    f"关闭{name}占擂成功提示",
                )

            # A freshly claimed guardian panel may omit its name and duration.
            # Close both panels and verify the target's fresh map flag instead.
            self.close_guardian()

            def map_owner(frame):
                if not self.arena_find("map_anchor", frame):
                    return None
                state = self.vision.map_owner(frame, name, self.options.player_name)
                return state if state in {"self", "other"} else None

            owner = self.wait_for(map_owner, f"返回区域地图确认{name}占擂结果")
            if owner == "self":
                if not challenge_sent:
                    # Another input source claimed it before our first tap.
                    # Reopen for its actual elapsed time, never assume six hours.
                    return self.inspect_dungeon(name)
                if self._claim_attempt_at is None:
                    raise AutomationError("占擂成功但缺少挑战点击时间，未保存守护期限。")
                self.report.claimed = challenge_sent
                self.report.claimed_at = self._claim_attempt_at.isoformat(timespec="seconds")
                self.event("guardian_claim_confirmed", dungeon=name, claimed=True,
                           confirmed_on="map", claimed_at=self.report.claimed_at)
            return owner
        raise AutomationError("占擂前置核验反复跨越北京时间零点，已停止。")

    def record_new_claim_expiration(self, name: str) -> None:
        frame = self.observe()
        if (not self.report.claimed or not self.report.claimed_at
                or not self.arena_find("map_anchor", frame)
                or self.vision.map_owner(frame, name, self.options.player_name) != "self"):
            raise AutomationError("区域地图未能确认本次占擂，未保存守护期限。")
        expired_at = guard_expiration(datetime.fromisoformat(self.report.claimed_at), 0) + timedelta(minutes=1)
        path = persist_arena_state(self.ctx, expired_at, name)
        self.report.observed_at = self.frame_time.isoformat(timespec="seconds")
        self.report.arena_expired_time = expired_at.isoformat(timespec="seconds")
        self.report.config_file = str(path)
        self.event("arena_expiration_saved", dungeon=name, source="claim_time",
                   claimed_at=self.report.claimed_at, observed_at=self.report.observed_at,
                   arena_expired_time=self.report.arena_expired_time, config_file=str(path))
        self.ctx.log.info("区域地图已确认%s占擂成功；按首次挑战时间加六小时01分安排下次检查：%s",
                          name, self.report.arena_expired_time)

    def record_expiration(self, name: str) -> None:
        frame = self.observe()
        if self.guardian_owner(frame) != "self":
            raise AutomationError("读取守护时间前未能确认本人仍为擂主，未修改配置。")
        observed_at = self.frame_time
        elapsed = self.time_reader.read(frame)
        if type(elapsed) is not int or elapsed < 0:
            self.event("guardian_time_unreadable", dungeon=name,
                       text=self.time_reader.last_text, error=self.time_reader.last_error)
            raise AutomationError("无法完整识别已守护时间，未修改 arena_expired_time。")
        expired_at = guard_expiration(observed_at, elapsed)
        self.report.guard_seconds = elapsed
        self.report.observed_at = observed_at.isoformat(timespec="seconds")
        path = persist_arena_state(self.ctx, expired_at, name)
        self.report.arena_expired_time = expired_at.isoformat(timespec="seconds")
        self.report.config_file = str(path)
        self.event("arena_expiration_saved", dungeon=name, guard_seconds=elapsed,
                   observed_at=self.report.observed_at, arena_expired_time=self.report.arena_expired_time,
                   config_file=str(path))
        self.ctx.log.info("%s已守护 %d 秒，守护结束时间（北京时间）：%s",
                          name, elapsed, self.report.arena_expired_time)

    def close_guardian(self) -> None:
        if self.is_map(self.observe()):
            return
        self.click_until(
            lambda f: self.arena_find("guardian_close", f),
            lambda f: (self.is_map(f) or self.find("panel_close", f))
            if not self.arena_find("guardian_panel", f) else None,
            "关闭关卡守护神页面",
        )
        if self.find("panel_close"):
            self.click_until(lambda f: self.find("panel_close", f), self.is_map, "关闭副本面板")
        self.wait_for(self.is_map, "返回魔都区域地图")

    def finish_in_town(self) -> None:
        self.return_to_town()
        self.report.returned_to_town = True
        self.event("check_arena_returned_to_town")

    def preview(self) -> None:
        frame = self.observe()
        self.ctx.log.info("[预演] 魔都副本优先级：%s；本人：%s",
                          " → ".join(self.options.arena_list), self.options.player_name)
        if self.arena_find("map_anchor", frame):
            for name in self.options.arena_list:
                owner = self.vision.map_owner(frame, name, self.options.player_name)
                self.report.checked.append({"dungeon": name, "owner": owner or "unknown"})
                self.ctx.log.info("[预演] %s：%s", name, owner or "无法确认")
        elif self.arena_find("guardian_panel", frame):
            self.ctx.log.info("[预演] 当前守护页面擂主归属：%s", self.guardian_owner(frame) or "无法确认")
        elif not (self.is_world(frame) or self.is_map(frame) or self.find("adventure", frame)):
            raise AutomationError("未识别到天星镇、地图或关卡守护神页面。")
        self.report.status = "preview"

    def run(self) -> CheckArenaReport:
        self.ctx.log.info("check_arena开始；诊断目录：%s", self.run_dir)
        self.ctx.log.info("本次擂主检查优先级：%s", " → ".join(self.options.arena_list))
        try:
            self.event("check_arena_start", arena_list=self.options.arena_list,
                       player_name=self.options.player_name)
            if self.ctx.dry_run:
                self.preview()
                return self.report
            if not getattr(self.options, "points_arena_enabled", True):
                self.resume_hero_points()
            self.prepare_map()
            for index, name in enumerate(self.options.arena_list, start=1):
                self.ctx.log.info("检查擂主 [%d/%d]：%s", index, len(self.options.arena_list), name)
                owner = self.wait_for(
                    lambda f: self.vision.map_owner(f, name, self.options.player_name), f"确认{name}的擂主",
                )
                entry = {"dungeon": name, "owner": owner}
                self.report.checked.append(entry)
                self.event("guardian_checked", **entry)
                if owner == "other":
                    self.ctx.log.info("%s：擂主为其他玩家，跳过。", name)
                    continue
                if owner == "self":
                    self.ctx.log.info("%s：地图显示擂主为本人，进入确认守护时间。", name)
                else:
                    self.ctx.log.info("%s：地图暂未发现擂主，进入页面再次确认。", name)
                owner = self.inspect_dungeon(name)
                entry["detail_owner"] = owner
                if owner == "empty":
                    owner = self.claim_empty(name)
                    entry["after_challenge"] = owner
                if owner == "hero_required":
                    entry["reason"] = "当天英雄级尚未通过"
                    self.ctx.log.info("%s：当天英雄级尚未通过，跳过占擂。", name)
                    continue
                if owner == "other":
                    self.ctx.log.info("%s：页面内擂主已变为其他玩家，关闭并跳过。", name)
                    self.close_guardian()
                    continue
                if owner != "self":
                    raise AutomationError("未能确认关卡擂主归属，已停止。")
                self.report.dungeon = name
                if not self.report.claimed:
                    self.record_expiration(name)
                self.close_guardian()
                self.finish_in_town()
                self.report.status = "completed"
                self.report.reason = "已确认成为擂主并返回天星镇。" if self.report.claimed else "已保存守护结束时间并返回天星镇。"
                return self.report
            self.finish_in_town()
            if any(entry.get("after_challenge") == "hero_required" for entry in self.report.checked):
                self.report.status = "blocked"
                raise AutomationError("arena_list中暂无可占擂台，部分副本当天英雄级尚未通过；已返回天星镇。")
            self.report.status = "exhausted"
            raise AutomationError("arena_list中的所有副本均由其他玩家守护；已返回天星镇。")
        except TaskInterrupted as exc:
            self.report.status = exc.status
            self.report.reason = exc.reason
            raise
        except KeyboardInterrupt:
            self.report.status = "interrupted"
            self.report.reason = "用户按Ctrl+C停止。"
            raise
        except Exception as exc:
            if self.report.status not in {"exhausted", "blocked"}:
                self.report.status = "stopped"
            self.report.reason = str(exc)
            raise
        finally:
            if self.frame is not None:
                self.frame.save(self.run_dir / "last_frame.jpg")
            (self.run_dir / "summary.json").write_text(
                json.dumps(asdict(self.report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            self.ctx.log.info("check_arena：%s；副本：%s；已返回天星镇：%s。",
                              self.report.status, self.report.dungeon or "无", self.report.returned_to_town)
