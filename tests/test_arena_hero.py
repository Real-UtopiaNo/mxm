from __future__ import annotations

from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from pet_alliance.adb import AutomationError
from pet_alliance.arena_hero import ArenaHeroPrerequisite
from pet_alliance.check_arena import CheckArena, CheckArenaReport
from pet_alliance.cooperation import TaskInterrupted
from pet_alliance.daily_hero import DailyHero
from pet_alliance.dragon_schedule import BEIJING
from pet_alliance.scheduler_jobs import ManagedCheckArena
from pet_alliance.vision import Match
from tests.test_check_arena import BOUNDS, Page, ReplayContext


ROOT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 11, 6, 23, 30, tzinfo=BEIJING)


class Frame:
    def __init__(self, state):
        self.state = state

    def save(self, path):
        Path(path).write_text(self.state, encoding="utf-8")


class HeroHarness(ArenaHeroPrerequisite):
    """Drive the real exact-entry and verification flow with explicit states."""

    def __init__(self, directory, states, *, counter=0, resume_date=None, pending=False):
        self.clock = NOW
        ctx = types.SimpleNamespace(
            project_root=ROOT, settings={}, log=logging.getLogger("arena-hero-test"),
            hero_resume_date=resume_date or NOW.date().isoformat(),
            hero_resume_data={"completed_since_arena": counter, "points_arena_pending": pending},
            active_pet_loadout="刷图",
        )
        super().__init__(ctx, now=lambda: self.clock)
        self.run_dir.rmdir()
        self.run_dir = Path(directory)
        self.page = Frame("map")
        self.states = iter(states)
        self.inputs = []
        self.entrances = []
        self.arena_vision = types.SimpleNamespace(
            dungeon_names=("COCO号", "恶魔城"),
            find=lambda frame, name: self.control("map_anchor") if frame.state == "map" else None,
            dungeon=self.exact_dungeon,
        )
        self.reset_vision = types.SimpleNamespace(
            hero_state=lambda frame: frame.state if frame.state in {"cleared", "incomplete"} else None,
        )

    @staticmethod
    def control(name):
        return Match(name, 100, 100, 30, 30, 1)

    def exact_dungeon(self, frame, name):
        self.entrances.append(name)
        return self.control(name)

    def observe(self):
        self.frame = self.page
        return self.page

    def find(self, name, frame=None):
        state = (frame or self.page).state
        if name == "panel_close" and state in {"cleared", "incomplete"}:
            return self.control(name)
        if name == "challenge" and state == "incomplete":
            return self.control(name)
        return None

    def wait_for(self, predicate, description):
        result = predicate(self.observe())
        if not result:
            raise AutomationError(description)
        return result

    def click_until(self, locate, arrived, description, **kwargs):
        if result := arrived(self.observe()):
            return result
        match = locate(self.page)
        if match is None:
            raise AutomationError(description)
        self.inputs.append(match.name)
        if match.name in {"COCO号", "恶魔城"}:
            self.page = Frame(next(self.states))
        elif match.name == "challenge":
            self.page = Frame("battle")
        else:
            raise AssertionError(f"Unexpected control {match.name}")
        result = arrived(self.observe())
        if not result:
            raise AssertionError(f"Unexpected destination after {match.name}")
        return result

    def battle_state(self, frame):
        return ("battle", self.control("skip")) if frame.state == "battle" else None

    def return_to_map(self):
        self.page = Frame("map")
        self.observe()

    def switch_battle_loadout(self, group):
        # This state-machine harness isolates the already-tested town route.
        self.ctx.active_pet_loadout = group
        self.return_to_map()

    def battle(self, region, target):
        self.open_dungeon(target)
        self.report.completed.append({"region": region["name"], "marker": target.center})
        self.report.victories += 5
        self.return_to_map()
        return True


class ArenaHeroTests(unittest.TestCase):
    def test_pvp_to_hero_changes_group_then_relocates_and_rechecks_exact_dungeon(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["incomplete", "incomplete", "cleared"])
            helper.ctx.active_pet_loadout = "PVP"
            switches = []
            original_switch = helper.switch_battle_loadout

            def switch(group):
                switches.append(group)
                original_switch(group)

            def fight(region, target):
                self.assertEqual(helper.ctx.active_pet_loadout, "刷图")
                self.assertEqual(target.name, "恶魔城")
                return helper.battle(region, target)

            with patch.object(helper, "switch_battle_loadout", side_effect=switch), \
                    patch.object(DailyHero, "fight_dungeon", side_effect=fight):
                self.assertTrue(helper.ensure_cleared("恶魔城", True))
            self.assertEqual(switches, ["刷图", "PVP"])
            self.assertEqual(helper.inputs, ["恶魔城", "恶魔城", "challenge", "恶魔城"])
            self.assertEqual(set(helper.entrances), {"恶魔城"})
            self.assertEqual(helper.ctx.active_pet_loadout, "PVP")

    def test_completed_hero_inspection_does_not_change_pvp_group(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["cleared"])
            helper.ctx.active_pet_loadout = "PVP"
            with patch.object(helper, "switch_battle_loadout") as switch:
                self.assertTrue(helper.ensure_cleared("COCO号", True))
            switch.assert_not_called()

    def test_paused_prerequisite_battle_never_sends_restore_group_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["incomplete", "incomplete"])
            helper.ctx.active_pet_loadout = "PVP"
            original_switch = helper.switch_battle_loadout
            with patch.object(helper, "switch_battle_loadout", side_effect=original_switch) as switch, \
                    patch.object(DailyHero, "fight_dungeon", side_effect=TaskInterrupted("paused", "暂停")):
                with self.assertRaises(TaskInterrupted):
                    helper.ensure_cleared("COCO号", True)
            self.assertEqual([call.args[0] for call in switch.call_args_list], ["刷图"])
            self.assertEqual(helper.ctx.active_pet_loadout, "刷图")
            self.assertEqual(helper.report.status, "paused")

    def test_failed_hero_battle_restores_pvp_before_continuing_arena_list(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["incomplete", "incomplete"])
            helper.ctx.active_pet_loadout = "PVP"

            def defeat(region, target):
                helper.return_to_map()
                return False

            original_switch = helper.switch_battle_loadout
            with patch.object(helper, "switch_battle_loadout", side_effect=original_switch) as switch, \
                    patch.object(DailyHero, "fight_dungeon", side_effect=defeat):
                self.assertFalse(helper.ensure_cleared("COCO号", True))
            self.assertEqual([call.args[0] for call in switch.call_args_list], ["刷图", "PVP"])
            self.assertEqual(helper.page.state, "map")
            self.assertEqual(helper.report.status, "hero_required")

    def test_loadout_route_returns_to_demon_city_and_clears_stale_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["cleared"])
            helper.panel_dungeon = "COCO号"

            def switch(context, group):
                context.active_pet_loadout = group
                context.pet_loadout_last_click_at = 12.5

            with patch("pet_alliance.pet_loadouts.switch_pet_loadout", side_effect=switch) as select, \
                    patch.object(helper, "world") as world, \
                    patch.object(helper, "enter_region") as enter:
                ArenaHeroPrerequisite.switch_battle_loadout(helper, "PVP")
            select.assert_called_once_with(helper.ctx, "PVP")
            world.assert_called_once()
            enter.assert_called_once_with(helper.demon_region)
            self.assertIsNone(helper.panel_dungeon)
            self.assertEqual(helper.last_click_at, 12.5)

    def test_already_cleared_inspects_exact_name_without_battle_or_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["cleared"])
            with patch.object(DailyHero, "fight_dungeon") as battle:
                self.assertTrue(helper.ensure_cleared("恶魔城", False))
            battle.assert_not_called()
            self.assertEqual(helper.inputs, ["恶魔城"])
            self.assertEqual(set(helper.entrances), {"恶魔城"})
            self.assertEqual(helper.verified_date, "2026-11-06")
            self.assertEqual(helper.page.state, "map")
            self.assertEqual(helper.report.completed, [])

    def test_standalone_does_not_start_incomplete_battle(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["incomplete"])
            with patch.object(DailyHero, "fight_dungeon") as battle:
                self.assertFalse(helper.ensure_cleared("COCO号", False))
            battle.assert_not_called()
            self.assertEqual(helper.inputs, ["COCO号"])
            self.assertEqual(helper.report.status, "hero_required")
            self.assertIsNone(helper.verified_date)

    def test_managed_battle_only_targets_requested_dungeon_and_rechecks_five_stamps(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["incomplete", "cleared"])
            with patch.object(DailyHero, "fight_dungeon", side_effect=helper.battle) as battle:
                self.assertTrue(helper.ensure_cleared("恶魔城", True))
            self.assertEqual(battle.call_count, 1)
            self.assertEqual(battle.call_args.args[1].name, "恶魔城")
            self.assertEqual(helper.inputs, ["恶魔城", "challenge", "恶魔城"])
            self.assertNotIn("reset", helper.inputs)
            self.assertNotIn("confirm_reset", helper.inputs)
            self.assertEqual(len(helper.report.completed), 1)
            self.assertEqual(helper.progress_by_day["2026-11-06"]["completed_since_arena"], 1)
            summary = json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["verified_date"], "2026-11-06")

    def test_unverified_battle_victory_does_not_grant_claim_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["incomplete", "incomplete"])
            with patch.object(DailyHero, "fight_dungeon", side_effect=helper.battle) as battle, \
                    patch.object(helper, "run_points_arena"):
                self.assertFalse(helper.ensure_cleared("COCO号", True))
            self.assertEqual(battle.call_count, 1)
            self.assertIsNone(helper.verified_date)

    def test_exhausted_battle_retries_do_not_grant_claim_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["incomplete"])

            def defeat(region, target):
                helper.report.failures += 11
                helper.report.skipped.append({"region": region["name"]})
                helper.return_to_map()
                return False

            with patch.object(DailyHero, "fight_dungeon", side_effect=defeat):
                self.assertFalse(helper.ensure_cleared("COCO号", True))
            self.assertIsNone(helper.verified_date)
            self.assertEqual(helper.progress_by_day["2026-11-06"]["failures"], 11)

    def test_shared_counter_triggers_normal_points_arena_after_third_dungeon(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["incomplete", "cleared"], counter=2)
            with patch.object(DailyHero, "fight_dungeon", side_effect=helper.battle), \
                    patch.object(helper, "run_points_arena") as arena:
                self.assertTrue(helper.ensure_cleared("COCO号", True))
            arena.assert_called_once_with(helper.demon_region)
            self.assertEqual(helper.ctx.hero_resume_data["completed_since_arena"], 0)
            self.assertEqual(helper.progress_by_day["2026-11-06"]["completed_since_arena"], 0)

    def test_yesterday_counter_is_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["incomplete", "cleared"], counter=2, resume_date="2026-11-05")
            with patch.object(DailyHero, "fight_dungeon", side_effect=helper.battle), \
                    patch.object(helper, "run_points_arena") as arena:
                self.assertTrue(helper.ensure_cleared("COCO号", True))
            arena.assert_not_called()
            self.assertEqual(helper.completed_since_arena, 1)

    def test_pause_during_third_completion_points_visit_keeps_pending_and_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["incomplete"], counter=2)
            with patch.object(DailyHero, "fight_dungeon", side_effect=helper.battle), \
                    patch.object(helper, "run_points_arena", side_effect=TaskInterrupted("paused", "暂停")):
                with self.assertRaises(TaskInterrupted):
                    helper.ensure_cleared("COCO号", True)
            self.assertEqual(len(helper.report.completed), 1)
            self.assertTrue(helper.ctx.hero_resume_data["points_arena_pending"])
            self.assertEqual(helper.ctx.hero_resume_data["completed_since_arena"], 2)
            self.assertTrue(helper.progress_by_day["2026-11-06"]["points_arena_pending"])

    def test_pending_points_visit_resumes_before_inspecting_next_hero(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["cleared"], counter=2, pending=True)
            timeline = []

            def points(region):
                timeline.append("points")
                self.assertEqual(helper.inputs, [])

            with patch.object(helper, "run_points_arena", side_effect=points) as arena:
                self.assertTrue(helper.ensure_cleared("COCO号", True))
            arena.assert_called_once_with(helper.demon_region)
            self.assertEqual(timeline, ["points"])
            self.assertFalse(helper.ctx.hero_resume_data["points_arena_pending"])
            self.assertEqual(helper.ctx.hero_resume_data["completed_since_arena"], 0)

    def test_yesterday_pending_points_is_not_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["cleared"], counter=2, pending=True, resume_date="2026-11-05")
            with patch.object(helper, "run_points_arena") as arena:
                self.assertTrue(helper.ensure_cleared("COCO号", True))
            arena.assert_not_called()
            self.assertFalse(helper.ctx.hero_resume_data["points_arena_pending"])

    def test_midnight_during_panel_close_reopens_and_rechecks(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["cleared", "incomplete"])
            original = helper.close_panel
            closes = []

            def close():
                original()
                closes.append(True)
                if len(closes) == 1:
                    helper.clock += timedelta(days=1)

            helper.close_panel = close
            self.assertFalse(helper.ensure_cleared("COCO号", False))
            self.assertEqual(helper.inputs, ["COCO号", "COCO号"])
            self.assertIsNone(helper.verified_date)

    def test_interrupted_battle_retains_partial_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            helper = HeroHarness(directory, ["incomplete"])

            def pause(region, target):
                helper.report.victories = 2
                raise TaskInterrupted("paused", "手动暂停")

            with patch.object(DailyHero, "fight_dungeon", side_effect=pause):
                with self.assertRaises(TaskInterrupted):
                    helper.ensure_cleared("COCO号", True)
            self.assertEqual(helper.report.status, "paused")
            self.assertEqual(helper.progress_by_day["2026-11-06"]["victories"], 2)


class ClaimProofTests(unittest.TestCase):
    def checker(self):
        checker = object.__new__(CheckArena)
        checker.report = CheckArenaReport()
        checker.actions = 0
        checker.event = Mock()
        checker.clock = NOW
        checker.now = lambda: checker.clock
        checker.hero_verified_date = None
        checker._claim_proof_date = None
        checker._claim_attempt_at = None
        checker.frame = Frame("guardian")
        checker.options = types.SimpleNamespace(player_name="1111")
        checker.vision = types.SimpleNamespace(map_owner=Mock(return_value="self"))
        checker.guardian_owner = lambda frame: "empty"
        checker.arena_find = lambda name, frame: (
            Match(name, 1, 1, 4, 4, 1)
            if name == "challenge_guardian"
            or name == "map_anchor" and frame.state == "map" else None)

        def close():
            checker.frame = Frame("map")

        def wait(predicate, description):
            result = predicate(checker.frame)
            if not result:
                raise AutomationError(description)
            return result

        checker.close_guardian = Mock(side_effect=close)
        checker.wait_for = Mock(side_effect=wait)
        return checker

    @staticmethod
    def sent_tap(checker):
        """Keep CheckArena.tap's proof/time guards; replace only device input."""
        def send(*args, **kwargs):
            checker.actions += 1
        return patch.object(DailyHero, "tap", side_effect=send)

    def test_owner_changed_during_prerequisite_is_never_challenged(self):
        checker = self.checker()
        checker.prepare_claim = Mock(return_value="other")
        checker.click_until = Mock()
        self.assertEqual(checker.claim_empty("COCO号"), "other")
        checker.click_until.assert_not_called()

    def test_missing_clear_proof_never_clicks(self):
        checker = self.checker()
        checker.prepare_claim = Mock(return_value="empty")
        checker.click_until = Mock()
        with self.assertRaises(AutomationError):
            checker.claim_empty("COCO号")
        checker.click_until.assert_not_called()

    def test_midnight_before_claim_repeats_prerequisite(self):
        checker = self.checker()
        preparations = []

        def prepare(name):
            preparations.append(name)
            checker.hero_verified_date = checker.now().date().isoformat()
            if len(preparations) == 1:
                checker.clock += timedelta(days=1)
            return "empty"

        checker.prepare_claim = prepare

        def click(locate, arrived, description):
            checker.tap(locate(checker.frame), description)
            return "unrefreshed"

        checker.click_until = Mock(side_effect=click)
        with self.sent_tap(checker):
            self.assertEqual(checker.claim_empty("COCO号"), "self")
        self.assertEqual(preparations, ["COCO号", "COCO号"])
        checker.click_until.assert_called_once()
        self.assertEqual(checker.report.claimed_at, checker.now().isoformat(timespec="seconds"))
        checker.close_guardian.assert_called_once()
        checker.vision.map_owner.assert_called_once_with(checker.frame, "COCO号", "1111")

    def test_midnight_before_supplementary_click_repeats_prerequisite(self):
        checker = self.checker()
        preparations = []
        clicks = []

        def prepare(name):
            preparations.append(name)
            checker.hero_verified_date = checker.now().date().isoformat()
            return "empty"

        def click(locate, arrived, description):
            control = locate(checker.frame)
            self.assertIsNotNone(control)
            checker.tap(control, description)
            clicks.append(checker.now().date().isoformat())
            if len(clicks) == 1:
                checker.clock += timedelta(days=1)
                locate(checker.frame)
            return "unrefreshed"

        checker.prepare_claim = prepare
        checker.click_until = click
        with self.sent_tap(checker):
            self.assertEqual(checker.claim_empty("COCO号"), "self")
        self.assertEqual(preparations, ["COCO号", "COCO号"])
        self.assertEqual(clicks, ["2026-11-06", "2026-11-07"])
        self.assertTrue(checker.report.claimed)
        self.assertEqual(checker.report.claimed_at, checker.now().isoformat(timespec="seconds"))

    def test_claim_timestamp_keeps_first_sent_click_before_supplement_and_map_confirmation(self):
        checker = self.checker()
        checker.hero_verified_date = NOW.date().isoformat()
        checker.prepare_claim = Mock(return_value="empty")

        def click(locate, arrived, description):
            checker.tap(locate(checker.frame), description)
            checker.clock += timedelta(seconds=10)
            checker.tap(locate(checker.frame), description)
            checker.clock += timedelta(seconds=20)
            return "unrefreshed"

        checker.click_until = click
        with self.sent_tap(checker):
            self.assertEqual(checker.claim_empty("COCO号"), "self")
        self.assertEqual(checker.actions, 2)
        self.assertEqual(checker.report.claimed_at, NOW.isoformat(timespec="seconds"))
        self.assertEqual(checker.now(), NOW + timedelta(seconds=30))
        self.assertTrue(checker.report.claimed)
        checker.vision.map_owner.assert_called_once_with(checker.frame, "COCO号", "1111")

    def test_standalone_and_managed_have_distinct_battle_permission(self):
        self.assertFalse(CheckArena.allow_hero_battle)
        self.assertTrue(ManagedCheckArena.allow_hero_battle)

    def test_managed_arena_resumes_pending_points_before_inspecting_ownership(self):
        checker = object.__new__(ManagedCheckArena)
        checker.resume_hero_points = Mock()
        with patch.object(CheckArena, "prepare_map") as prepare:
            checker.prepare_map()
        prepare.assert_called_once()
        checker.resume_hero_points.assert_called_once()

    def test_prepare_claim_does_not_reopen_guardian_after_failed_hero(self):
        checker = self.checker()
        checker.close_guardian = Mock()
        checker.ensure_hero_prerequisite = Mock(return_value=False)
        checker.inspect_dungeon = Mock()
        self.assertEqual(checker.prepare_claim("COCO号"), "hero_required")
        checker.close_guardian.assert_called_once()
        checker.inspect_dungeon.assert_not_called()

    def test_inspection_switches_retained_hero_panel_back_to_normal(self):
        checker = self.checker()
        checker.page = Frame("map")
        inputs = []
        checker.vision = types.SimpleNamespace(dungeon=lambda frame, name: HeroHarness.control(name))

        def find(name, frame=None):
            state = (frame or checker.page).state
            return HeroHarness.control(name) if (name == "panel_close" and state in {"hero", "normal"}
                                                  or name == "hero_dragon" and state == "hero") else None

        def arena_find(name, frame=None):
            state = (frame or checker.page).state
            expected = {"map_anchor": "map", "dungeon_panel": "normal", "arena_button": "normal"}
            return HeroHarness.control(name) if expected.get(name) == state else None

        def click(locate, arrived, description):
            if result := arrived(checker.page):
                return result
            match = locate(checker.page)
            self.assertIsNotNone(match)
            inputs.append(match.name)
            checker.page = Frame({"COCO号": "hero", "hero_dragon": "normal", "arena_button": "guardian"}[match.name])
            result = arrived(checker.page)
            self.assertTrue(result)
            return result

        checker.find = find
        checker.arena_find = arena_find
        checker.observe = lambda: checker.page
        checker.guardian_owner = lambda frame: "empty" if frame.state == "guardian" else None
        checker.click_until = click
        self.assertEqual(checker.inspect_dungeon("COCO号"), "empty")
        self.assertEqual(inputs, ["COCO号", "hero_dragon", "arena_button"])

    def test_final_tap_rejects_stale_proof_even_after_locating_button(self):
        checker = self.checker()
        checker._claim_proof_date = "2026-11-05"
        with patch.object(DailyHero, "tap") as tap:
            with self.assertRaises(AutomationError):
                checker.tap(HeroHarness.control("challenge_guardian"), "占擂")
        tap.assert_not_called()

    def test_all_incomplete_candidates_return_to_town_with_blocked_report(self):
        checker = self.checker()
        checker.ctx = types.SimpleNamespace(log=logging.getLogger("arena-blocked"), dry_run=False)
        checker.options = types.SimpleNamespace(arena_list=["COCO号", "恶魔城"], player_name="1111")
        checker.frame = None
        checker.prepare_map = Mock()
        checker.wait_for = Mock(return_value="empty")
        checker.inspect_dungeon = Mock(return_value="empty")
        checker.claim_empty = Mock(return_value="hero_required")
        checker.return_to_town = Mock()
        with tempfile.TemporaryDirectory() as directory:
            checker.run_dir = Path(directory)
            with self.assertRaises(AutomationError):
                checker.run()
            summary = json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(checker.report.status, "blocked")
        self.assertEqual(summary["status"], "blocked")
        self.assertTrue(checker.report.returned_to_town)
        self.assertEqual([entry["dungeon"] for entry in checker.report.checked], ["COCO号", "恶魔城"])
        checker.return_to_town.assert_called_once()


class ArenaHeroCaptureReplayTests(unittest.TestCase):
    def test_real_images_verify_clear_then_reopen_guardian_and_claim_with_dropped_inputs(self):
        # The supplied completed hero panel is from another region; its common
        # controls/stamps exercise the production matcher after an independently
        # checked exact Demon City entrance, without inventing a new screenshot.
        pages = [
            Page("17", "绝望监狱"), Page("18", "arena_button"), Page("19", "guardian_close"),
            Page("17", "绝望监狱"), Page("18", "normal_dragon"),
            Page("../daily_hero_reset/15", "dungeon_close"),
            Page("17", "绝望监狱"), Page("18", "arena_button"),
            Page("19", "challenge_guardian"), Page("20", "guardian_close"), Page("17_prison_self"),
        ]
        for size in ((1374, 772), (849, 477)):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                ctx = ReplayContext(pages, size=size, arena_list=["绝望监狱"])
                ctx.dropped = {4: 1, 8: 1}
                with patch("pet_alliance.daily_hero.time.monotonic", ctx.clock.monotonic), \
                        patch.dict(BOUNDS, normal_dragon=(310, 122, 462, 258)):
                    checker = CheckArena(ctx)
                    checker.run_dir.rmdir()
                    checker.run_dir = Path(directory) / "check"
                    checker.run_dir.mkdir()
                    checker.now = lambda: NOW + timedelta(seconds=ctx.clock.elapsed)
                    checker.world = Mock()
                    checker.enter_region = Mock()
                    checker.return_to_town = Mock()

                    def helper_factory(context, **kwargs):
                        helper = ArenaHeroPrerequisite(context, **kwargs)
                        helper.run_dir.rmdir()
                        helper.run_dir = Path(directory) / "hero"
                        helper.run_dir.mkdir()
                        return helper

                    with patch("pet_alliance.check_arena.ArenaHeroPrerequisite", side_effect=helper_factory):
                        report = checker.run()
                self.assertTrue(report.claimed)
                self.assertTrue(report.returned_to_town)
                self.assertTrue(report.hero_checks[0]["cleared"])
                self.assertEqual(report.hero_checks[0]["verified_date"], "2026-11-06")
                controls = [control for _, control, _, _ in ctx.inputs]
                self.assertEqual(controls.count("绝望监狱"), 3)
                self.assertEqual(controls.count("normal_dragon"), 2)
                self.assertEqual(controls.count("challenge_guardian"), 2)
                self.assertFalse(any("reset" in control for control in controls))
                self.assertEqual(report.hero_progress["completed"], [])


if __name__ == "__main__":
    unittest.main()
