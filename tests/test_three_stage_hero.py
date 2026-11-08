"""Resume the remaining hero stages after an early Exit Battle result.

``west_three_stage_complete.jpg`` is an unchanged copy of
``artifacts/runs/Daily_Hero_20261106_010424_859874/completed_12.jpg``.
Despite its legacy fixture filename, the user has clarified that this is an
intermediate third-monster Exit screen, not proof of a cleared dungeon.
The original run records the western entrance at (640, 424); its exact name
cannot be established from that screenshot alone.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch

from pet_alliance.daily_hero_reset import DailyHeroReset
from tests import test_daily_hero as replay


MIDPOINT = "west_three_stage_complete"
WEST = {"key": "west", "name": "西部大陆"}
FIRST_THREE = ["battle", "victory", "battle", "victory", "battle", MIDPOINT]
LAST_TWO = ["hero_progress", "battle", "victory", "battle", "complete", "reward", "region_after"]


class ThreeStageHeroTests(unittest.TestCase):
    # Reuse the real-image workflow factory and its ReplayContext / ReplayClock.
    hero = replay.DailyHeroTests.hero

    def fight(self, hero):
        hero.observe()
        return hero.fight_dungeon(WEST, hero.markers(hero.frame)[0])

    def events(self, hero):
        return [json.loads(line) for line in (hero.run_dir / "events.jsonl").read_text(
            encoding="utf-8").splitlines()]

    def assert_five_stage_clear(self, hero):
        self.assertEqual(hero.report.victories, 5)
        self.assertEqual(hero.report.failures, 0)
        self.assertEqual(len(hero.report.completed), 1)
        self.assertEqual(hero.report.skipped, [])
        self.assertTrue(hero.is_map(hero.frame))
        self.assertEqual(hero.ctx.states[hero.ctx.index], "region_after")
        events = self.events(hero)
        self.assertEqual(sum(event["event"] == "tap" and event["template"] == "next_stage"
                             for event in events), 3)
        self.assertEqual(sum(event["event"] == "dungeon_complete" for event in events), 1)

    def assert_intermediate_has_no_completion(self, hero):
        """Check the actual report while the intermediate panel is visible."""
        original = hero.ctx.capture
        observations = []

        def capture():
            if hero.ctx.states[hero.ctx.index] == "hero_progress":
                observations.append(hero.report.victories)
                self.assertEqual(hero.report.completed, [])
                self.assertEqual(hero.report.arena_visits, 0)
                self.assertFalse(hero.report.points_arena_pending)
            return original()

        hero.ctx.capture = capture
        return observations

    def reset_hero(self, directory, states, **options):
        base = self.hero(directory, states, **options)
        hero = DailyHeroReset(base.ctx)
        hero.run_dir.rmdir()
        hero.run_dir = base.run_dir
        hero.started = hero.ctx.clock.now
        return hero

    def test_daily_hero_exits_after_third_then_rechallenges_remaining_two(self):
        states = ["region_before", "normal", "hero", *FIRST_THREE, *LAST_TWO]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, states)
            for size in ((1920, 1080), (1280, 720)):
                with self.subTest(resolution=size):
                    frame = replay.frame(MIDPOINT, size)
                    for control in ("victory", "next_disabled", "exit_battle"):
                        self.assertIsNotNone(hero.find(control, frame))
                    self.assertIsNone(hero.find("next_stage", frame))
            midpoint_observations = self.assert_intermediate_has_no_completion(hero)
            self.assertTrue(self.fight(hero))
            self.assert_five_stage_clear(hero)
            self.assertTrue(midpoint_observations)
            self.assertEqual(set(midpoint_observations), {3})
            point = hero.find("challenge", replay.frame("hero_progress")).center
            self.assertEqual([(x, y) for state, x, y in hero.ctx.inputs if state == "hero_progress"],
                             [point])

    def test_reset_continuation_only_challenges_without_reopening_reset_flow(self):
        # Start after a successful paid reset, with the challenge-ready panel
        # visible. The actual reset open/challenge and battle methods still run.
        with tempfile.TemporaryDirectory() as directory:
            hero = self.reset_hero(directory, ["hero", *FIRST_THREE, *LAST_TWO])
            midpoint_observations = self.assert_intermediate_has_no_completion(hero)
            target = replay.Match("entrance", 590, 410, 100, 28, 1)
            with patch.object(hero, "open_dungeon_panel"), \
                    patch.object(hero, "open_dungeon", wraps=hero.open_dungeon) as opening:
                self.assertTrue(hero.fight_dungeon(WEST, target))
            self.assert_five_stage_clear(hero)
            self.assertTrue(midpoint_observations)
            self.assertEqual(opening.call_count, 1)
            self.assertEqual(hero.report.resets, 0)
            self.assertEqual(hero.report.diamonds_spent, 0)

    def test_missed_intermediate_exit_and_challenge_both_relocate_before_retry(self):
        states = ["region_before", "normal", "hero", *FIRST_THREE, *LAST_TWO]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, states)
            hero.ctx.dropped_clicks = {MIDPOINT: 1, "hero_progress": 1}
            original_capture, original_tap = hero.ctx.capture, hero.ctx.tap
            captures = 0
            taps = {MIDPOINT: [], "hero_progress": []}

            def capture():
                nonlocal captures
                captures += 1
                return original_capture()

            def tap(x, y):
                state = hero.ctx.states[hero.ctx.index]
                if state in taps:
                    taps[state].append((captures, hero.ctx.clock.now, x, y))
                original_tap(x, y)

            hero.ctx.capture, hero.ctx.tap = capture, tap
            self.assertTrue(self.fight(hero))
            self.assert_five_stage_clear(hero)
            for state, template in ((MIDPOINT, "exit_battle"), ("hero_progress", "challenge")):
                with self.subTest(state=state):
                    self.assertEqual(len(taps[state]), 2)
                    self.assertGreater(taps[state][1][0], taps[state][0][0])
                    self.assertGreaterEqual(taps[state][1][1] - taps[state][0][1], .4 - 1e-9)
                    point = hero.find(template, replay.frame(state)).center
                    self.assertEqual([entry[2:] for entry in taps[state]], [point, point])

    def test_points_arena_waits_for_all_five_and_counts_one_whole_dungeon(self):
        states = ["region_before", "normal", "hero", *FIRST_THREE, *LAST_TWO]
        for completed_before, expected_visits in ((0, 0), (2, 1)):
            with self.subTest(completed_before=completed_before), tempfile.TemporaryDirectory() as directory:
                hero = self.hero(directory, states, max_dungeons=1)
                hero.ctx.resume_data = {"completed_since_arena": completed_before}
                observations = self.assert_intermediate_has_no_completion(hero)
                with patch.object(hero, "run_points_arena") as visit:
                    report = hero.run()
                self.assertEqual(report.status, "limited")
                self.assert_five_stage_clear(hero)
                self.assertTrue(observations)
                self.assertEqual(visit.call_count, expected_visits)

    def test_a_third_monster_defeat_remains_failure_instead_of_continuation(self):
        states = ["region_before", "normal", "hero", *FIRST_THREE[:-1], "defeat", "region_after"]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, states, max_retries=0)
            self.assertFalse(self.fight(hero))
            self.assertEqual(hero.report.victories, 2)
            self.assertEqual(hero.report.failures, 1)
            self.assertEqual(hero.report.completed, [])
            self.assertEqual(len(hero.report.skipped), 1)
            self.assertTrue(hero.is_map(hero.frame))

    def test_reset_fourth_monster_failure_keeps_continuation_without_another_reset(self):
        # Fail the fourth monster once, then return to the same unfinished
        # hero panel and retry. Grey-reset handling for ordinary initial /
        # failed entries stays covered by test_daily_hero_reset; this is the
        # special continuation which must only press Challenge.
        states = ["hero", *FIRST_THREE, "hero_progress", "battle", "defeat", *LAST_TWO]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.reset_hero(directory, states, max_retries=1)
            observations = self.assert_intermediate_has_no_completion(hero)
            target = replay.Match("entrance", 590, 410, 100, 28, 1)
            with patch.object(hero, "open_dungeon_panel"), \
                    patch.object(hero, "open_dungeon", wraps=hero.open_dungeon) as opening:
                self.assertTrue(hero.fight_dungeon(WEST, target))
            self.assertEqual(hero.report.victories, 5)
            self.assertEqual(hero.report.failures, 1)
            self.assertEqual(len(hero.report.completed), 1)
            self.assertEqual(hero.report.skipped, [])
            self.assertEqual(opening.call_count, 1)
            self.assertEqual(hero.report.resets, 0)
            self.assertEqual(hero.report.diamonds_spent, 0)
            self.assertTrue(observations)
            point = hero.find("challenge", replay.frame("hero_progress")).center
            self.assertEqual([(x, y) for state, x, y in hero.ctx.inputs if state == "hero_progress"],
                             [point, point])

    def test_genuine_three_monster_clear_with_reward_still_finishes_once(self):
        states = ["region_before", "normal", "hero", *FIRST_THREE, "reward", "region_after"]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, states)
            self.assertTrue(self.fight(hero))
            self.assertEqual(hero.report.victories, 3)
            self.assertEqual(len(hero.report.completed), 1)
            self.assertEqual(hero.report.failures, 0)
            self.assertTrue(hero.is_map(hero.frame))

    def test_third_exit_to_map_reopens_same_entrance_before_remaining_two(self):
        states = ["region_before", "normal", "hero", *FIRST_THREE,
                  "region_before", "normal", *LAST_TWO]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, states)
            target = hero.markers(replay.frame("region_before"))[2]
            self.assertEqual(target.center, (309, 581))
            observations = self.assert_intermediate_has_no_completion(hero)
            self.assertTrue(hero.fight_dungeon(WEST, target))
            self.assert_five_stage_clear(hero)
            self.assertTrue(observations)
            expected = (target.center[0] + hero.dungeon_entry_offset[0],
                        target.center[1] + hero.dungeon_entry_offset[1])
            self.assertEqual([(x, y) for state, x, y in hero.ctx.inputs if state == "region_before"],
                             [expected, expected])

    def test_map_reentry_without_badge_uses_only_the_same_fresh_green_nameplate(self):
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, ["region_after", "normal"])
            original_map = replay.frame("region_before")
            target = hero.markers(original_map)[2]
            self.assertEqual(target.center, (309, 581))
            current_map = replay.frame("region_after")
            self.assertFalse(any(hero.near(marker, target) for marker in hero.markers(current_map)))
            # A stale frame must not supply the vanished red marker's point.
            hero.frame = original_map
            hero.open_segment_panel(target)
            self.assertEqual(hero.ctx.states[hero.ctx.index], "normal")
            self.assertEqual(hero.ctx.inputs, [("region_after", 228, 600)])
            self.assertEqual(hero.report.completed, [])

    def test_partial_progress_reaches_midpoint_after_one_win_and_reopens_map_entrance(self):
        # The first two monsters were cleared before this run. Start on an
        # already-open unfinished panel, then receive only the third win.
        states = ["hero_progress", "battle", MIDPOINT, "region_before", "normal", *LAST_TWO]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, states)
            target = hero.markers(replay.frame("region_before"))[2]
            original_open = hero.open_dungeon_panel

            def open_panel(target, resume=False):
                if hero.ctx.index == 0:
                    return
                return original_open(target, resume)

            region = {"key": "__current__", "name": "当前区域"}
            with patch.object(hero, "open_dungeon_panel", side_effect=open_panel):
                self.assertTrue(hero.fight_dungeon(region, target))
            self.assertEqual(hero.report.victories, 3)
            self.assertEqual(hero.report.failures, 0)
            self.assertEqual(len(hero.report.completed), 1)
            self.assertTrue(hero.is_map(hero.frame))
            expected = (target.center[0] + hero.dungeon_entry_offset[0],
                        target.center[1] + hero.dungeon_entry_offset[1])
            self.assertEqual([(x, y) for state, x, y in hero.ctx.inputs if state == "region_before"],
                             [expected])
            events = self.events(hero)
            self.assertEqual(sum(event["event"] == "hero_segment_exit" for event in events), 1)
            self.assertFalse(any(event["event"] == "tap" and event["template"] == "world_button"
                                 for event in events))

    def test_fourth_monster_failure_to_map_without_badge_reenters_same_green_nameplate(self):
        for reset in (False, True):
            with self.subTest(reset=reset), tempfile.TemporaryDirectory() as directory:
                states = ["hero", *FIRST_THREE, "hero_progress", "battle", "defeat",
                          "region_after", "normal", *LAST_TWO]
                hero = (self.reset_hero(directory, states, max_retries=1) if reset
                        else self.hero(directory, states, max_retries=1))
                original_map = replay.frame("region_before")
                target = (next(entrance for entrance in hero.markers(original_map)
                               if entrance.center == (228, 600)) if reset
                          else hero.markers(original_map)[2])
                self.assertEqual(target.center, (228, 600) if reset else (309, 581))
                original_open = hero.open_dungeon_panel

                def open_panel(target, resume=False):
                    # Start after any reset purchase on the playable panel;
                    # every later map/panel reentry uses the production method.
                    if hero.ctx.index == 0:
                        return
                    return original_open(target, resume)

                with patch.object(hero, "open_dungeon_panel", side_effect=open_panel):
                    self.assertTrue(hero.fight_dungeon(WEST, target))
                self.assertEqual(hero.report.victories, 5)
                self.assertEqual(hero.report.failures, 1)
                self.assertEqual(len(hero.report.completed), 1)
                self.assertEqual(hero.report.skipped, [])
                self.assertTrue(hero.is_map(hero.frame))
                self.assertEqual([(x, y) for state, x, y in hero.ctx.inputs if state == "region_after"],
                                 [(228, 600)])
                if reset:
                    self.assertEqual(hero.report.resets, 0)
                    self.assertEqual(hero.report.diamonds_spent, 0)

    def test_ordinary_exit_to_unfinished_panel_closes_without_moru_continuation(self):
        for reset in (False, True):
            with self.subTest(reset=reset), tempfile.TemporaryDirectory() as directory:
                states = ["hero", "battle", "complete", "hero_progress", "region_after"]
                hero = (self.reset_hero(directory, states) if reset else self.hero(directory, states))
                original_map = replay.frame("region_before")
                target = (next(entrance for entrance in hero.markers(original_map)
                               if entrance.center == (228, 600)) if reset
                          else hero.markers(original_map)[2])
                with patch.object(hero, "open_dungeon_panel"):
                    self.assertTrue(hero.fight_dungeon(WEST, target))
                self.assertEqual(hero.report.victories, 1)
                self.assertEqual(len(hero.report.completed), 1)
                self.assertEqual(hero.report.failures, 0)
                close = hero.find("panel_close", replay.frame("hero_progress")).center
                self.assertEqual([(x, y) for state, x, y in hero.ctx.inputs if state == "hero_progress"],
                                 [close])
                self.assertFalse(any(event["event"] == "hero_segment_exit" for event in self.events(hero)))
                if reset:
                    self.assertEqual(hero.report.resets, 0)
                    self.assertEqual(hero.report.diamonds_spent, 0)

    def test_known_nonwest_region_rejects_moru_midpoint_continuation(self):
        states = ["hero", "battle", MIDPOINT, "hero_progress", "region_after"]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, states)
            target = hero.markers(replay.frame("region_before"))[2]
            with patch.object(hero, "open_dungeon_panel"):
                self.assertTrue(hero.fight_dungeon({"key": "demon", "name": "魔都"}, target))
            self.assertEqual(hero.report.victories, 1)
            self.assertEqual(len(hero.report.completed), 1)
            close = hero.find("panel_close", replay.frame("hero_progress")).center
            self.assertEqual([(x, y) for state, x, y in hero.ctx.inputs if state == "hero_progress"],
                             [close])
            self.assertFalse(any(event["event"] == "hero_segment_exit" for event in self.events(hero)))

    def test_west_ordinary_third_win_to_map_does_not_reopen_a_dungeon(self):
        states = ["region_before", "normal", "hero", *FIRST_THREE[:-1], "complete", "region_before"]
        with tempfile.TemporaryDirectory() as directory:
            hero = self.hero(directory, states)
            target = hero.markers(replay.frame("region_before"))[2]
            with patch.object(hero, "open_segment_panel", wraps=hero.open_segment_panel) as reopen:
                self.assertTrue(hero.fight_dungeon(WEST, target))
            self.assertEqual(hero.report.victories, 3)
            self.assertEqual(len(hero.report.completed), 1)
            self.assertEqual(reopen.call_count, 0)
            self.assertEqual(sum(state == "region_before" for state, _, _ in hero.ctx.inputs), 1)
            self.assertTrue(hero.is_map(hero.frame))


if __name__ == "__main__":
    unittest.main()
