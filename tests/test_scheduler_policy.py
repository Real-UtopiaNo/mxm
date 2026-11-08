from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from pet_alliance.adb import AutomationError
from pet_alliance.dragon_schedule import BEIJING
from pet_alliance.scheduler_policy import choose_task


class SchedulerPolicyTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 6, 9, 30, tzinfo=BEIJING)
        self.pending = {"daily_hero": False, "daily_hero_reset": False}
        self.done = {"daily_hero": True, "daily_hero_reset": True}

    def choose(self, now=None, flags=None, **kwargs):
        return choose_task(now or self.now, self.pending if flags is None else flags, **kwargs)

    def test_unknown_arena_deadline_checks_before_dragon_and_hero(self):
        for deadline in (None, ""):
            for checked in (False, True):
                with self.subTest(deadline=deadline, checked=checked):
                    self.assertEqual(self.choose(arena_expired_time=deadline, arena_checked=checked), "check_arena")

    def test_arena_due_includes_exact_deadline(self):
        for difference in (0, 1, 86400):
            with self.subTest(difference=difference):
                expiry = self.now - timedelta(seconds=difference)
                self.assertEqual(self.choose(arena_expired_time=expiry.isoformat()), "check_arena")

    def test_future_arena_deadline_skips_check_even_on_first_start(self):
        self.assertEqual(self.choose(arena_expired_time=self.now + timedelta(seconds=1)), "slay_dragon")

    def test_arena_backoff_allows_dragon(self):
        self.assertEqual(self.choose(arena_retry_at=self.now + timedelta(minutes=5)), "slay_dragon")

    def test_pending_points_visit_is_recovered_even_with_future_guard_deadline(self):
        records = {"daily_hero": {"done": True, "date": "2026-11-06",
                                   "progress": {"points_arena_pending": True}}}
        self.assertEqual(self.choose(flags=records, arena_expired_time=self.now + timedelta(hours=6),
                                     arena_retry_at=self.now + timedelta(minutes=5)), "check_arena")
        records["daily_hero"]["progress"]["points_arena_pending"] = False
        self.assertEqual(self.choose(flags=records, arena_expired_time=self.now + timedelta(hours=6)), "slay_dragon")

    def test_yesterdays_pending_points_do_not_cross_midnight(self):
        records = {"daily_hero": {"done": True, "date": "2026-11-05",
                                   "progress": {"points_arena_pending": True}}}
        self.assertEqual(self.choose(flags=records, arena_expired_time=self.now + timedelta(hours=6)), "slay_dragon")

    def test_disabled_points_skip_pending_without_suppressing_normal_due_arena(self):
        records = {"daily_hero": {"done": True, "date": "2026-11-06",
                                  "progress": {"points_arena_pending": True}}}
        self.assertEqual(self.choose(flags=records, arena_expired_time=self.now + timedelta(hours=6),
                                     points_arena_enabled=False), "slay_dragon")
        self.assertEqual(self.choose(flags=records, arena_expired_time=self.now,
                                     points_arena_enabled=False), "check_arena")

    def test_arena_backoff_allows_daily_and_expires_exactly(self):
        now = self.now.replace(hour=11)
        retry = now + timedelta(minutes=5)
        self.assertEqual(self.choose(now, arena_retry_at=retry), "daily_hero")
        self.assertEqual(self.choose(retry, arena_retry_at=retry), "check_arena")

    def test_backoff_does_not_force_arena_if_known_expiry_is_later(self):
        self.assertEqual(self.choose(arena_expired_time=self.now + timedelta(hours=1),
                                     arena_retry_at=self.now - timedelta(seconds=1)), "slay_dragon")

    def test_hero_then_reset_then_idle_outside_dragon_hours(self):
        now = self.now.replace(hour=11)
        kwargs = {"arena_expired_time": now + timedelta(hours=6)}
        self.assertEqual(self.choose(now, **kwargs), "daily_hero")
        self.assertEqual(self.choose(now, {"daily_hero": True, "daily_hero_reset": False}, **kwargs), "daily_hero_reset")
        self.assertIsNone(self.choose(now, self.done, **kwargs))

    def test_all_timetable_transitions_choose_dragon_only_when_active(self):
        for hour in range(24):
            with self.subTest(hour=hour):
                now = self.now.replace(hour=hour, minute=0)
                expected = "slay_dragon" if (9 <= hour < 11 or 12 <= hour < 16 or 18 <= hour < 22) else "daily_hero"
                self.assertEqual(self.choose(now, arena_expired_time=now + timedelta(days=1)), expected)

    def test_beijing_midnight_ignores_yesterdays_daily_record(self):
        now = datetime(2026, 11, 7, tzinfo=BEIJING)
        records = {task: {"done": True, "date": "2026-11-06"} for task in self.done}
        self.assertEqual(self.choose(now, records, arena_expired_time=now + timedelta(hours=6)), "daily_hero")

    def test_today_record_is_supported_without_mutating_snapshot(self):
        now = self.now.replace(hour=11)
        records = {"daily_hero": {"done": True, "date": "2026-11-06"},
                   "daily_hero_reset": {"done": False, "date": "2026-11-06"}}
        self.assertEqual(self.choose(now, records, arena_expired_time=now + timedelta(hours=6)), "daily_hero_reset")
        self.assertFalse(records["daily_hero_reset"]["done"])

    def test_truthy_non_boolean_done_does_not_suppress_task(self):
        now = self.now.replace(hour=11)
        for value in (1, "completed", "false", None):
            with self.subTest(value=value):
                self.assertEqual(self.choose(now, {"daily_hero": value},
                                             arena_expired_time=now + timedelta(hours=1)), "daily_hero")

    def test_timezone_conversion_and_naive_beijing_system_time(self):
        for now in (self.now, self.now.replace(tzinfo=None), self.now.astimezone(timezone.utc)):
            with self.subTest(now=now):
                self.assertEqual(self.choose(now, arena_expired_time="2026-11-06T02:00:00Z"), "slay_dragon")
                self.assertEqual(self.choose(now, arena_expired_time="2026-11-06T09:30:00+08:00"), "check_arena")

    def test_custom_timetable_is_used(self):
        now = self.now.replace(hour=23, minute=45)
        self.assertEqual(self.choose(now, arena_expired_time=now + timedelta(hours=1), start_times=["23:30"]), "slay_dragon")

    def test_invalid_deadlines_fail_instead_of_spinning(self):
        for key in ("arena_expired_time", "arena_retry_at"):
            for value in ("tomorrow", 123, True):
                with self.subTest(key=key, value=value), self.assertRaises(AutomationError):
                    self.choose(**{key: value})


if __name__ == "__main__":
    unittest.main()
