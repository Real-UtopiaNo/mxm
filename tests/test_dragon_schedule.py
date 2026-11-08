from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
import unittest
from unittest.mock import patch

from pet_alliance.adb import AutomationError
from pet_alliance.dragon_schedule import (
    BEIJING,
    DEFAULT_START_TIMES,
    beijing_now,
    is_dragon_active,
    next_dragon_start,
    parse_start_times,
)


class DragonScheduleTests(unittest.TestCase):
    def test_schedule_matches_supplied_timetable(self):
        self.assertEqual(DEFAULT_START_TIMES, (
            "09:00", "10:00", "12:00", "13:00", "14:00",
            "15:00", "18:00", "19:00", "20:00", "21:00",
        ))

    def test_active_windows_match_supplied_timetable(self):
        for hour in range(24):
            with self.subTest(hour=hour):
                expected = 9 <= hour < 11 or 12 <= hour < 16 or 18 <= hour < 22
                self.assertEqual(is_dragon_active(datetime(2026, 11, 5, hour, 30)), expected)

    def test_active_start_boundaries_to_microsecond_precision(self):
        continuous_starts = {"10:00", "13:00", "14:00", "15:00", "19:00", "20:00", "21:00"}
        for start in DEFAULT_START_TIMES:
            scheduled = datetime.fromisoformat(f"2026-11-05T{start}").replace(tzinfo=BEIJING)
            with self.subTest(start=start, offset=-1):
                self.assertEqual(is_dragon_active(scheduled - timedelta(microseconds=1)),
                                 start in continuous_starts)
            with self.subTest(start=start, offset=0):
                self.assertTrue(is_dragon_active(scheduled))
            with self.subTest(start=start, offset=1):
                self.assertTrue(is_dragon_active(scheduled + timedelta(microseconds=1)))

    def test_active_end_boundaries_to_microsecond_precision(self):
        final_ends = {11, 16, 22}
        for start in DEFAULT_START_TIMES:
            end = datetime.fromisoformat(f"2026-11-05T{start}").replace(tzinfo=BEIJING) + timedelta(hours=1)
            with self.subTest(end=end, offset=-1):
                self.assertTrue(is_dragon_active(end - timedelta(microseconds=1)))
            with self.subTest(end=end, offset=0):
                self.assertEqual(is_dragon_active(end), end.hour not in final_ends)
            with self.subTest(end=end, offset=1):
                self.assertEqual(is_dragon_active(end + timedelta(microseconds=1)),
                                 end.hour not in final_ends)

    def test_active_aware_input_converts_to_beijing(self):
        for utc_hour, active in ((0, False), (1, True), (2, True), (3, False),
                                 (4, True), (8, False), (10, True), (14, False)):
            with self.subTest(utc_hour=utc_hour):
                self.assertEqual(is_dragon_active(datetime(2026, 11, 5, utc_hour,
                                                           tzinfo=timezone.utc)), active)

    def test_active_naive_input_means_beijing(self):
        self.assertTrue(is_dragon_active(datetime(2026, 11, 5, 9, 15)))
        self.assertFalse(is_dragon_active(datetime(2026, 11, 5, 11, 15)))

    def test_custom_active_session_crosses_midnight(self):
        settings = ["23:30"]
        for now, active in ((datetime(2026, 11, 5, 23, 29, 59, 999999), False),
                            (datetime(2026, 11, 5, 23, 30), True),
                            (datetime(2026, 11, 6), True),
                            (datetime(2026, 11, 6, 0, 29, 59, 999999), True),
                            (datetime(2026, 11, 6, 0, 30), False),
                            (datetime(2026, 11, 6, 0, 30, 0, 1), False),
                            (datetime(2027, 1, 1), True)):
            with self.subTest(now=now):
                self.assertEqual(is_dragon_active(now, settings), active)
        self.assertTrue(is_dragon_active(datetime(2026, 11, 5, 16, 15, tzinfo=timezone.utc), settings))
        self.assertEqual(settings, ["23:30"])

    def test_custom_active_sessions_use_sorted_unique_configured_starts(self):
        settings = ["18:30", "09:15", "18:30"]
        for now, active in ((datetime(2026, 11, 5, 9, 14, 59, 999999), False),
                            (datetime(2026, 11, 5, 9, 15), True),
                            (datetime(2026, 11, 5, 10, 14, 59, 999999), True),
                            (datetime(2026, 11, 5, 10, 15), False),
                            (datetime(2026, 11, 5, 18, 30), True),
                            (datetime(2026, 11, 5, 19, 30), False)):
            with self.subTest(now=now):
                self.assertEqual(is_dragon_active(now, settings), active)

    def test_active_check_validates_configuration(self):
        with self.assertRaises(AutomationError):
            is_dragon_active(datetime(2026, 11, 5), [])

    def test_missing_session_gaps_choose_next_start(self):
        for current, expected in (("08:00", "09:00"), ("11:00", "12:00"),
                                  ("16:00", "18:00"), ("17:59", "18:00")):
            with self.subTest(current=current):
                result = next_dragon_start(datetime.fromisoformat(f"2026-11-05T{current}"))
                self.assertEqual(result, datetime.fromisoformat(f"2026-11-05T{expected}").replace(tzinfo=BEIJING))

    def test_active_hour_still_waits_for_next_start(self):
        self.assertEqual(next_dragon_start(datetime(2026, 11, 5, 9, 30)),
                         datetime(2026, 11, 5, 10, tzinfo=BEIJING))

    def test_every_exact_start_is_eligible_immediately(self):
        for start in DEFAULT_START_TIMES:
            with self.subTest(start=start):
                now = datetime.fromisoformat(f"2026-11-05T{start}").replace(tzinfo=BEIJING)
                self.assertEqual(next_dragon_start(now), now)

    def test_one_microsecond_before_each_start_waits_for_that_start(self):
        for start in DEFAULT_START_TIMES:
            with self.subTest(start=start):
                scheduled = datetime.fromisoformat(f"2026-11-05T{start}").replace(tzinfo=BEIJING)
                self.assertEqual(next_dragon_start(scheduled - timedelta(microseconds=1)), scheduled)

    def test_one_microsecond_after_each_start_chooses_following_start(self):
        for index, start in enumerate(DEFAULT_START_TIMES):
            with self.subTest(start=start):
                scheduled = datetime.fromisoformat(f"2026-11-05T{start}").replace(tzinfo=BEIJING)
                if index + 1 < len(DEFAULT_START_TIMES):
                    expected = datetime.fromisoformat(f"2026-11-05T{DEFAULT_START_TIMES[index + 1]}").replace(tzinfo=BEIJING)
                else:
                    expected = datetime(2026, 11, 6, 9, tzinfo=BEIJING)
                self.assertEqual(next_dragon_start(scheduled + timedelta(microseconds=1)), expected)

    def test_after_last_session_rolls_to_next_day(self):
        for now in (datetime(2026, 11, 5, 21, 30), datetime(2026, 11, 5, 22),
                    datetime(2026, 11, 5, 23, 59, 59, 999999)):
            with self.subTest(now=now):
                self.assertEqual(next_dragon_start(now), datetime(2026, 11, 6, 9, tzinfo=BEIJING))

    def test_year_rollover(self):
        self.assertEqual(next_dragon_start(datetime(2026, 12, 31, 23)),
                         datetime(2027, 1, 1, 9, tzinfo=BEIJING))

    def test_utc_input_uses_beijing_calendar_day(self):
        self.assertEqual(next_dragon_start(datetime(2026, 11, 5, 17, tzinfo=timezone.utc)),
                         datetime(2026, 11, 6, 9, tzinfo=BEIJING))

    def test_utc_exact_start_converts_to_beijing(self):
        self.assertEqual(next_dragon_start(datetime(2026, 11, 5, 1, tzinfo=timezone.utc)),
                         datetime(2026, 11, 5, 9, tzinfo=BEIJING))

    def test_naive_input_means_beijing(self):
        self.assertEqual(next_dragon_start(datetime(2026, 11, 5, 9, 15)),
                         datetime(2026, 11, 5, 10, tzinfo=BEIJING))

    def test_custom_start_times_are_sorted_and_deduplicated(self):
        settings = ["18:30", "09:15", "18:30"]
        self.assertEqual(parse_start_times(settings), (time(9, 15), time(18, 30)))
        self.assertEqual(next_dragon_start(datetime(2026, 11, 5, 9, 16), settings),
                         datetime(2026, 11, 5, 18, 30, tzinfo=BEIJING))
        self.assertEqual(settings, ["18:30", "09:15", "18:30"])

    def test_custom_midnight_start(self):
        self.assertEqual(next_dragon_start(datetime(2026, 11, 5, 23, 59), ["00:00"]),
                         datetime(2026, 11, 6, 0, tzinfo=BEIJING))

    def test_invalid_configuration_is_rejected(self):
        invalid = (None, "09:00", [], (), {}, {"09:00"}, [9], [None], [True],
                   ["9:00"], ["09:0"], ["24:00"], ["12:60"], ["-1:00"],
                   [" 09:00"], ["09:00 "], ["09:00:00"], ["０９:００"], ["09:00\n"],
                   ["09:00", "bad"])
        for values in invalid:
            with self.subTest(values=values), self.assertRaisesRegex(AutomationError, "start_times"):
                parse_start_times(values)

    def test_next_start_validates_configuration(self):
        with self.assertRaises(AutomationError):
            next_dragon_start(datetime(2026, 11, 5), [])

    def test_beijing_now_reads_clock_with_explicit_fixed_timezone(self):
        expected = datetime(2026, 11, 5, 12, 34, tzinfo=BEIJING)
        with patch("pet_alliance.dragon_schedule.datetime") as clock:
            clock.now.return_value = expected
            self.assertEqual(beijing_now(), expected)
            clock.now.assert_called_once_with(BEIJING)
        self.assertEqual(BEIJING.utcoffset(None), timedelta(hours=8))


if __name__ == "__main__":
    unittest.main()
