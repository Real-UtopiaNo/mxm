from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from pet_alliance.dragon_schedule import BEIJING
from pet_alliance.guardian_time import (
    GUARD_TIME_ROI,
    GuardianTimeReader,
    guard_expiration,
    parse_guard_duration,
)
from pet_alliance.vision import VisualFrame


ROOT = Path(__file__).resolve().parents[1]


class GuardDurationTests(unittest.TestCase):
    def test_complete_chinese_durations(self):
        examples = {
            "19分58秒": 1198,
            "5小时59分59秒": 21599,
            "1时2分3秒": 3723,
            "6小时": 21600,
            "7小时1秒": 25201,
            "126分8秒": 7568,
            "59秒": 59,
            "1秒": 1,
            "0分0秒": 0,
            "1小时0分钟0秒钟": 3600,
            " １９ 分 ５８ 秒 ": 1198,
            "守护时间：19分58秒": 1198,
            "05:59:59": 21599,
        }
        for text, seconds in examples.items():
            with self.subTest(text=text):
                self.assertEqual(parse_guard_duration(text), seconds)

    def test_ambiguous_or_malformed_text_does_not_guess(self):
        for text in ("", "-", "1111", "1958", "19分58", "19:58", "1分61秒",
                     "5小时60分", "19分58秒99", "1分2分", "6小时19分garbage",
                     "1小时-1分", "19.5分", "00:60:00", "00:01:60", "3秒守护时间", None):
            with self.subTest(text=text):
                self.assertIsNone(parse_guard_duration(text))

    def test_expiration_keeps_seconds_and_crosses_midnight(self):
        now = datetime(2026, 11, 5, 23, 10, 9, tzinfo=BEIJING)
        self.assertEqual(guard_expiration(now, 1198),
                         datetime(2026, 11, 6, 4, 50, 11, tzinfo=BEIJING))

    def test_exact_six_hours_or_longer_has_already_expired(self):
        now = datetime(2026, 11, 5, 23, 10, 9, tzinfo=BEIJING)
        for elapsed in (21600, 21601, 30000):
            with self.subTest(elapsed=elapsed):
                self.assertEqual(guard_expiration(now, elapsed),
                                 now - timedelta(seconds=elapsed - 21600))

    def test_zero_elapsed_gives_six_hours(self):
        now = datetime(2026, 12, 31, 22, 1, 2, tzinfo=BEIJING)
        self.assertEqual(guard_expiration(now, 0), now + timedelta(hours=6))

    def test_input_time_is_always_interpreted_in_beijing(self):
        now = datetime(2026, 11, 5, 23, 10, 9)
        self.assertEqual(guard_expiration(now, 1198).tzinfo, BEIJING)
        utc = now.replace(tzinfo=BEIJING).astimezone(timezone.utc)
        self.assertEqual(guard_expiration(utc, 1198), guard_expiration(now, 1198))

    def test_invalid_elapsed_is_rejected(self):
        for elapsed in (-1, True, 12.5, None):
            with self.subTest(elapsed=elapsed):
                with self.assertRaises(ValueError):
                    guard_expiration(datetime(2026, 11, 5), elapsed)


class GuardianTimeReaderTests(unittest.TestCase):
    def setUp(self):
        self.reader = GuardianTimeReader()
        self.frame = VisualFrame(np.zeros((1080, 1920, 3), dtype=np.uint8))

    def result(self, text):
        return subprocess.CompletedProcess([], 0, json.dumps({"text": text}).encode(), b"")

    def test_ocr_only_receives_guardian_time_crop(self):
        self.frame.image[:] = (1, 2, 3)
        left, top, right, bottom = GUARD_TIME_ROI
        self.frame.image[top:bottom, left:right] = (30, 60, 90)

        def run(command, **kwargs):
            filename = Path(command[command.index("-ImagePath") + 1])
            patch_image = cv2.imread(str(filename))
            self.assertEqual(patch_image.shape, ((bottom - top) * 3, (right - left) * 3, 3))
            np.testing.assert_array_equal(patch_image[0, 0], [30, 60, 90])
            self.assertEqual(kwargs["timeout"], 15)
            self.assertIn("WindowsPowerShell", command[0])
            return self.result("19分58秒")

        with patch("pet_alliance.guardian_time.subprocess.run", side_effect=run):
            self.assertEqual(self.reader.read(self.frame), 1198)
        self.assertEqual(self.reader.last_text, "19分58秒")
        self.assertEqual(self.reader.last_error, "")

    def test_unknown_text_does_not_produce_an_expiration(self):
        with patch("pet_alliance.guardian_time.subprocess.run", return_value=self.result("-")):
            self.assertIsNone(self.reader.read(self.frame))
        self.assertIn("无法解析", self.reader.last_error)

    def test_incomplete_read_retries_contrast_without_guessing_missing_zero(self):
        with patch("pet_alliance.guardian_time.subprocess.run", side_effect=[
            self.result("分9秒"), self.result("0分9秒"),
        ]) as run:
            self.assertEqual(self.reader.read(self.frame), 9)
        self.assertEqual(run.call_count, 2)

    def test_ocr_failure_is_not_retried_as_a_duration(self):
        with patch("pet_alliance.guardian_time.subprocess.run",
                   side_effect=FileNotFoundError("missing executable")) as run:
            self.assertIsNone(self.reader.read(self.frame))
        self.assertEqual(run.call_count, 1)

    def test_missing_chinese_engine_is_a_read_failure(self):
        result = subprocess.CompletedProcess([], 1, b"", b"Windows OCR language unavailable")
        with patch("pet_alliance.guardian_time.subprocess.run", return_value=result):
            self.assertIsNone(self.reader.read(self.frame))
        self.assertIn("unavailable", self.reader.last_error)

    def test_process_failure_and_invalid_response_return_none(self):
        for result in (self.result(None), subprocess.CompletedProcess([], 0, b"invalid json", b""),
                       subprocess.CompletedProcess([], 0, b"[]", b"")):
            with self.subTest(result=result):
                with patch("pet_alliance.guardian_time.subprocess.run", return_value=result):
                    self.assertIsNone(self.reader.read(self.frame))

    def test_timeout_and_missing_executable_are_bounded_read_failures(self):
        for error in (subprocess.TimeoutExpired([], 15), FileNotFoundError("missing executable")):
            with self.subTest(error=error):
                with patch("pet_alliance.guardian_time.subprocess.run", side_effect=error):
                    self.assertIsNone(self.reader.read(self.frame))
                self.assertTrue(self.reader.last_error)

    def test_shared_crop_reader_supports_exact_player_text(self):
        with patch("pet_alliance.guardian_time.subprocess.run", return_value=self.result("1111")):
            self.assertEqual(self.reader.read_text(self.frame, (765, 310, 1025, 375)), "1111")

    def test_invalid_crop_does_not_start_ocr(self):
        with patch("pet_alliance.guardian_time.subprocess.run") as run:
            self.assertIsNone(self.reader.read_text(self.frame, (2000, 3000, 2200, 3100)))
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
