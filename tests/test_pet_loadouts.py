from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from pet_alliance.adb import AutomationError, Screenshot
from pet_alliance.cooperation import TaskInterrupted
from pet_alliance.daily_hero import HeroReport
from pet_alliance.pet_loadouts import PetLoadoutSwitcher, run_with_loadout


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "pet_loadouts"
# Hit areas are independently observed in the user's 1920 x 1080 captures,
# rather than derived from the profile or the matches under test.
BOUNDS = {
    "town_pet": (660, 12, 753, 111),
    "刷图": (315, 676, 407, 736),
    "屠龙": (742, 676, 842, 736),
    "PVP": (1168, 676, 1282, 736),
    "apply": (1381, 867, 1606, 965),
    "confirm": (700, 719, 910, 809),
    "cancel": (1007, 719, 1210, 809),
    "back": (1578, 529, 1691, 644),
    "close": (1790, 2, 1888, 109),
}


@dataclass
class Page:
    fixture: str
    control: str | None = None
    after: tuple[float, int] | None = None
    shift: tuple[int, int] = (0, 0)


class ReplayContext:
    """Read real frames, reject stale taps, and never communicate with ADB."""

    def __init__(self, directory, pages, *, size=(1920, 1080), dry_run=False):
        self.project_root = Path(directory)
        self.settings = {"daily_hero": {
            "poll_interval": .1, "min_click_interval": .1,
            "click_retry_interval": .4, "max_click_retries": 2,
            "transition_timeout": 1.8, "max_runtime": 4,
        }}
        self.log = logging.getLogger("pet-loadout-replay")
        self.dry_run = dry_run
        self.pages = pages
        self.size = size
        self.index = 0
        self.elapsed = 0.0
        self.entered = 0.0
        self.inputs = []
        self.captures = []
        self.dropped = {}
        self.pause_after_inputs = None
        self.screenshots = {}

    def monotonic(self):
        return self.elapsed

    def check_pause(self):
        if self.pause_after_inputs is not None and len(self.inputs) >= self.pause_after_inputs:
            raise TaskInterrupted("paused", "用户暂停测试任务。")

    def wait(self, seconds):
        self.check_pause()
        self.elapsed += seconds

    def enter(self, index):
        self.index = index
        self.entered = self.elapsed

    def capture(self):
        self.check_pause()
        page = self.pages[self.index]
        while page.after and self.elapsed - self.entered >= page.after[0]:
            self.enter(page.after[1])
            page = self.pages[self.index]
        self.captures.append((self.elapsed, self.index))
        if self.index not in self.screenshots:
            if page.fixture == "unknown":
                image = np.zeros((1080, 1920, 3), np.uint8)
            else:
                image = cv2.imdecode(np.fromfile(str(FIXTURES / page.fixture), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise AssertionError(f"Missing fixture: {page.fixture}")
            if page.shift != (0, 0):
                matrix = np.float32([[1, 0, page.shift[0]], [0, 1, page.shift[1]]])
                image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
            image = cv2.resize(image, self.size)
            self.screenshots[self.index] = Screenshot(cv2.imencode(".png", image)[1].tobytes(), *self.size)
        return self.screenshots[self.index]

    def tap(self, x, y):
        self.check_pause()
        if self.dry_run:
            raise AssertionError("Preview must not tap")
        page = self.pages[self.index]
        if page.control is None:
            raise AssertionError(f"Unexpected tap on {page.fixture}")
        left, top, right, bottom = BOUNDS[page.control]
        rx, ry = x * 1920 / self.size[0], y * 1080 / self.size[1]
        sx, sy = page.shift
        if not (left + sx <= rx <= right + sx and top + sy <= ry <= bottom + sy):
            raise AssertionError(f"Unexpected/stale tap {(rx, ry)} for {page.control}")
        self.inputs.append((self.elapsed, self.index, page.control, (rx, ry)))
        if self.dropped.get(self.index, 0):
            self.dropped[self.index] -= 1
        else:
            self.enter(self.index + 1)

    def restart_app(self, package):
        raise AssertionError("Pet loadout replay must not restart the game")


class PetLoadoutTests(unittest.TestCase):
    @contextmanager
    def replay(self, pages, group="刷图", **options):
        with tempfile.TemporaryDirectory() as directory:
            for assets in ("daily_hero", "pet_loadouts"):
                shutil.copytree(ROOT / "assets" / assets, Path(directory) / "assets" / assets)
            ctx = ReplayContext(directory, pages, **options)
            with patch("pet_alliance.daily_hero.time.monotonic", ctx.monotonic):
                yield PetLoadoutSwitcher(ctx, group), ctx

    @staticmethod
    def normal_pages(group="刷图", *, via_preview=False):
        # Only the farm preview and generic confirmation were supplied. For
        # dragon/PVP this state factory verifies selected menu coordinates and
        # common transition order, not those groups' modal title recognition.
        pages = [Page("town.png", "town_pet"), Page("pets.png", group),
                 Page("preset_farm.png", "apply"), Page("confirm.png", "confirm")]
        if via_preview:
            pages.append(Page("preset_farm.png", "back"))
        return pages + [Page("pets.png", "close"), Page("town.png")]

    def summary(self, switcher):
        return json.loads((switcher.run_dir / "summary.json").read_text(encoding="utf-8"))

    def controls(self, ctx):
        return [item[2] for item in ctx.inputs]

    def test_farm_switch_uses_real_four_step_flow_and_returns_town(self):
        for size in ((1920, 1080), (960, 540), (1374, 772)):
            with self.subTest(size=size), self.replay(self.normal_pages(), size=size) as (switcher, ctx):
                result = switcher.run()
                self.assertEqual(self.controls(ctx), ["town_pet", "刷图", "apply", "confirm", "close"])
                self.assertEqual(result.status, "completed")
                self.assertEqual(ctx.active_pet_loadout, "刷图")
                self.assertEqual(ctx.pet_loadout_last_click_at, ctx.inputs[-1][0])
                self.assertEqual(self.summary(switcher)["group"], "刷图")
                self.assertTrue((switcher.run_dir / "last_frame.png").is_file())

    def test_fresh_town_badge_from_failed_run_allows_named_groups_and_return(self):
        # This is the untouched native failure frame from the November 8 run.
        # Only the farm detail page has been supplied; dragon/PVP variants
        # verify the selected group and the common confirmation flow.
        town = "town_daily_badge_20261108.png"
        for size in ((1920, 1080), (1280, 720)):
            for group in ("刷图", "屠龙", "PVP"):
                pages = [Page(town, "town_pet"), *self.normal_pages(group)[1:-1], Page(town)]
                with self.subTest(size=size, group=group), self.replay(
                        pages, group, size=size) as (switcher, ctx):
                    # Keep production navigation and all pet page guards real.
                    # Large virtual polls bound a no-tap timeout's test time.
                    for assets in ("points_arena", "check_arena", "slay_dragon", "scheduler"):
                        shutil.copytree(ROOT / "assets" / assets,
                                        ctx.project_root / "assets" / assets)
                    switcher.options = replace(switcher.options, poll_interval=20,
                                               transition_timeout=30, max_runtime=400)
                    result = switcher.run()
                    self.assertEqual(self.controls(ctx), ["town_pet", group, "apply", "confirm", "close"])
                    self.assertAlmostEqual(ctx.inputs[0][3][0], 706, delta=4)
                    self.assertAlmostEqual(ctx.inputs[0][3][1], 63, delta=4)
                    self.assertEqual(result.status, "completed")
                    self.assertEqual(ctx.active_pet_loadout, group)
                    self.assertTrue(switcher.pet_find("town_page", switcher.frame))
                    self.assertEqual(self.summary(switcher)["status"], "completed")

    def test_confirmation_returning_to_preview_only_goes_back_then_closes(self):
        with self.replay(self.normal_pages(via_preview=True)) as (switcher, ctx):
            switcher.run()
            self.assertEqual(self.controls(ctx), ["town_pet", "刷图", "apply", "confirm", "back", "close"])
            self.assertEqual(self.controls(ctx).count("apply"), 1)

    def test_groups_click_their_own_named_menu_centres(self):
        selected_x = []
        for group in ("刷图", "屠龙", "PVP"):
            with self.subTest(group=group), self.replay(self.normal_pages(group), group) as (switcher, ctx):
                switcher.run()
                self.assertEqual(self.controls(ctx), ["town_pet", group, "apply", "confirm", "close"])
                self.assertEqual(ctx.active_pet_loadout, group)
                selected_x.append(ctx.inputs[1][3][0])
        self.assertLess(selected_x[0], selected_x[1])
        self.assertLess(selected_x[1], selected_x[2])

    def test_already_pet_page_does_not_navigate_back_to_town_first(self):
        with self.replay(self.normal_pages()[1:]) as (switcher, ctx):
            switcher.run()
            self.assertEqual(self.controls(ctx), ["刷图", "apply", "confirm", "close"])

    def test_startup_confirm_is_cancelled_then_requested_group_is_selected(self):
        pages = [Page("confirm.png", "cancel"), Page("preset_farm.png", "back"),
                 *self.normal_pages("PVP")[1:]]
        with self.replay(pages, "PVP") as (switcher, ctx):
            switcher.run()
            self.assertEqual(self.controls(ctx), ["cancel", "back", "PVP", "apply", "confirm", "close"])

    def test_startup_preset_returns_to_group_menu_before_applying(self):
        pages = [Page("preset_farm.png", "back"), *self.normal_pages("屠龙")[1:]]
        with self.replay(pages, "屠龙") as (switcher, ctx):
            switcher.run()
            self.assertEqual(self.controls(ctx), ["back", "屠龙", "apply", "confirm", "close"])

    def test_each_missed_click_is_relocated_and_retried(self):
        for missed_index in range(5):
            with self.subTest(missed_index=missed_index), self.replay(self.normal_pages()) as (switcher, ctx):
                ctx.dropped[missed_index] = 1
                switcher.run()
                self.assertEqual(len(ctx.inputs), 6)
                repeats = [entry for entry in ctx.inputs if entry[1] == missed_index]
                self.assertEqual(len(repeats), 2)
                self.assertGreaterEqual(repeats[1][0] - repeats[0][0], .4 - 1e-6)

    def test_retry_follows_visible_shifted_control_instead_of_stale_point(self):
        pages = [Page("town.png", "town_pet"),
                 Page("pets.png", "刷图", after=(.3, 2)),
                 Page("pets.png", "刷图", shift=(8, 0)),
                 *self.normal_pages()[2:]]
        with self.replay(pages) as (switcher, ctx):
            ctx.dropped[1] = 1
            switcher.run()
            first, second = [entry for entry in ctx.inputs if entry[2] == "刷图"]
            self.assertAlmostEqual(second[3][0] - first[3][0], 8, delta=1)

    def test_unknown_loading_page_is_observed_without_stale_clicks(self):
        pages = [Page("town.png", "town_pet"), Page("unknown", after=(.7, 2)),
                 *self.normal_pages()[1:]]
        with self.replay(pages) as (switcher, ctx):
            switcher.run()
            self.assertNotIn(1, [entry[1] for entry in ctx.inputs])
            self.assertEqual(self.controls(ctx), ["town_pet", "刷图", "apply", "confirm", "close"])

    def test_wrong_page_after_entry_times_out_and_preserves_failure_png_and_summary(self):
        with self.replay([Page("town.png", "town_pet"), Page("unknown")]) as (switcher, ctx):
            with self.assertRaises(AutomationError):
                switcher.run()
            self.assertEqual(self.controls(ctx), ["town_pet"])
            self.assertIsNone(ctx.active_pet_loadout)
            self.assertEqual(self.summary(switcher)["status"], "stopped")
            self.assertTrue(self.summary(switcher)["reason"])
            png = switcher.run_dir / "last_frame.png"
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_unexpected_dark_modal_does_not_click_underlying_group_or_close(self):
        # The preset darkens the entire pet page. Once a group is being sought,
        # an unrelated modal must never be treated as that background pet page.
        pages = [Page("pets.png", "刷图", after=(.05, 1)), Page("preset_farm.png")]
        with self.replay(pages) as (switcher, ctx):
            ctx.elapsed = .1
            # Startup intentionally handles a known preset; exercise the group
            # selection transition itself to isolate the dark-background gate.
            with self.assertRaises(AutomationError):
                switcher.click_until(
                    lambda f: switcher.pet_find("group_farm", f) if switcher.pets_ready(f) else None,
                    lambda f: False, "测试暗底不可操作", timeout=.8)
            self.assertEqual(ctx.inputs, [])

    def test_pause_at_every_boundary_sends_no_further_input_and_records_paused(self):
        for count in range(6):
            with self.subTest(count=count), self.replay(self.normal_pages()) as (switcher, ctx):
                ctx.pause_after_inputs = count
                with self.assertRaises(TaskInterrupted):
                    switcher.run()
                self.assertEqual(len(ctx.inputs), count)
                self.assertEqual(self.summary(switcher)["status"], "paused")
                self.assertIsNone(ctx.active_pet_loadout)

    def test_dry_run_neither_observes_nor_navigates_or_inputs(self):
        with self.replay([Page("unknown")], dry_run=True) as (switcher, ctx):
            with patch("pet_alliance.scheduler_navigation.SchedulerNavigator") as navigator:
                result = switcher.run()
            self.assertEqual(result.status, "preview")
            self.assertEqual(ctx.inputs, [])
            self.assertEqual(ctx.captures, [])
            navigator.assert_not_called()
            self.assertEqual(self.summary(switcher)["status"], "preview")

    def test_unknown_group_fails_before_observation_or_input(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = ReplayContext(directory, [Page("unknown")])
            with self.assertRaisesRegex(AutomationError, "没有注册的宠物组"):
                PetLoadoutSwitcher(ctx, "其他")
            self.assertEqual(ctx.inputs, [])
            self.assertEqual(ctx.captures, [])

    def test_run_with_loadout_aborts_job_if_selection_fails(self):
        job = SimpleNamespace(ctx=SimpleNamespace(dry_run=False), run=Mock())
        with patch("pet_alliance.pet_loadouts.switch_pet_loadout", side_effect=AutomationError("换组失败")):
            with self.assertRaisesRegex(AutomationError, "换组失败"):
                run_with_loadout(job, "刷图")
        job.run.assert_not_called()

    def test_run_with_loadout_passes_last_click_time_before_job_run(self):
        ctx = SimpleNamespace(dry_run=False)
        job = SimpleNamespace(ctx=ctx, last_click_at=None)
        job.run = Mock(side_effect=lambda: job.last_click_at)

        def apply(context, group):
            self.assertIs(context, ctx)
            self.assertEqual(group, "PVP")
            context.pet_loadout_last_click_at = 123.5

        with patch("pet_alliance.pet_loadouts.switch_pet_loadout", side_effect=apply):
            self.assertEqual(run_with_loadout(job, "PVP"), 123.5)
        job.run.assert_called_once_with()

    def test_run_with_loadout_preview_calls_only_job_preview(self):
        job = SimpleNamespace(ctx=SimpleNamespace(dry_run=True), last_click_at=7, run=Mock(return_value="preview"))
        with patch("pet_alliance.pet_loadouts.switch_pet_loadout") as switch:
            self.assertEqual(run_with_loadout(job, "屠龙"), "preview")
        switch.assert_not_called()
        self.assertEqual(job.last_click_at, 7)

    def test_deferred_loadout_preparation_is_left_to_job_run(self):
        job = SimpleNamespace(ctx=SimpleNamespace(dry_run=False),
                              defer_loadout_preparation=True, last_click_at=7,
                              run=Mock(return_value="deferred"))
        with patch("pet_alliance.pet_loadouts.switch_pet_loadout") as switch:
            self.assertEqual(run_with_loadout(job, "屠龙"), "deferred")
        switch.assert_not_called()
        job.run.assert_called_once_with()
        self.assertEqual(job.last_click_at, 7)

    def test_preparation_failure_persists_main_job_summary_and_propagates(self):
        cases = ((AutomationError("识别换组失败"), "stopped"),
                 (TaskInterrupted("paused", "用户暂停"), "paused"),
                 (KeyboardInterrupt("用户中断"), "interrupted"))
        for error, expected_status in cases:
            with self.subTest(status=expected_status), tempfile.TemporaryDirectory() as directory:
                job = SimpleNamespace(ctx=SimpleNamespace(dry_run=False),
                                      report=HeroReport(completed=[{"name": "已完成副本"}]),
                                      run_dir=Path(directory), run=Mock())
                with patch("pet_alliance.pet_loadouts.switch_pet_loadout", side_effect=error):
                    with self.assertRaises(type(error)) as raised:
                        run_with_loadout(job, "刷图")
                self.assertIs(raised.exception, error)
                job.run.assert_not_called()
                summary = json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["status"], expected_status)
                self.assertIn("准备宠物组“刷图”", summary["reason"])
                self.assertIn(str(error), summary["reason"])
                self.assertEqual(summary["completed"], [{"name": "已完成副本"}])


if __name__ == "__main__":
    unittest.main()
