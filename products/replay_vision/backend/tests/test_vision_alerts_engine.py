from datetime import UTC, datetime, timedelta
from typing import Any

from posthog.test.base import BaseTest
from unittest.mock import MagicMock, patch

from django.utils import timezone

from parameterized import parameterized

from posthog.models.utils import uuid7

from products.replay_vision.backend.models.replay_observation import (
    ObservationStatus,
    ObservationTrigger,
    ReplayObservation,
)
from products.replay_vision.backend.models.replay_scanner import ReplayScanner, ScannerModel, ScannerType
from products.replay_vision.backend.models.vision_alert import (
    VisionAlertConfiguration,
    VisionAlertEvent,
    VisionAlertKind,
    VisionAlertMetric,
    VisionAlertState,
)
from products.replay_vision.backend.temporal.vision_alerts.activities import (
    DiscoverDueAlertsInput,
    EvaluateAlertBatchInput,
    _discover_due,
    _evaluate_batch,
)
from products.replay_vision.backend.tests.helpers import snapshot_for

_ACTIVITIES = "products.replay_vision.backend.temporal.vision_alerts.activities"


class TestVisionAlertEngine(BaseTest):
    def setUp(self) -> None:
        super().setUp()
        self.scanner = ReplayScanner.objects.create(
            team=self.team,
            name="Checkout monitor",
            scanner_type=ScannerType.MONITOR,
            scanner_config={"prompt": "did the user check out?"},
            model=ScannerModel.GEMINI_3_7_FLASH,
        )

    def _make_alert(self, **overrides: Any) -> VisionAlertConfiguration:
        fields: dict[str, Any] = {
            "team": self.team,
            "scanner": self.scanner,
            "name": f"alert-{uuid7()}",
            "kind": VisionAlertKind.METRIC,
            "metric": VisionAlertMetric.COUNT,
            "threshold": 2,
            "selection": {"verdict": ["fail"]},
            "window_days": 1,
            "first_enabled_at": timezone.now(),
        }
        fields.update(overrides)
        return VisionAlertConfiguration.objects.for_team(self.team.id).create(**fields)

    def _make_observation(
        self,
        verdict: str = "fail",
        status: str = ObservationStatus.SUCCEEDED,
        completed_at: datetime | None = None,
        score: float | None = None,
    ) -> ReplayObservation:
        model_output: dict[str, Any] = {"scanner_type": self.scanner.scanner_type, "verdict": verdict}
        if score is not None:
            model_output["score"] = score
        return ReplayObservation.objects.create(
            scanner=self.scanner,
            team=self.team,
            session_id=f"s-{uuid7()}",
            status=status,
            completed_at=completed_at or timezone.now(),
            scanner_snapshot=snapshot_for(self.scanner),
            scanner_result={"model_output": model_output} if status == ObservationStatus.SUCCEEDED else {},
            triggered_by=ObservationTrigger.SCHEDULE,
        )

    def _run_batch(self, alert: VisionAlertConfiguration, *, delivered: bool = True):
        produce_result = MagicMock()
        with (
            patch(f"{_ACTIVITIES}.produce_alert_internal_event", return_value=produce_result) as produce,
            patch(f"{_ACTIVITIES}.flush_alert_internal_events"),
            patch(f"{_ACTIVITIES}.alert_internal_event_delivered", return_value=delivered),
        ):
            output = _evaluate_batch(EvaluateAlertBatchInput(alert_ids=[str(alert.id)]))
        alert.refresh_from_db()
        return output, produce

    def test_breach_fires_and_resolves(self) -> None:
        alert = self._make_alert()
        self._make_observation()
        self._make_observation()

        output, produce = self._run_batch(alert)
        assert output.alerts_fired == 1
        assert alert.state == VisionAlertState.FIRING
        assert alert.last_notified_at is not None
        assert alert.next_check_at is not None
        assert produce.call_args.kwargs["event_name"] == "$replay_vision_alert_firing"
        props = produce.call_args.kwargs["properties"]
        assert props["metric_value"] == 2.0
        assert props["scanner_name"] == "Checkout monitor"
        check_events = VisionAlertEvent.objects.filter(alert=alert, kind=VisionAlertEvent.Kind.CHECK)
        assert check_events.count() == 1
        assert check_events.get().threshold_breached is True

        # Observations age out of the window -> resolve notification on the next check.
        ReplayObservation.objects.filter(team_id=self.team.id).update(completed_at=timezone.now() - timedelta(days=2))
        alert.next_check_at = None
        alert.save(update_fields=["next_check_at"])
        output, produce = self._run_batch(alert)
        assert output.alerts_resolved == 1
        assert alert.state == VisionAlertState.NOT_FIRING
        assert produce.call_args.kwargs["event_name"] == "$replay_vision_alert_resolved"

    @parameterized.expand(
        [
            ("outside_window", {"completed_at_days_ago": 2}, VisionAlertState.NOT_FIRING),
            ("wrong_verdict", {"verdict": "pass"}, VisionAlertState.NOT_FIRING),
            ("failed_status_excluded_by_default", {"status": ObservationStatus.FAILED}, VisionAlertState.NOT_FIRING),
        ]
    )
    def test_non_matching_observations_do_not_fire(
        self, _name: str, observation_kwargs: dict[str, Any], expected_state: str
    ) -> None:
        alert = self._make_alert(threshold=1)
        kwargs = dict(observation_kwargs)
        days_ago = kwargs.pop("completed_at_days_ago", None)
        if days_ago is not None:
            kwargs["completed_at"] = timezone.now() - timedelta(days=days_ago)
        self._make_observation(**kwargs)
        _output, produce = self._run_batch(alert)
        assert alert.state == expected_state
        produce.assert_not_called()

    def test_failed_observation_count_with_status_selection(self) -> None:
        alert = self._make_alert(threshold=1, selection={"statuses": ["failed"]})
        self._make_observation(status=ObservationStatus.FAILED)
        output, _ = self._run_batch(alert)
        assert output.alerts_fired == 1
        assert alert.state == VisionAlertState.FIRING

    def test_avg_score_empty_window_is_inconclusive(self) -> None:
        alert = self._make_alert(metric=VisionAlertMetric.AVG_SCORE, threshold=3, selection={})
        _output, produce = self._run_batch(alert)
        assert alert.state == VisionAlertState.NOT_FIRING
        assert alert.consecutive_failures == 0
        produce.assert_not_called()

    def test_avg_score_below_direction_fires(self) -> None:
        alert = self._make_alert(metric=VisionAlertMetric.AVG_SCORE, threshold=3, direction="below", selection={})
        self._make_observation(score=1.0)
        self._make_observation(score=2.0)
        output, produce = self._run_batch(alert)
        assert output.alerts_fired == 1
        assert produce.call_args.kwargs["properties"]["metric_value"] == 1.5

    def test_undelivered_notification_rolls_back_state(self) -> None:
        alert = self._make_alert()
        self._make_observation()
        self._make_observation()
        output, _ = self._run_batch(alert, delivered=False)
        assert output.alerts_fired == 0
        assert alert.state == VisionAlertState.NOT_FIRING
        assert alert.last_notified_at is None
        # The next check re-evaluates and re-fires instead of assuming "notified".
        alert.next_check_at = None
        alert.save(update_fields=["next_check_at"])
        output, _ = self._run_batch(alert, delivered=True)
        assert output.alerts_fired == 1
        assert alert.state == VisionAlertState.FIRING

    def test_discover_skips_match_broken_snoozed_and_future(self) -> None:
        due = self._make_alert()
        self._make_alert(name="match", kind=VisionAlertKind.MATCH, threshold=None)
        self._make_alert(name="broken", state=VisionAlertState.BROKEN)
        self._make_alert(
            name="snoozed",
            state=VisionAlertState.SNOOZED,
            snooze_until=datetime.now(UTC) + timedelta(hours=1),
        )
        self._make_alert(name="future", next_check_at=datetime.now(UTC) + timedelta(hours=1))
        self._make_alert(name="disabled", enabled=False)

        output = _discover_due(DiscoverDueAlertsInput())
        assert output.batches == [[str(due.id)]]
