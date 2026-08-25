"""Observation-completion hook for match-kind vision alerts.

Called from inside the observation terminal-state transaction, after the conditional
status UPDATE has actually transitioned the row (the exactly-once guard). Inserts one
VisionAlertMatch outbox row per matching enabled match alert; the alert-check workflow
drains undelivered rows into one bundled notification per alert per tick.
"""

from __future__ import annotations

from uuid import UUID

from django.db import transaction

import structlog

from posthog.exceptions_capture import capture_exception

from products.replay_vision.backend.models.replay_observation import ReplayObservation
from products.replay_vision.backend.models.vision_alert import (
    PREDICATE_SELECTION_KEYS,
    VisionAlertConfiguration,
    VisionAlertKind,
    VisionAlertMatch,
    selection_statuses,
)
from products.replay_vision.backend.temporal.vision_actions.synthesis import apply_observation_predicate

logger = structlog.get_logger(__name__)


def record_alert_matches_guarded(*, observation_id: UUID, status: str) -> None:
    """Match the observation against enabled match alerts inside a savepoint.

    A hook failure must not roll back the observation's status transition, so the
    insert runs in a nested atomic block and any exception is swallowed after logging:
    a lost match beats a broken scan.
    """
    try:
        with transaction.atomic():
            _record_alert_matches(observation_id=observation_id, status=status)
    except Exception as e:
        capture_exception(e, {"observation_id": str(observation_id), "phase": "vision_alert_match_hook"})
        logger.exception("vision_alert.match_hook_failed", observation_id=str(observation_id))


def _record_alert_matches(*, observation_id: UUID, status: str) -> int:
    observation = ReplayObservation.objects.filter(pk=observation_id).values("team_id", "scanner_id").first()
    if observation is None:
        return 0

    alerts = list(
        VisionAlertConfiguration.all_teams.filter(
            team_id=observation["team_id"],
            scanner_id=observation["scanner_id"],
            kind=VisionAlertKind.MATCH,
            enabled=True,
        ).only("id", "selection")
    )
    if not alerts:
        return 0

    rows: list[VisionAlertMatch] = []
    for alert in alerts:
        selection = alert.selection or {}
        if status not in selection_statuses(selection):
            continue
        has_predicate = any(selection.get(key) for key in PREDICATE_SELECTION_KEYS)
        if has_predicate:
            # Single-row predicate check; inside this transaction it sees the
            # scanner_result written by the status transition.
            if not apply_observation_predicate(ReplayObservation.objects.filter(pk=observation_id), selection).exists():
                continue
        rows.append(
            VisionAlertMatch(
                alert_id=alert.id,
                observation_id=observation_id,
                team_id=observation["team_id"],
            )
        )

    if rows:
        # ignore_conflicts: the unique (alert, observation) constraint makes a
        # double insert structurally impossible even if the exactly-once guard slips.
        VisionAlertMatch.all_teams.bulk_create(rows, ignore_conflicts=True)
    return len(rows)
