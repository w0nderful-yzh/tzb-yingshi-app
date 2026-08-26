from __future__ import annotations

import unittest
from unittest.mock import Mock

from app.modules.fall.multimodal_engine.services.risk_event import RiskEventNotFoundError, RiskEventService


class RiskEventDeletionTest(unittest.TestCase):
    def test_delete_event_uses_existing_repository_and_commits(self) -> None:
        session = Mock()
        repository = Mock()
        event = object()
        repository.get_by_event_id.return_value = event
        service = RiskEventService(
            session,
            repository=repository,
            monitoring_repository=Mock(),
        )

        service.delete_event("event-001")

        repository.delete.assert_called_once_with(event)
        session.commit.assert_called_once_with()

    def test_delete_missing_event_does_not_commit(self) -> None:
        session = Mock()
        repository = Mock()
        repository.get_by_event_id.return_value = None
        service = RiskEventService(
            session,
            repository=repository,
            monitoring_repository=Mock(),
        )

        with self.assertRaises(RiskEventNotFoundError):
            service.delete_event("missing")

        repository.delete.assert_not_called()
        session.commit.assert_not_called()

    def test_bulk_delete_commits_once_and_returns_deleted_count(self) -> None:
        session = Mock()
        repository = Mock()
        repository.delete_by_event_ids.return_value = 2
        service = RiskEventService(
            session,
            repository=repository,
            monitoring_repository=Mock(),
        )

        deleted_count = service.delete_events(["event-001", "event-002"])

        self.assertEqual(deleted_count, 2)
        repository.delete_by_event_ids.assert_called_once_with(
            ["event-001", "event-002"]
        )
        session.commit.assert_called_once_with()

    def test_delete_all_commits_once_and_returns_deleted_count(self) -> None:
        session = Mock()
        repository = Mock()
        repository.delete_all.return_value = 7
        service = RiskEventService(
            session,
            repository=repository,
            monitoring_repository=Mock(),
        )

        deleted_count = service.delete_all_events()

        self.assertEqual(deleted_count, 7)
        repository.delete_all.assert_called_once_with()
        session.commit.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
