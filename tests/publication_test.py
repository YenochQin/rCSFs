"""Atomic visibility and failure-reporting for CLI artifact publication."""

import errno
import os
from pathlib import Path

import pytest

from rcsfs import _publication


def test_same_volume_publish_is_complete_and_will_not_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"complete")
    destination = tmp_path / "destination"
    _publication.publish_outputs([source], [destination])
    assert destination.read_bytes() == b"complete"
    assert source.read_bytes() == b"complete"

    competitor = tmp_path / "competitor"
    competitor.write_bytes(b"another writer")
    with pytest.raises(_publication.PartialPublicationError) as raised:
        _publication.publish_outputs([competitor], [destination])
    assert raised.value.published == []
    assert destination.read_bytes() == b"complete"


def test_cross_volume_copy_is_hidden_until_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"complete")
    destination = tmp_path / "destination"
    original_link = os.link
    attempts: list[Path] = []

    def cross_volume_once(left: Path, right: Path) -> None:
        attempts.append(left)
        if left == source:
            raise OSError(errno.EXDEV, "cross-device link")
        assert right == destination
        assert left.read_bytes() == b"complete"
        assert not destination.exists()
        original_link(left, right)

    monkeypatch.setattr(_publication.os, "link", cross_volume_once)
    _publication.publish_outputs([source], [destination])
    assert len(attempts) == 2
    assert destination.read_bytes() == b"complete"
    assert not list(tmp_path.glob(".destination.rcsfs-*.tmp"))


def test_cross_volume_race_keeps_competitor_and_removes_private_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"complete")
    destination = tmp_path / "destination"
    original_link = os.link

    def competitor_wins(left: Path, right: Path) -> None:
        if left == source:
            raise OSError(errno.EXDEV, "cross-device link")
        destination.write_bytes(b"competitor")
        original_link(left, right)

    monkeypatch.setattr(_publication.os, "link", competitor_wins)
    with pytest.raises(_publication.PartialPublicationError) as raised:
        _publication.publish_outputs([source], [destination])
    assert raised.value.published == []
    assert destination.read_bytes() == b"competitor"
    assert not list(tmp_path.glob(".destination.rcsfs-*.tmp"))


def test_later_failure_reports_earlier_publications_without_deleting_them(
    tmp_path: Path,
) -> None:
    sources = [tmp_path / "first-source", tmp_path / "second-source"]
    for source in sources:
        source.write_bytes(b"complete")
    destinations = [tmp_path / "first", tmp_path / "second"]
    destinations[1].write_bytes(b"competitor")
    with pytest.raises(_publication.PartialPublicationError) as raised:
        _publication.publish_outputs(sources, destinations)
    assert raised.value.published == destinations[:1]
    assert raised.value.destination == destinations[1]
    assert destinations[0].read_bytes() == b"complete"
    assert destinations[1].read_bytes() == b"competitor"
