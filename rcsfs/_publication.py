"""No-overwrite publication of completed CLI generation artifacts.

Each destination becomes visible atomically, but several destinations are not
one transaction. On failure, successful publications stay in place and are
reported; deleting them automatically would risk removing a concurrent user's
replacement of one of those paths.
"""

from __future__ import annotations

import errno
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path


class PartialPublicationError(OSError):
    """A publication failed after zero or more destinations became visible."""

    def __init__(self, destination: Path, published: list[Path], cause: OSError) -> None:
        self.destination = destination
        self.published = published
        names = ", ".join(str(path) for path in published) or "none"
        super().__init__(
            f"Could not publish {destination}: {cause}. Already published: {names}. "
            "Completed files are left in place; inspect them before removing or retrying."
        )


def publish_outputs(sources: Sequence[Path], destinations: Sequence[Path]) -> None:
    """Publish complete files with an atomic create-if-absent per destination.

    Same-filesystem sources are hard-linked directly. For a cross-filesystem
    source, copy to a private file beside the destination, flush it, and then
    hard-link that complete file to the final name. Neither path ever exposes
    a partially copied final file or replaces a competitor's destination.
    """
    if len(sources) != len(destinations):
        raise ValueError("Publication source and destination counts differ")
    published: list[Path] = []
    for source, destination in zip(sources, destinations, strict=True):
        try:
            _publish_one(source, destination)
        except OSError as error:
            raise PartialPublicationError(destination, published.copy(), error) from error
        published.append(destination)


def _publish_one(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
        return
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise

    # The staging tree is on another filesystem. The private target-side file
    # is complete before its name is linked into place; cleanup is attempted
    # even when another process wins the final-name race.
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.rcsfs-",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as writer:
            temporary = Path(writer.name)
            with source.open("rb") as reader:
                shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            # Once the final hard link exists, a temporary-file cleanup error
            # must not turn a completed publication into a false failure.
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
