"""
tests/unit/test_api_security.py
Security and session-cleanup tests for api.py.

New tests added by code review (round 1):
  test_upload_path_traversal_prevented_when_filename_contains_dotdot   (HIGH – Bug 3)
  test_upload_path_traversal_prevented_when_filename_is_absolute_path  (HIGH – Bug 3)
  test_cleanup_removes_upload_dir_when_session_expires                  (MEDIUM – Edge Case 3)
  test_upload_async_read_used_instead_of_sync_read                     (HIGH – Bug 4)

New tests added by code review (round 2):
  test_upload_response_filename_is_sanitised_basename_…  (MEDIUM – raw filename in response)
  test_cleanup_loop_logs_exception_not_silently_dropped  (MEDIUM – silent exception swallow)
"""

import os
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import app, SessionManager


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def session_id(client):
    resp = client.post("/api/session/create")
    assert resp.status_code == 200
    return resp.json()["session_id"]


# ────────────────────────────────────────────────────────────────────────────
# HIGH – Bug 3: Path traversal in file upload
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestImageUploadPathTraversal:
    """HIGH: file.filename must be sanitised before joining to session_dir."""

    def test_upload_path_traversal_prevented_when_filename_contains_dotdot(
        self, client, session_id
    ):
        """
        HIGH - Sanitise file.filename with Path(name).name before path join.
        Attacker supplies '../../evil.sh'; stored file must stay inside the
        session upload directory, not two levels above it.
        """
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"fake image data")
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                resp = client.post(
                    "/api/upload-image",
                    data={"session_id": session_id},
                    files={"file": ("../../evil.sh", f, "image/jpeg")},
                )

            assert resp.status_code == 200
            stored = Path(resp.json()["file_path"])

            # After fix: session_id must be an ancestor of the stored path
            assert session_id in str(stored), (
                f"Stored path {stored} does not contain session_id. "
                "Path traversal not prevented."
            )
            # The stored filename must not contain '..'
            assert ".." not in stored.parts, (
                f"Path component '..' found in {stored.parts}. "
                "Directory traversal still possible."
            )
            # The filename must be only the basename 'evil.sh', not '../../evil.sh'.
            # LOW: use 'in' rather than exact equality so the assertion survives if
            # the server ever prefixes filenames (e.g. with a UUID).
            assert "evil.sh" in stored.name, (
                f"Expected 'evil.sh' in stored name, got '{stored.name}'"
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_upload_path_traversal_prevented_when_filename_is_absolute_path(
        self, client, session_id
    ):
        """
        HIGH - Absolute path like '/etc/cron.d/pwn' must be reduced to 'pwn'.
        """
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"data")
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                resp = client.post(
                    "/api/upload-image",
                    data={"session_id": session_id},
                    files={"file": ("/etc/cron.d/pwn", f, "image/jpeg")},
                )

            assert resp.status_code == 200
            stored = Path(resp.json()["file_path"])
            assert stored.name == "pwn", (
                f"Expected basename 'pwn', got '{stored.name}'. "
                "Absolute-path traversal not prevented."
            )
            assert session_id in str(stored)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# MEDIUM – Edge Case 3: Cleanup must remove upload directories
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestSessionCleanupRemovesUploads:
    """MEDIUM: cleanup_expired_sessions() must delete upload dirs for expired sessions."""

    def test_cleanup_removes_upload_dir_when_session_expires(self):
        """
        MEDIUM: api.py cleanup_expired_sessions() deletes the session record but
        currently leaves uploaded files on disk.  After the fix, the upload
        directory for the expired session must also be removed.
        """
        from datetime import datetime, timedelta

        # Arrange — create a manager with a very short expiry
        manager = SessionManager(expiration_hours=1)

        # Create a session and its upload directory manually
        sid = manager.create_session()
        session_dir = manager.upload_dir / sid
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "damage_photo.jpg").write_bytes(b"fake image")

        # Back-date last_updated so the session looks expired
        manager.sessions[sid].last_updated = (
            datetime.now() - timedelta(hours=2)
        )

        assert session_dir.exists(), "Pre-condition: upload dir must exist"

        # Act
        removed_count = manager.cleanup_expired_sessions()

        # Assert — session removed from dict and upload dir deleted
        assert sid not in manager.sessions, "Expired session must be removed from dict"
        assert removed_count == 1
        assert not session_dir.exists(), (
            f"Upload directory {session_dir} still exists after cleanup. "
            "Uploaded files are not being cleaned up for expired sessions."
        )

    def test_cleanup_leaves_active_session_upload_dir_intact(self):
        """Cleanup must not remove dirs for sessions that are still active."""
        # LOW: test teardown was after the assertions, so a failing assertion leaked
        # the temp directory.  Moved cleanup into a finally block so it always runs.
        import shutil
        manager = SessionManager(expiration_hours=24)

        sid = manager.create_session()
        session_dir = manager.upload_dir / sid
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "photo.jpg").write_bytes(b"data")

        try:
            manager.cleanup_expired_sessions()

            assert sid in manager.sessions, "Active session must not be removed"
            assert session_dir.exists(), "Active session upload dir must not be removed"
        finally:
            shutil.rmtree(session_dir, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────────────
# HIGH – Bug 4: async upload handler must use await file.read()
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestUploadImageUsesAsyncRead:
    """HIGH: upload_image is async; it must use await file.read(), not file.file.read()."""

    def test_upload_image_uses_await_file_read_not_sync_file_read(self):
        """
        HIGH – Replace file.file.read() with await file.read() in SessionManager.upload_image.
        The synchronous file.file.read() blocks the event loop; await file.read() does not.
        We verify this by inspecting the source of SessionManager.upload_image.
        """
        import inspect
        from src.api import SessionManager

        source = inspect.getsource(SessionManager.upload_image)

        assert "await file.read()" in source, (
            "SessionManager.upload_image must use 'await file.read()' (async, non-blocking). "
            "Found synchronous 'file.file.read()' instead, which blocks the event loop.\n"
            f"Source:\n{source}"
        )
        # Check only non-comment lines for the synchronous call
        code_lines = [
            ln for ln in source.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        assert "file.file.read()" not in code_only, (
            "Synchronous 'file.file.read()' still present in upload_image code (not a comment)."
        )


# ────────────────────────────────────────────────────────────────────────────
# MEDIUM: Response filename must be the sanitised basename, not the raw input
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestUploadResponseFilename:
    """MEDIUM: ImageUploadResponse.filename must reflect the sanitised stored name."""

    def test_upload_response_filename_is_sanitised_basename_when_path_traversal_supplied(
        self, client, session_id
    ):
        """
        MEDIUM: The upload endpoint returns file.filename (raw, user-supplied) in
        ImageUploadResponse.filename. After the fix it must return the sanitised
        basename so callers cannot observe the original traversal path.

        Before fix: response["filename"] == "../../evil.sh"
        After fix:  response["filename"] == "evil.sh"
        """
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"data")
            tmp_path = tmp.name

        try:
            # Arrange / Act
            with open(tmp_path, "rb") as f:
                resp = client.post(
                    "/api/upload-image",
                    data={"session_id": session_id},
                    files={"file": ("../../evil.sh", f, "image/jpeg")},
                )

            assert resp.status_code == 200
            # Assert
            assert resp.json()["filename"] == "evil.sh", (
                f"Response filename '{resp.json()['filename']}' exposes the raw traversal "
                "path. The endpoint must return only the sanitised basename."
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# MEDIUM: Background cleanup loop must log exceptions, not silently drop them
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestCleanupLoopLogging:
    """MEDIUM: Background cleanup loop must log exceptions rather than bare-pass them."""

    def test_cleanup_loop_logs_exception_not_silently_dropped(self):
        """
        MEDIUM: Background cleanup loop uses bare 'except Exception: pass', which
        silently discards errors and leaves ops with no visibility into failures.
        After the fix the exception must be bound ('except Exception as exc:') and
        logged/printed so that repeated failures surface in application logs.

        Test: inspect create_app source for 'except Exception as exc' in the
        cleanup loop — absent before fix, present after.
        """
        import inspect
        import src.api as api_module

        # Arrange / Act
        source = inspect.getsource(api_module.create_app)

        # Assert — exception must be bound for logging, not silently discarded
        assert "except Exception as exc" in source, (
            "Background _cleanup_loop must use 'except Exception as exc' to enable "
            "logging. Currently 'except Exception: pass' silently discards errors."
        )