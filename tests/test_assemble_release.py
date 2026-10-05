"""Tests for release-asset assembly from index parts (HF publish path)."""

import hashlib
import sys
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from assemble_release import assemble  # noqa: E402

PART_A = b"A" * 1024
PART_B = b"B" * 700


class TestAssemble(unittest.TestCase):
    def setUp(self):
        import hashlib

        self.whole = PART_A + PART_B
        self.sha = hashlib.sha256(self.whole).hexdigest()
        self.bad_sha = hashlib.sha256(b"other").hexdigest()

    def _serve(self, handler):
        server = HTTPServer(("127.0.0.1", 0), handler)
        Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}"

    def _entry(self, base: str, sha: str) -> dict:
        return {
            "parts": [
                {"url": f"{base}/x.part-00", "sha256": hashlib.sha256(PART_A).hexdigest(),
                    "size": len(PART_A)},
                {"url": f"{base}/x.part-01", "sha256": hashlib.sha256(PART_B).hexdigest(),
                    "size": len(PART_B)},
            ],
            "sha256": sha,
            "filename": "x-1.0.zip",
        }

    def test_assembles_parts_in_order_and_verifies_sha(self):
        parts = {"x.part-00": PART_A, "x.part-01": PART_B}
        base = self._serve(_files_handler(parts))
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        out = Path(d.name) / "x.zip"
        assemble(self._entry(base, self.sha), out)
        self.assertEqual(out.read_bytes(), self.whole)

    def test_sha_mismatch_rejected(self):
        parts = {"x.part-00": PART_A, "x.part-01": PART_B}
        base = self._serve(_files_handler(parts))
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        out = Path(d.name) / "x.zip"
        with self.assertRaisesRegex(ValueError, "sha256"):
            assemble(self._entry(base, self.bad_sha), out)

    def test_single_url_entry_assembles_and_verifies(self):
        base = self._serve(_files_handler({"x-1.0.zip": self.whole}))
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        out = Path(d.name) / "x.zip"
        entry = {"url": f"{base}/x-1.0.zip", "sha256": self.sha, "filename": "x-1.0.zip"}
        assemble(entry, out)
        self.assertEqual(out.read_bytes(), self.whole)

    def test_single_url_entry_sha_mismatch_rejected(self):
        base = self._serve(_files_handler({"x-1.0.zip": self.whole}))
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        out = Path(d.name) / "x.zip"
        entry = {"url": f"{base}/x-1.0.zip", "sha256": self.bad_sha, "filename": "x-1.0.zip"}
        with self.assertRaisesRegex(ValueError, "sha256"):
            assemble(entry, out)

    def test_part_sha_mismatch_rejected(self):
        parts = {"x.part-00": b"X" * 1024, "x.part-01": PART_B}
        base = self._serve(_files_handler(parts))
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        out = Path(d.name) / "x.zip"
        with self.assertRaisesRegex(ValueError, "part-00"):
            assemble(self._entry(base, self.sha), out)


def _files_handler(parts: dict[str, bytes]):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            name = self.path.rsplit("/", 1)[-1]
            data = parts.get(name)
            if data is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    return H


import contextlib  # noqa: E402
import tempfile  # noqa: E402


@contextlib.contextmanager
def _tmpdir():
    d = tempfile.TemporaryDirectory()
    try:
        yield Path(d.name)
    finally:
        d.cleanup()


if __name__ == "__main__":
    unittest.main()
