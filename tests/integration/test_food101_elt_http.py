"""Loopback-only end-to-end checks for the real Food-101 ELT runner."""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml
from PIL import Image

from food_assistant.data_pipeline import pipeline
from food_assistant.ml import food101_verifier


def _build_fixture_archive(path: Path, *, traversal: bool = False) -> bytes:
    """Create a tiny official-shaped archive containing three valid JPEGs."""
    members: list[tuple[str, bytes]] = [
        ("food-101/meta/classes.txt", b"pizza\n"),
        ("food-101/meta/train.txt", b"pizza/train_a\npizza/train_b\n"),
        ("food-101/meta/test.txt", b"pizza/test_a\n"),
        (
            "food-101/meta/train.json",
            json.dumps({"pizza": ["pizza/train_a", "pizza/train_b"]}).encode(),
        ),
        ("food-101/meta/test.json", json.dumps({"pizza": ["pizza/test_a"]}).encode()),
        ("food-101/meta/labels.txt", b"Pizza\n"),
        ("food-101/license_agreement.txt", os.urandom(2 * 1024 * 1024)),
    ]
    colors = ((220, 30, 20), (20, 210, 40), (30, 40, 220))
    image_ids = ("train_a", "train_b", "test_a")
    for image_id, color in zip(image_ids, colors, strict=True):
        buffer = io.BytesIO()
        Image.new("RGB", (24, 18), color).save(buffer, format="JPEG")
        members.append((f"food-101/images/pizza/{image_id}.jpg", buffer.getvalue()))
    if traversal:
        members.append(("food-101/../../escape.txt", b"must not publish"))

    with tarfile.open(path, "w:gz") as archive:
        for name, body in members:
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    return path.read_bytes()


def _serve_archive(payload: bytes, *, interrupt_once: bool = False):
    """Start an ephemeral loopback server with deterministic retry/range behavior."""
    observations: dict[str, object] = {"ranges": [], "probe_503": 0, "interrupted": 0}

    class ArchiveHandler(BaseHTTPRequestHandler):
        """Serve only the in-memory fixture and record observed byte ranges."""

        protocol_version = "HTTP/1.1"

        def log_message(self, _format: str, *_args: object) -> None:
            """Keep fixture HTTP logs out of test output."""

        def do_GET(self) -> None:
            """Serve a strict 206 response and inject one bounded transient fault."""
            range_header = self.headers.get("Range", "")
            ranges = observations["ranges"]
            assert isinstance(ranges, list)
            ranges.append(range_header)
            if range_header == "bytes=0-0" and observations["probe_503"] == 0:
                observations["probe_503"] = 1
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start_text, end_text = range_header.removeprefix("bytes=").split("-")
            start, end = int(start_text), int(end_text)
            if start < 0 or end >= len(payload) or start > end:
                self.send_error(416)
                return
            body = payload[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", '"fixture-v1"')
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            if (
                interrupt_once
                and start == 0
                and end == len(payload) - 1
                and observations["interrupted"] == 0
            ):
                observations["interrupted"] = 1
                self.wfile.write(body[: 1024 * 1024])
                self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(body)
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), ArchiveHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address, port = server.server_address
    return server, thread, f"http://{address}:{port}/fixture.tar.gz", observations


def _write_config(workspace: Path, archive_url: str) -> Path:
    """Write a temporary supported POC config whose every output stays in tmp_path."""
    (workspace / "configs").mkdir(parents=True, exist_ok=True)
    (workspace / "configs/classes.yaml").write_text("classes:\n  - pizza\n", encoding="utf-8")
    value = {
        "schema_version": 1,
        "paths": {
            "raw_root": "data/raw/food-101",
            "staging_root": "data/staging/food101",
            "archive": "data/raw/food-101.tar.gz",
            "manifest_root": "data/splits/poc-v2",
            "report_root": "data/reports/elt/poc-v2",
        },
        "approval_file": "configs/approval.json",
        "class_selection_file": "configs/classes.yaml",
        "dataset_scope": "poc",
        "split_version": "poc-v2",
        "seed": 42,
        "val_fraction": 0.5,
        "download": {"policy": "approved_only", "retries": 2, "backoff_seconds": 0},
        "quality": {"require_full_integrity": True, "reject_cross_split_duplicates": True},
        "locks": {"timeout_seconds": 2},
        "fixtures": {"enabled": True, "root": "tests/fixtures"},
    }
    config_path = workspace / "configs/elt-http.yaml"
    config_path.write_text(yaml.safe_dump(value), encoding="utf-8")
    (workspace / "configs/approval.json").write_text(
        json.dumps({"archive_url": archive_url, "academic_approval": "test-only"}),
        encoding="utf-8",
    )
    return config_path


def _install_fixture_contract(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    """Bind the runner to this explicit loopback fixture without changing production policy."""
    for module in (food101_verifier,):
        monkeypatch.setattr(module, "EXPECTED_CLASSES", 1)
        monkeypatch.setattr(module, "EXPECTED_IMAGES", 3)
        monkeypatch.setattr(module, "EXPECTED_TRAIN_PER_CLASS", 2)
        monkeypatch.setattr(module, "EXPECTED_TEST_PER_CLASS", 1)

    from scripts import download_food101

    monkeypatch.setattr(download_food101, "EXPECTED_CLASSES", 1)
    monkeypatch.setattr(download_food101, "EXPECTED_IMAGES", 3)
    monkeypatch.setattr(download_food101, "EXPECTED_TRAIN_PER_CLASS", 2)
    monkeypatch.setattr(download_food101, "EXPECTED_TEST_PER_CLASS", 1)
    monkeypatch.setattr(
        pipeline,
        "_approved_source",
        lambda _config: {
            "academic_approval": "confirmed_by_user",
            "archive_url": url,
            "source_url": url,
            "official_sha256": None,
            "academic_approval_date": "synthetic test fixture",
            "citation": "synthetic test fixture",
            "use": "isolated integration test",
            "rights_status": "synthetic",
        },
    )


@pytest.mark.parametrize("interrupt_once", [False, True])
def test_loopback_download_extract_quality_transform_validate_report(
    tmp_path, monkeypatch, capsys, interrupt_once
):
    """Run real HTTP Extract through real Safe Load and POC Transform to report."""
    archive_path = tmp_path / "fixture.tar.gz"
    payload = _build_fixture_archive(archive_path)
    server, thread, url, observations = _serve_archive(payload, interrupt_once=interrupt_once)
    try:
        config_path = _write_config(tmp_path, url)
        _install_fixture_contract(monkeypatch, url)
        monkeypatch.chdir(tmp_path)

        from scripts.run_food101_elt import main

        run_id = f"http-e2e-{interrupt_once}"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_food101_elt.py",
                "--config",
                str(config_path),
                "--stage",
                "all",
                "--run-id",
                run_id,
            ],
        )
        exit_code = main()
        result = json.loads(capsys.readouterr().out)

        assert exit_code == 0
        assert result["status"] == "PASS"
        assert [
            result["stages"][stage]["status"]
            for stage in (
                "inspect",
                "extract",
                "load",
                "quality",
                "transform",
                "validate",
                "report",
            )
        ] == ["PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS"]
        assert result["stages"]["quality"]["details"]["actual_image_count"] == 3
        assert result["stages"]["validate"]["split_counts"] == {
            "train": 1,
            "val": 1,
            "test": 1,
        }
        assert (tmp_path / "data/raw/food-101/download_metadata.json").is_file()
        manifest = tmp_path / "data/splits/poc-v2/dataset_manifest.csv"
        with manifest.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 3
        assert {row["class_id"] for row in rows} == {"0"}
        assert {row["split"] for row in rows} == {"train", "val", "test"}
        report_dir = tmp_path / "artifacts/reports/elt" / f"http-e2e-{interrupt_once}"
        saved = json.loads((report_dir / "run_report.json").read_text(encoding="utf-8"))
        assert saved["status"] == "PASS"
        assert (report_dir / "run_report.md").is_file()
        assert observations["probe_503"] == 1
        assert urlsplit(url).hostname == "127.0.0.1"
        if interrupt_once:
            assert observations["interrupted"] == 1
            ranges = observations["ranges"]
            assert isinstance(ranges, list)
            assert any(value.startswith("bytes=1048576-") for value in ranges)
        else:
            assert observations["interrupted"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_loopback_path_traversal_fails_without_raw_or_manifest_publish(
    tmp_path, monkeypatch, capsys
):
    """Ensure an unsafe archive fails Load and blocks downstream publication."""
    archive_path = tmp_path / "traversal.tar.gz"
    payload = _build_fixture_archive(archive_path, traversal=True)
    server, thread, url, _observations = _serve_archive(payload)
    try:
        config_path = _write_config(tmp_path, url)
        _install_fixture_contract(monkeypatch, url)
        monkeypatch.chdir(tmp_path)

        from scripts.run_food101_elt import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_food101_elt.py",
                "--config",
                str(config_path),
                "--stage",
                "all",
                "--run-id",
                "http-traversal",
            ],
        )
        exit_code = main()
        result = json.loads(capsys.readouterr().out)

        assert exit_code == 1
        assert result["status"] == "FAIL"
        assert result["stages"]["extract"]["status"] == "PASS"
        assert result["stages"]["load"]["status"] == "FAIL"
        assert result["stages"]["transform"]["status"] == "BLOCKED"
        assert not (tmp_path / "data/raw/food-101").exists()
        assert not (tmp_path / "data/splits/poc-v2").exists()
        assert not (tmp_path / "escape.txt").exists()
        report = tmp_path / "artifacts/reports/elt/http-traversal/run_report.json"
        assert json.loads(report.read_text(encoding="utf-8"))["status"] == "FAIL"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
