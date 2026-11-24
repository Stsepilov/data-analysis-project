import json
import os
import subprocess
import sys


def test_offline_demo_is_reproducible_and_journalled(tmp_path):
    def run():
        return subprocess.run(
            [sys.executable, "-m", "river_forecast", "demo"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )

    run()
    metrics_path = tmp_path / "artifacts/demo/metrics.json"
    first_metrics = metrics_path.read_bytes()
    first_daily = (tmp_path / "data/processed/demo/daily.csv").read_bytes()
    assert json.loads(first_metrics)["data_kind"] == "synthetic_demo"
    run()
    assert metrics_path.read_bytes() == first_metrics
    assert (tmp_path / "data/processed/demo/daily.csv").read_bytes() == first_daily
    records = [
        json.loads(line) for line in (tmp_path / "artifacts/runs.jsonl").read_text().splitlines()
    ]
    assert [r["status"] for r in records] == ["started", "success", "started", "success"]


def test_failed_run_is_logged_and_never_switches_to_demo(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "river_forecast", "fetch", "--config", "missing.toml"],
        cwd=tmp_path,
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "artifacts/runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [r["status"] for r in records] == ["started", "failed"]
    assert records[-1]["error_type"] == "FileNotFoundError"
    assert records[-1]["error"] == "Файл не найден: missing.toml"
    assert records[-1]["error"] in result.stderr
    assert not (tmp_path / "data/processed/demo").exists()
