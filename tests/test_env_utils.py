"""Blank or malformed environment variables must fall back to defaults, not crash on import."""
import os
import subprocess
import sys
from pathlib import Path

from delivery_agent.env_utils import env_float, env_int, env_str

ROOT = Path(__file__).resolve().parents[1]


def test_blank_and_invalid_values_use_the_default(monkeypatch):
    monkeypatch.setenv("VENLIX_TEST_NUM", "")
    assert env_float("VENLIX_TEST_NUM", 3.0) == 3.0
    assert env_int("VENLIX_TEST_NUM", 4) == 4
    assert env_str("VENLIX_TEST_NUM", "x") == "x"
    monkeypatch.setenv("VENLIX_TEST_NUM", "  abc ")
    assert env_float("VENLIX_TEST_NUM", 3.0) == 3.0
    assert env_int("VENLIX_TEST_NUM", 4) == 4
    monkeypatch.setenv("VENLIX_TEST_NUM", " 2.5 ")
    assert env_float("VENLIX_TEST_NUM", 3.0) == 2.5
    monkeypatch.setenv("VENLIX_TEST_NUM", "8")
    assert env_int("VENLIX_TEST_NUM", 4) == 8


def test_app_starts_with_blank_settings(tmp_path):
    # Reproduces the Vercel failure: variables created with empty values (e.g. pasted .env.example).
    blank = ["BACKEND_URL", "BACKEND_TIMEOUT", "BACKEND_CACHE_TTL", "BACKEND_RETRY_AFTER",
             "AGENT_CONCURRENCY", "AGENT_LLM_TIMEOUT", "LLM_MAX_HISTORY", "GEMINI_RETRY_DELAY"]
    env = {k: v for k, v in os.environ.items() if k not in blank}
    env.update({name: "" for name in blank})
    env["VENLIX_DB_PATH"] = str(tmp_path / "cases.db")
    code = ("import main; from fastapi.testclient import TestClient; "
            "print(type(main.app).__name__, TestClient(main.app).get('/api/health').status_code)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split()[-2:] == ["FastAPI", "200"], out.stdout + out.stderr
