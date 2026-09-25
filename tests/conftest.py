"""
Shared fixtures. The environment is pinned *before* the agent is imported so a
developer's .env (real API keys, backend URL) never leaks into the tests.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ["GEMINI_API_KEY"] = ""
os.environ["OPENROUTER_API_KEY"] = ""
os.environ["GEMINI_BASE_URL"] = ""
os.environ["BACKEND_URL"] = "http://127.0.0.1:9"  # closed port: fails fast
os.environ["ALLOW_MOCK_FALLBACK"] = "true"
os.environ["GEMINI_MODEL"] = ""
os.environ["GEMINI_TASK_MODEL"] = ""
os.environ["GEMINI_RETRY_DELAY"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from delivery_agent import store  # noqa: E402
from delivery_agent.llm_manager import client, copilot  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("VENLIX_DB_PATH", str(tmp_path / "cases.db"))
    store.reset_singletons()
    client.reset_llm_metrics()
    client.reset_model_state()
    client.set_force_primary_failure(False)
    copilot.clear_backend_cache()
    yield
    store.reset_singletons()
    client.set_force_primary_failure(False)


class FakeLLMServer:
    """Speaks just enough of the Gemini REST API and the OpenAI chat API."""

    def __init__(self):
        self.requests = []
        self.gemini_text = "Hello from Gemini"
        self.openai_text = "Hello from OpenRouter"
        self.gemini_status = 200
        self.gemini_reject_thinking = False
        self.gemini_generic_thinking_error = False
        self.gemini_model_errors = {}  # model -> (http code, status, message)
        self.openai_status = 200
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body, content_type="application/json"):
                data = body.encode() if isinstance(body, str) else body
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                server.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
                if ":generateContent" in self.path or ":streamGenerateContent" in self.path:
                    return self._gemini(body)
                if self.path.endswith("/chat/completions"):
                    return self._openai(body)
                self._send(404, json.dumps({"error": "not found"}))

            def _gemini_error(self, code, status, message):
                return self._send(code, json.dumps({"error": {"code": code, "status": status, "message": message}}))

            def _gemini(self, body):
                model = self.path.split("/models/")[1].split(":")[0]
                thinking = (body.get("generationConfig") or {}).get("thinkingConfig")
                # The real API rejects request deadlines under 10 s
                deadline = self.headers.get("X-Server-Timeout")
                if deadline and int(deadline) < 10:
                    return self._gemini_error(400, "INVALID_ARGUMENT",
                                              f"Manually set deadline {deadline}s is too short. Minimum allowed deadline is 10s.")
                if model in server.gemini_model_errors:
                    return self._gemini_error(*server.gemini_model_errors[model])
                if server.gemini_generic_thinking_error and thinking is not None:
                    return self._gemini_error(400, "INVALID_ARGUMENT", "Request contains an invalid argument.")
                if server.gemini_reject_thinking and thinking is not None:
                    return self._send(400, json.dumps({"error": {
                        "code": 400, "status": "INVALID_ARGUMENT",
                        "message": "Budget 0 is invalid. This model only works in thinking mode."}}))
                if server.gemini_status != 200:
                    return self._send(server.gemini_status, json.dumps({"error": {
                        "code": server.gemini_status, "status": "UNAVAILABLE", "message": "The model is overloaded."}}))
                if ":streamGenerateContent" in self.path:
                    words = server.gemini_text.split(" ")
                    chunks = [w + (" " if i < len(words) - 1 else "") for i, w in enumerate(words)]
                    payload = "".join(
                        "data: " + json.dumps({"candidates": [{"content": {"role": "model", "parts": [{"text": c}]}}]}) + "\r\n\r\n"
                        for c in chunks)
                    return self._send(200, payload, "text/event-stream")
                return self._send(200, json.dumps({"candidates": [{
                    "content": {"role": "model", "parts": [{"text": server.gemini_text}]}, "finishReason": "STOP"}]}))

            def _openai(self, body):
                if server.openai_status != 200:
                    return self._send(server.openai_status, json.dumps({"error": {"message": "rate limited"}}))
                if body.get("stream"):
                    payload = ""
                    for word in server.openai_text.split(" "):
                        chunk = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": body["model"],
                                 "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}]}
                        payload += "data: " + json.dumps(chunk) + "\n\n"
                    payload += "data: [DONE]\n\n"
                    return self._send(200, payload, "text/event-stream")
                return self._send(200, json.dumps({
                    "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": server.openai_text}}]}))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def last(self, kind):
        marker = "/chat/completions" if kind == "openai" else "Content"
        return [r for r in self.requests if marker in r["path"]][-1]


@pytest.fixture
def fake_llm(monkeypatch):
    server = FakeLLMServer()
    server.thread.start()
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_BASE_URL", server.url)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", server.url + "/v1")
    client._gemini_client.cache_clear()
    client._openai_client.cache_clear()
    yield server
    server.httpd.shutdown()
    client._gemini_client.cache_clear()
    client._openai_client.cache_clear()
