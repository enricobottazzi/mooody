"""Exercise real deployed streaming without logging server credentials."""
import argparse
import ipaddress
import json
from pathlib import Path
import socket
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument("origin")
parser.add_argument("--private", action="store_true")
parser.add_argument("--resolve-ip", help="Use a verified authoritative address while preserving HTTPS hostname and certificate validation.")
parser.add_argument("--checks", choices=("full", "cancellation"), default="full")
parser.add_argument("--output", default="artifacts/deployment/live_smoke.json")
args = parser.parse_args()
if args.resolve_ip:
    ipaddress.ip_address(args.resolve_ip)
    hostname = urlsplit(args.origin).hostname
    system_getaddrinfo = socket.getaddrinfo
    def authoritative_getaddrinfo(host, port, *positional, **keywords):
        return system_getaddrinfo(args.resolve_ip if host == hostname else host, port, *positional, **keywords)
    socket.getaddrinfo = authoritative_getaddrinfo
headers = {"Content-Type": "application/json", "User-Agent": "mooody-deployment-check/1.0"}
if args.private:
    credentials = json.loads(Path("artifacts/deployment/worker_secrets.json").read_text())
    headers["x-mooody-proxy-token"] = credentials["MOOODY_PROXY_TOKEN"]
    headers["x-mooody-client-ip"] = "192.0.2.1"

def get(path):
    with urlopen(Request(args.origin + path, headers=headers), timeout=90) as response:
        return response.status, response.headers.get("Content-Type"), response.read()

status, content_type, body = get("/api/config")
config = json.loads(body)
assert status == 200 and config["model_id"] == "demivoleegaston/Qwen3.5-9B-mooody"
assert config["revision"] == "705afd95bced3ac0424d7e68b1299d8fcdffb858"
assert config["mood_vectors_available"] is True and config["thinking"] is False
assert config["steering_available"] is True
assert config["mood_vectors_source"] == "random_placeholder"
assert config["mood_vectors_validated"] is False
assert get("/")[0] == 200
assert get("/src/app.js")[0] == 200

def chat(messages, limit=64, cancel=False, mood=None):
    started = time.monotonic()
    coefficients = [0] * 6 if mood is None else list(mood)
    payload = {"messages": messages, "mood": coefficients, "max_new_tokens": limit}
    request = Request(args.origin + "/api/chat", data=json.dumps(payload).encode(), headers=headers)
    tokens, events, first_token = [], [], None
    with urlopen(request, timeout=90) as response:
        assert response.status == 200
        assert response.headers.get("Content-Type", "").startswith("text/event-stream")
        event, data = "", []
        for raw_line in response:
            line = raw_line.decode().rstrip("\r\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
            elif not line and data:
                value = json.loads("\n".join(data))
                events.append({"event": event, **value})
                if event == "status":
                    print(value["message"], flush=True)
                if event == "error":
                    raise RuntimeError(f"Live inference failed: {value}")
                if event == "token":
                    first_token = first_token or time.monotonic() - started
                    tokens.append(value["text"])
                    if cancel:
                        break
                if event == "done":
                    break
                event, data = "", []
    if not cancel:
        assert events and events[-1]["event"] == "done"
        assert events[-1]["moods_applied"] is bool(any(coefficients))
        assert events[-1]["mood_vectors_source"] == "random_placeholder"
    assert tokens
    result = {"response": "".join(tokens), "first_token_seconds": first_token,
              "elapsed_seconds": time.monotonic() - started,
              "cancelled_by_client": cancel, "mood_coefficients": coefficients, "events": events}
    print(json.dumps({key: value for key, value in result.items() if key != "events"}), flush=True)
    return result

checks = {"origin": args.origin, "configuration": config, "streaming": [],
          "authoritative_address_override": args.resolve_ip, "checks": args.checks}
if args.checks == "full":
    ready = chat([{"role": "user", "content": "Reply with the single word ready."}])
    assert "ready" in ready["response"].lower()
    checks["streaming"].append(ready)
    remember = [{"role": "user", "content": "Remember this word: meadow. Reply with the word only."}]
    first = chat(remember)
    assert "meadow" in first["response"].lower()
    checks["streaming"].append(first)
    history = remember + [{"role": "assistant", "content": first["response"]},
                          {"role": "user", "content": "What word did I give you? Reply with the word only."}]
    followup = chat(history)
    assert "meadow" in followup["response"].lower()
    checks["streaming"].append(followup)
    checks["mood_steering"] = chat(
        [{"role": "user", "content": "Describe a sunrise in one sentence."}],
        mood=[2, -2, 1, 0, -1, 2],
    )
# Random placeholder directions have no semantic guarantees. Use the same
# streamable fixture as the nonzero check rather than relying on a counting task.
cancelled = chat([{"role": "user", "content": "Describe a sunrise in one sentence."}],
                 limit=512, cancel=True, mood=[2, -2, 1, 0, -1, 2])
checks["cancellation"] = cancelled
time.sleep(2)
checks["post_cancel_reply"] = chat([{"role": "user", "content": "Reply with the single word ready."}])
assert "ready" in checks["post_cancel_reply"]["response"].lower()
checks["passed"] = True
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(checks, indent=2) + "\n")
print(f"Live {args.checks} checks passed: {output}", flush=True)
