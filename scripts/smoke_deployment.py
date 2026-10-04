"""Exercise real deployed streaming without logging server credentials."""
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import socket
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument("origin")
parser.add_argument("--private", action="store_true")
parser.add_argument("--resolve-ip", help="Use a verified authoritative address while preserving HTTPS hostname and certificate validation.")
parser.add_argument("--checks", choices=("full", "cancellation", "demo"), default="full")
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
release = json.loads(Path("deployment/persona_release.json").read_text())
traits = json.loads(Path("data/persona_traits/protocol.json").read_text())["trait_order"]
assert status == 200 and config["model_id"] == "demivoleegaston/Qwen3.5-9B-mooody"
assert config["revision"] == "705afd95bced3ac0424d7e68b1299d8fcdffb858"
assert config["system_prompt_present"] is False
assert config["mood_conditioning"] == "vectors_with_prompt_assistance"
assert config["mood_vectors_available"] is True and config["thinking"] is False
assert config["steering_available"] is True
assert config["mood_vectors_source"] == "persona_vectors"
assert config["mood_vectors_repo_id"] == release["repo_id"]
assert config["mood_vectors_revision"] == release["revision"]
assert config["axes"] == traits
assert config["steering_method"] == "paper_incremental_all_layers"
assert config["steering_incremental_definition"] == "raw_layer_vector_minus_previous_layer_vector"
assert config["steering_first_layer_previous_vector"] == "zero"
assert config["mood_vectors_published_inference_method"] == "direct_raw_all_layers"
assert config["steering_token_scope"] == "final_formatted_prompt_token_then_generated_content_tokens"
assert config["mood_vectors_validated"] is False
assert config["mood_coefficients"] == [-0.25, -0.125, 0, 0.125, 0.25]
positive_coefficient = config["mood_coefficients"][3]
assert get("/")[0] == 200
asset_hashes = {}
for name in ("app.js", "model.js"):
    asset_status, _, asset_body = get(f"/src/{name}")
    expected_body = (Path("dist/src") / name).read_bytes()
    assert asset_status == 200 and asset_body == expected_body, f"Published {name} differs from the release build"
    asset_hashes[name] = hashlib.sha256(asset_body).hexdigest()

protected_origin = "https://enricobottazzi--mooody-web.modal.run"
try:
    with urlopen(Request(protected_origin + "/api/config", headers={"User-Agent": headers["User-Agent"]}), timeout=90) as response:
        origin_status = response.status
except HTTPError as error:
    origin_status = error.code
assert origin_status == 403, f"Unauthenticated origin returned {origin_status}"

def chat(messages, limit=64, cancel=False, mood=None, guard_repetition=False):
    started = time.monotonic()
    coefficients = [0] * 6 if mood is None else list(mood)
    payload = {"messages": messages, "mood": coefficients, "max_new_tokens": limit}
    request = Request(args.origin + "/api/chat", data=json.dumps(payload).encode(), headers=headers)
    tokens, events, first_token = [], [], None
    try:
        response = urlopen(request, timeout=90)
    except HTTPError as error:
        try:
            failure = json.loads(error.read())
        except (ValueError, UnicodeDecodeError):
            failure = {}
        print(json.dumps({"http_status": error.code, "code": failure.get("code"), "message": failure.get("message")}), flush=True)
        raise
    with response:
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
                    if guard_repetition:
                        words = re.findall(r"\w+", "".join(tokens).casefold())
                        longest = current = 0
                        previous = None
                        for word in words:
                            current = current + 1 if word == previous else 1
                            longest = max(longest, current)
                            previous = word
                        if longest >= 8:
                            raise RuntimeError("Live steered response entered a repeated-word loop")
                    if cancel:
                        break
                if event == "done":
                    break
                event, data = "", []
    if not cancel:
        assert events and events[-1]["event"] == "done"
        assert events[-1]["moods_applied"] is bool(any(coefficients))
        assert events[-1]["mood_vectors_source"] == "persona_vectors"
        assert events[-1]["mood_vectors_integrity_verified"] is True
        assert events[-1]["mood_vectors_repo_id"] == release["repo_id"]
        assert events[-1]["mood_vectors_revision"] == release["revision"]
        assert events[-1]["steering_method"] == "paper_incremental_all_layers"
        assert events[-1]["mood_coefficients_applied"] == coefficients
        assert events[-1]["system_prompt_present"] is False
        assert events[-1]["mood_conditioning"] == "vectors_with_prompt_assistance"
        assert events[-1]["mood_prompt_assistance_applied"] is bool(any(coefficients))
    assert tokens
    result = {"response": "".join(tokens), "first_token_seconds": first_token,
              "elapsed_seconds": time.monotonic() - started,
              "cancelled_by_client": cancel, "mood_coefficients": coefficients, "events": events}
    summary = {key: value for key, value in result.items() if key not in {"events", "response"}}
    if guard_repetition:
        summary.update(response_sha256=hashlib.sha256(result["response"].encode()).hexdigest(),
                       max_repeated_word_run=longest)
        result["max_repeated_word_run"] = longest
    else:
        summary["response"] = result["response"]
    print(json.dumps(summary), flush=True)
    return result

checks = {"origin": args.origin, "configuration": config, "streaming": [],
          "published_asset_sha256": asset_hashes,
          "protected_origin": protected_origin, "unauthenticated_origin_status": origin_status,
          "authoritative_address_override": args.resolve_ip, "checks": args.checks}
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
checks["passed"] = False
output.write_text(json.dumps(checks, indent=2) + "\n")
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
        [{"role": "user", "content": "what's on your mind"}],
        limit=config["max_output_tokens"],
        mood=[0, 0, 0, config["mood_coefficients"][-1], 0, 0],
        guard_repetition=True,
    )
elif args.checks == "demo":
    checks["mood_comparison"] = {}
    for trait in ("depression", "euphoria", "curiosity", "all_traits"):
        mood = [config["mood_coefficients"][-1]] * 6 if trait == "all_traits" else [0] * 6
        if trait != "all_traits":
            mood[traits.index(trait)] = config["mood_coefficients"][-1]
        checks["mood_comparison"][trait] = chat(
            [{"role": "user", "content": "what's your feeling today ?"}],
            limit=192, mood=mood, guard_repetition=True,
        )
    assert checks["mood_comparison"]["depression"]["response"] != checks["mood_comparison"]["euphoria"]["response"]
# Exercise cancellation with the real curiosity vector, then verify cleanup
# through an independent neutral request.
cancelled = chat([{"role": "user", "content": "Write a detailed explanation of how sunlight creates a rainbow, with several paragraphs and examples."}],
                 limit=512, cancel=True, mood=[0, positive_coefficient, 0, 0, 0, 0])
checks["cancellation"] = cancelled
time.sleep(2)
checks["post_cancel_reply"] = chat([{"role": "user", "content": "Reply with the single word ready."}])
assert "ready" in checks["post_cancel_reply"]["response"].lower()
checks["passed"] = True
output.write_text(json.dumps(checks, indent=2) + "\n")
print(f"Live {args.checks} checks passed: {output}", flush=True)
