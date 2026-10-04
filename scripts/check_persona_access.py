"""Verify persona extraction credentials and the pinned judge before GPU work."""

from pathlib import Path
import json

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("mooody-persona-access-check")
image = modal.Image.debian_slim(python_version="3.12").add_local_file(
    ROOT / "data/persona_traits/protocol.json", "/protocol.json", copy=True
)


@app.function(
    image=image, cpu=0.25, memory=256, timeout=180,
    secrets=[
        modal.Secret.from_name("mooody-hf", required_keys=["HF_TOKEN"]),
        modal.Secret.from_name("mooody-openrouter", required_keys=["OPENROUTER_API_KEY"]),
    ],
)
def verify():
    import os
    import re
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    protocol = json.loads(Path("/protocol.json").read_text())

    def request(url, token=None, payload=None):
        headers = {"Content-Type": "application/json", "X-OpenRouter-Metadata": "enabled"}
        if token:
            headers["Authorization"] = "Bearer " + token
        body = None if payload is None else json.dumps(payload).encode()
        try:
            with urlopen(Request(url, data=body, headers=headers), timeout=60) as response:
                return json.load(response)
        except HTTPError as error:
            try:
                detail = str(json.loads(error.read()).get("error", {}).get("message", ""))
            except (ValueError, AttributeError):
                detail = ""
            if token:
                detail = detail.replace(token, "[redacted]")
            detail = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", detail)[:200]
            raise RuntimeError(f"Credential/provider preflight returned HTTP {error.code}: {detail}") from None

    model = protocol["checkpoint"]
    info = request(
        f"https://huggingface.co/api/models/{model['model_id']}/revision/{model['revision']}",
        os.environ["HF_TOKEN"],
    )
    if info.get("sha") != model["revision"]:
        raise RuntimeError("Hugging Face checkpoint revision did not match the protocol")
    judge = protocol["judging"]
    request("https://openrouter.ai/api/v1/key", os.environ["OPENROUTER_API_KEY"])
    catalogue = request("https://openrouter.ai/api/v1/models")
    matches = [entry for entry in catalogue["data"] if entry["id"] == judge["model"]]
    if len(matches) != 1:
        raise RuntimeError("Pinned judge is missing from the current OpenRouter catalogue")
    payload = {
        "model": judge["model"],
        "messages": [{"role": "user", "content": "Return only the integer 0, with no other text."}],
        "provider": judge["provider"],
        "reasoning": judge["reasoning"],
        "max_completion_tokens": judge["max_completion_tokens"],
        "stream": False,
    }
    scored = request(
        "https://openrouter.ai/api/v1/chat/completions",
        os.environ["OPENROUTER_API_KEY"], payload,
    )
    choices = scored.get("choices", [])
    content = choices[0].get("message", {}).get("content") if choices else None
    if (not isinstance(content, str) or not re.fullmatch(r"(?:100|[1-9]?[0-9])", content.strip())
            or choices[0].get("finish_reason") != "stop"):
        raise RuntimeError("Judge preflight did not return a complete bare integer score")
    entry = matches[0]
    return {
        "passed": True,
        "checkpoint": {"model_id": model["model_id"], "revision": info["sha"]},
        "judge": {
            "requested_model": judge["model"], "catalogue_slug": entry.get("canonical_slug"),
            "actual_model": scored.get("model"), "provider": scored.get("provider"),
            "finish_reason": choices[0].get("finish_reason"), "usage": scored.get("usage"),
            "score": int(content.strip()), "pricing": entry.get("pricing"),
        },
    }


@app.local_entrypoint()
def main():
    receipt = verify.remote()
    output = ROOT / "artifacts/persona/preflight_access.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
