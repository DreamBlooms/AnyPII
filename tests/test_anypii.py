#!/usr/bin/env python3
"""End-to-end tests for AnyPII.

Starts the local server unless ANYPPI_URL is set. Uses only the Python standard
library so it runs without the project's build environment.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build", "anypii")
DEFAULT_MODEL = "/home/dohnuts/models/qwen35-0.8b-gguf/Qwen3.5-0.8B-Q8_0.gguf"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_health(base, timeout=120):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy: {last}")


class Client:
    def __init__(self, base):
        self.base = base

    def post(self, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/v1/anypii",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)


def load(name):
    with open(os.path.join(ROOT, "examples", name), encoding="utf-8") as f:
        return json.load(f)


def flat_spans(result):
    out = []
    for group in result["spans"]:
        for occ in group["occurrences"]:
            out.append(
                (group["type"], occ["start"], occ["end"], group["text"], group["redacted"])
            )
    out.sort(key=lambda x: (x[1], x[2]))
    return out


def assert_spans(result, expected):
    got = flat_spans(result)
    assert got == expected, f"spans mismatch:\n got={got}\n exp={expected}"
    assert result["usage"]["generated_tokens"] == 0


MEDICAL_TEXT = "患者[REDACTED]因肺炎入院。主治医师王芳、责任护士赵敏。予阿莫西林（商品名[REDACTED]，[REDACTED]）抗感染。"
MEDICAL_SPANS = [
    ("patient_name", 2, 5, "李小明", True),
    ("clinician_name", 15, 17, "王芳", False),
    ("clinician_name", 22, 24, "赵敏", False),
    ("drug_generic", 26, 30, "阿莫西林", False),
    ("drug_brand", 34, 37, "阿莫仙", True),
    ("pharma_company", 38, 42, "辉瑞制药", True),
]

CONTACT_TEXT = "联系人手机 [REDACTED]，邮箱 [REDACTED]；工单 TCK-2024-0912 已受理。"
CONTACT_SPANS = [
    ("phone", 6, 17, "13800138000", True),
    ("email", 21, 38, "alice@example.com", True),
    ("ticket_id", 42, 55, "TCK-2024-0912", False),
]


def test_medical(c):
    r = c.post(load("medical.json"))
    assert r["redacted_text"] == MEDICAL_TEXT, r["redacted_text"]
    assert_spans(r, MEDICAL_SPANS)
    assert r["usage"]["decodes"] <= 8, r["usage"]
    print("ok medical")


def test_contact(c):
    r = c.post(load("contact.json"))
    assert r["redacted_text"] == CONTACT_TEXT, r["redacted_text"]
    assert_spans(r, CONTACT_SPANS)
    print("ok contact")


def test_redact_flip(c):
    req = load("medical.json")
    req["types"]["patient_name"]["redact"] = False
    req["types"]["clinician_name"]["redact"] = True
    r = c.post(req)
    expected = "患者李小明因肺炎入院。主治医师[REDACTED]、责任护士[REDACTED]。予阿莫西林（商品名[REDACTED]，[REDACTED]）抗感染。"
    assert r["redacted_text"] == expected, r["redacted_text"]
    by_text = {s["text"]: s for s in r["spans"]}
    assert by_text["李小明"]["redacted"] is False
    assert by_text["王芳"]["redacted"] is True
    assert by_text["赵敏"]["redacted"] is True
    print("ok redact flip")


def test_length_mask(c):
    req = load("contact.json")
    req["mask_style"] = "length"
    r = c.post(req)
    expected = "联系人手机 " + "█" * 11 + "，邮箱 " + "█" * 17 + "；工单 TCK-2024-0912 已受理。"
    assert r["redacted_text"] == expected, r["redacted_text"]
    print("ok length mask")


def test_auto_redact(c):
    req = load("contact.json")
    req["types"]["phone"].pop("redact", None)  # auto
    r = c.post(req)
    by_text = {s["text"]: s for s in r["spans"]}
    assert by_text["13800138000"]["redacted"] is True
    assert "redact_score" in by_text["13800138000"]["occurrences"][0]
    assert by_text["TCK-2024-0912"]["redacted"] is False
    print("ok auto redact")


def test_skip_keep_types(c):
    req = load("medical.json")
    req["skip_keep_types"] = True
    r = c.post(req)
    assert r["redacted_text"] == MEDICAL_TEXT, r["redacted_text"]
    expected = [
        ("patient_name", 2, 5, "李小明", True),
        ("drug_brand", 34, 37, "阿莫仙", True),
        ("pharma_company", 38, 42, "辉瑞制药", True),
    ]
    assert_spans(r, expected)
    print("ok skip_keep_types")


def test_all_keep_skip(c):
    req = load("contact.json")
    for t in req["types"].values():
        t["redact"] = False
    req["skip_keep_types"] = True
    r = c.post(req)
    assert r["redacted_text"] == req["state"]["text"]
    assert r["spans"] == []
    assert r["usage"]["decodes"] == 0
    print("ok all_keep_skip")


def test_typed_mask(c):
    req = load("medical.json")
    req["mask_style"] = "typed"
    r = c.post(req)
    expected = (
        "患者[patient_name:0]因肺炎入院。主治医师王芳、责任护士赵敏。"
        "予阿莫西林（商品名[drug_brand:4]，[pharma_company:5]）抗感染。"
    )
    assert r["redacted_text"] == expected, r["redacted_text"]
    print("ok typed mask")


def test_merge_occurrences(c):
    req = {
        "state": {
            "text": "手机 13800138000，备用 13800138000。",
            "policy": "打码手机号。",
        },
        "types": {"phone": {"description": "手机号码。", "redact": True}},
    }
    r = c.post(req)
    assert len(r["spans"]) == 1, r["spans"]
    g = r["spans"][0]
    assert g["type"] == "phone" and g["text"] == "13800138000"
    assert g["redacted"] is True
    assert [(o["start"], o["end"]) for o in g["occurrences"]] == [(3, 14), (18, 29)]
    assert r["redacted_text"] == "手机 [REDACTED]，备用 [REDACTED]。"
    print("ok merge occurrences")


def test_long_text(c):
    text = ("手机 13800138000。\n") * 900
    req = {
        "state": {"text": text, "policy": "打码手机号。"},
        "types": {"phone": {"description": "手机号码。", "redact": True}},
    }
    assert len(text) <= 16000
    r = c.post(req)
    assert len(r["spans"]) == 1, len(r["spans"])
    assert len(r["spans"][0]["occurrences"]) == 900
    assert r["usage"]["decodes"] == 0
    print("ok long text")


def test_empty_and_errors(c):
    req = load("medical.json")
    req["state"]["text"] = ""
    r = c.post(req)
    assert r["redacted_text"] == ""
    assert r["spans"] == []

    bad = load("medical.json")
    bad["threshold"] = 2.0
    try:
        c.post(bad)
        raise AssertionError("expected HTTP 400 for threshold > 1")
    except urllib.error.HTTPError as e:
        assert e.code == 400, e.code
    print("ok empty/errors")


def main():
    base = os.environ.get("ANYPPI_URL")
    proc = None
    if not base:
        if not os.path.exists(BIN):
            raise SystemExit(f"missing binary {BIN}; run ./build.sh first")
        model = os.environ.get("ANYPPI_MODEL", DEFAULT_MODEL)
        if not os.path.exists(model):
            raise SystemExit(f"missing model {model}; set ANYPPI_MODEL")
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        proc = subprocess.Popen(
            [BIN, "--model", model, "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wait_health(base)

    c = Client(base)
    try:
        test_medical(c)
        test_contact(c)
        test_redact_flip(c)
        test_length_mask(c)
        test_auto_redact(c)
        test_typed_mask(c)
        test_merge_occurrences(c)
        test_long_text(c)
        test_skip_keep_types(c)
        test_all_keep_skip(c)
        test_empty_and_errors(c)
        print("all tests passed")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
