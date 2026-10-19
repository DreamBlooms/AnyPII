#!/usr/bin/env python3
"""Run every examples/*.json against a running AnyPII server."""

import glob
import json
import os
import time
import urllib.request

BASE = os.environ.get("ANYPPI_URL", "http://127.0.0.1:8090")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def post(payload):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/v1/anypii", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    for path in sorted(glob.glob(os.path.join(ROOT, "examples", "*.json"))):
        with open(path, encoding="utf-8") as f:
            req = json.load(f)
        t0 = time.time()
        out = post(req)
        dt = time.time() - t0
        print("=" * 72)
        print(os.path.basename(path), f"{dt:.2f}s", out.get("usage"))
        print(out["redacted_text"])
        for s in out["spans"]:
            print(
                f"  {s['type']:<18} {s['text']!r:<28} "
                f"redacted={s['redacted']} occurrences={len(s['occurrences'])}"
            )
            for occ in s["occurrences"][:3]:
                print(
                    f"      {occ['start']:>3}-{occ['end']:<3} "
                    f"score={occ['score']:.3f}"
                )


if __name__ == "__main__":
    main()
