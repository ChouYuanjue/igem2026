from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "NJU-experimental-touch-cascade/1.0 (research audit)"


def get_bytes(url: str, params: dict | None = None, timeout: int = 60, retries: int = 7) -> tuple[bytes, dict]:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), dict(r.headers.items())
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except Exception as e:
            last = e
        time.sleep(min(30.0, 0.75 * (2 ** i)))
    raise last


def get_json(url: str, params: dict | None = None, timeout: int = 60, retries: int = 7):
    body, headers = get_bytes(url, params=params, timeout=timeout, retries=retries)
    return json.loads(body.decode("utf-8")), headers


def get_text(url: str, params: dict | None = None, timeout: int = 60, retries: int = 7) -> tuple[str, dict]:
    body, headers = get_bytes(url, params=params, timeout=timeout, retries=retries)
    return body.decode("utf-8"), headers
