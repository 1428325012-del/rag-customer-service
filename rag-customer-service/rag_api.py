import os
import sys
import json
import math
import urllib.request
import urllib.error

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_env():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(base_dir, ".env")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env()

BASE = "https://ark.cn-beijing.volces.com/api/v3"
KEY = os.environ.get("ARK_API_KEY", "")
CHAT_MODEL = os.environ.get("ARK_CHAT_MODEL", "")
EMBED_MODEL = os.environ.get("ARK_EMBED_MODEL", "")
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL", BASE)
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", KEY)


def post_url(url, payload, api_key):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError("HTTP %s: %s" % (e.code, body))


def post(path, payload):
    return post_url(BASE + path, payload, KEY)


def embed(texts):
    url = EMBED_BASE_URL.rstrip("/") + "/embeddings"
    res = post_url(url, {"model": EMBED_MODEL, "input": texts}, EMBED_API_KEY)
    return [d["embedding"] for d in res["data"]]


def chat(messages):
    res = post("/chat/completions", {"model": CHAT_MODEL, "messages": messages})
    return res["choices"][0]["message"]["content"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
