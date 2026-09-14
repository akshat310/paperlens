"""
Live smoke test against a running server. Uses real quota; not part of pytest.

    python eval/e2e_smoke.py http://localhost:8000 [arxiv-url]

Registers a throwaway user, ingests a paper from arXiv, waits for the report,
asks one question with citation quotes, downloads the PDF and prints the usage
ledger for the paper. Exit code 0 means every step worked.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import uuid

import httpx

base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
url = sys.argv[2] if len(sys.argv) > 2 else "https://arxiv.org/abs/1706.03762"
client = httpx.Client(base_url=base, timeout=120)

email = f"smoke-{uuid.uuid4().hex[:8]}@example.com"
token = client.post("/api/auth/register", json={"email": email, "password": "smoketest1"}).json()["access_token"]
h = {"Authorization": f"Bearer {token}"}
print("registered", email)

paper = client.post("/api/papers/from-url", json={"url": url}, headers=h)
paper.raise_for_status()
paper = paper.json()
pid = paper["id"]
print("paper", pid, "|", paper["title"], "|", (paper["authors"] or "")[:40])

started = time.time()
while True:
    job = client.get(f"/api/papers/{pid}/job", headers=h).json()
    print(f"  [{time.time() - started:5.0f}s] {job['stage']:<12} {job['message'][:60]}")
    if job["status"] in ("done", "failed"):
        break
    time.sleep(5)
if job["status"] == "failed":
    sys.exit(f"job failed: {job['error_message']}")

paper = client.get(f"/api/papers/{pid}", headers=h).json()
print("ready:", paper["num_pages"], "pages,", paper["num_chunks"], "chunks")

report = client.get(f"/api/papers/{pid}/report", headers=h).json()
print("report:", len(report["contributions"]), "contributions,", len(report["key_results"]), "results")

answer = client.post(
    f"/api/papers/{pid}/chat",
    json={"question": "What optimizer and learning rate schedule were used for training?"},
    headers=h,
)
answer.raise_for_status()
answer = answer.json()
print("answer:", answer["answer"][:300].replace("\n", " "))
print("grounded:", answer["grounded"], "| citations:", len(answer["citations"]),
      "| verified quotes:", sum(c["verified"] for c in answer["citations"]))
for c in answer["citations"]:
    print("   ", c["label"], "|", ("VERIFIED " if c["verified"] else "") + c["snippet"][:80])

sections = client.get(f"/api/papers/{pid}/sections", headers=h).json()
results = next((s for s in sections if s["kind"] == "results"), sections[-1])
scoped = client.post(
    f"/api/papers/{pid}/chat",
    json={"question": "What BLEU score is reported?", "section_id": results["id"]},
    headers=h,
).json()
print(f"scoped to '{results['name']}':", scoped["answer"][:160].replace("\n", " "))

pdf = client.get(f"/api/papers/{pid}/file", headers=h)
print("file:", pdf.status_code, pdf.headers.get("content-type"), len(pdf.content), "bytes")
assert pdf.content[:5] == b"%PDF-"

usage = client.get(f"/api/papers/{pid}/usage", headers=h).json()
print("usage:", usage)
print("OK")
