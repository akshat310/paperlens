"""
Fire concurrent requests at a running server and watch its memory.

    python eval/load_test.py http://localhost:8000 --clients 20 --seconds 30

The README argues that THREAD_LIMIT=2 keeps peak memory bounded under load:
a single request's working set is small by construction, and the limiter
means at most two are ever in flight. This script turns that argument into a
measurement. It hammers the cheap endpoints (health, job status, paper list)
and, with --chat, the expensive one, from N concurrent clients, while polling
/api/debug/memory every second, and prints the peak RSS and the latency
distribution.

No model quota is spent unless --chat is passed; --chat asks the same
question repeatedly and *does* spend quota (and will hit the per-user rate
limit, which is itself part of what is being tested -- 429s are counted).

Pure stdlib + httpx. No locust, no k6: the whole thing is a hundred lines and
prints exactly the two numbers the README needs.
"""

import argparse
import statistics
import sys
import threading
import time
import uuid

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", nargs="?", default="http://localhost:8000")
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--chat", action="store_true", help="include chat requests (uses quota)")
    parser.add_argument("--paper", default=None, help="paper id for --chat (default: newest ready)")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base, timeout=60)
    email = f"load-{uuid.uuid4().hex[:8]}@example.com"
    token = client.post(
        "/api/auth/register", json={"email": email, "password": "loadtest1"}
    ).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    paper_id = args.paper
    if args.chat and not paper_id:
        papers = [p for p in client.get("/api/papers", headers=headers).json() if p["status"] == "ready"]
        if not papers:
            sys.exit("--chat needs a ready paper owned by the test user; pass --paper <id>")
        paper_id = papers[0]["id"]

    latencies: dict[str, list[float]] = {"health": [], "list": [], "chat": []}
    statuses: dict[int, int] = {}
    lock = threading.Lock()
    stop = threading.Event()

    def record(kind: str, started: float, status: int) -> None:
        with lock:
            latencies[kind].append((time.perf_counter() - started) * 1000)
            statuses[status] = statuses.get(status, 0) + 1

    def worker(index: int) -> None:
        c = httpx.Client(base_url=args.base, timeout=120, headers=headers)
        n = 0
        while not stop.is_set():
            n += 1
            if args.chat and index % 4 == 0 and n % 3 == 0:
                started = time.perf_counter()
                r = c.post(f"/api/papers/{paper_id}/chat",
                           json={"question": "What is the main contribution of this paper?"})
                record("chat", started, r.status_code)
            elif n % 2 == 0:
                started = time.perf_counter()
                r = c.get("/api/papers")
                record("list", started, r.status_code)
            else:
                started = time.perf_counter()
                r = c.get("/api/health")
                record("health", started, r.status_code)

    def monitor(samples: list[float]) -> None:
        while not stop.is_set():
            try:
                info = client.get("/api/debug/memory").json()
                samples.append(info["rss_mb"])
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)

    baseline = client.get("/api/debug/memory").json()
    print(f"baseline RSS {baseline['rss_mb']:.1f} MB, thread_limit={baseline['thread_limit']}")

    samples: list[float] = []
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.clients)]
    mon = threading.Thread(target=monitor, args=(samples,), daemon=True)
    for t in threads:
        t.start()
    mon.start()
    time.sleep(args.seconds)
    stop.set()
    for t in threads:
        t.join(timeout=130)
    mon.join(timeout=5)

    total = sum(len(v) for v in latencies.values())
    print(f"\n{args.clients} clients for {args.seconds}s: {total} requests "
          f"({total / args.seconds:.0f}/s)")
    print("status codes:", dict(sorted(statuses.items())))
    for kind, values in latencies.items():
        if not values:
            continue
        values.sort()
        p50 = statistics.median(values)
        p95 = values[int(len(values) * 0.95) - 1]
        print(f"  {kind:<7} n={len(values):<5} p50={p50:7.0f} ms  p95={p95:7.0f} ms  max={values[-1]:7.0f} ms")
    if samples:
        print(f"\nRSS during load: min {min(samples):.1f}  mean {statistics.mean(samples):.1f}  "
              f"peak {max(samples):.1f} MB  (limit 512)")


if __name__ == "__main__":
    main()
