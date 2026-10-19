#!/usr/bin/env python3
"""
load_test_autoscaling.py
Simulates high-velocity banking transaction traffic against Triton / KServe.
Tests:
1. Dynamic batching efficiency (measuring p50, p95, p99 latency).
2. Autoscaling policies (generating sustained concurrency to trigger HPA).
3. SLA compliance (< 50ms sub-millisecond inference requirement).
Outputs latency benchmarks and markdown report.
"""

import sys
import time
import json
import argparse
import numpy as np
import concurrent.futures
import requests

def send_transaction_request(base_url: str, model_name: str, batch_size: int = 1) -> tuple:
    url = f"{base_url}/v2/models/{model_name}/infer"
    dummy_data = np.random.randn(batch_size, 30).astype(np.float32).tolist()
    payload = {
        "inputs": [
            {
                "name": "transaction_features",
                "shape": [batch_size, 30],
                "datatype": "FP32",
                "data": dummy_data
            }
        ]
    }

    t0 = time.perf_counter()
    try:
        r = requests.post(url, json=payload, timeout=5)
        latency = (time.perf_counter() - t0) * 1000.0
        success = (r.status_code == 200)
        return (success, latency, r.status_code)
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000.0
        return (False, latency, 0)


def run_load_test(base_url: str, model_name: str, concurrency: int = 20, total_requests: int = 500, batch_size: int = 1):
    print("=" * 65)
    print("  TRITON / KSERVE CONCURRENCY & AUTOSCALING BENCHMARK")
    print(f"  Target URL:        {base_url}")
    print(f"  Model Name:        {model_name}")
    print(f"  Concurrency Level: {concurrency} parallel client threads")
    print(f"  Total Requests:    {total_requests}")
    print(f"  Batch Size:        {batch_size}")
    print("=" * 65)

    latencies = []
    success_count = 0
    fail_count = 0

    start_wall_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_transaction_request, base_url, model_name, batch_size)
            for _ in range(total_requests)
        ]

        for future in concurrent.futures.as_completed(futures):
            success, latency, code = future.result()
            latencies.append(latency)
            if success:
                success_count += 1
            else:
                fail_count += 1

    total_time = time.perf_counter() - start_wall_time
    rps = total_requests / total_time if total_time > 0 else 0

    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)

    print("\n--- BENCHMARK RESULTS ---")
    print(f"Total Requests:     {total_requests}")
    print(f"Successful:         {success_count} ({success_count/total_requests:.1%})")
    print(f"Failed:             {fail_count}")
    print(f"Throughput:         {rps:.1f} req/sec")
    print(f"Duration:           {total_time:.2f} seconds")
    print(f"Latency min:        {min_lat:.2f} ms")
    print(f"Latency p50:        {p50:.2f} ms")
    print(f"Latency p90:        {p90:.2f} ms")
    print(f"Latency p95:        {p95:.2f} ms")
    print(f"Latency p99:        {p99:.2f} ms")
    print(f"Latency max:        {max_lat:.2f} ms")
    print(f"SLA Compliance:     {'PASSED (<50ms p95)' if p95 < 50.0 else 'WARN (>50ms p95)'}")

    report = {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": success_count,
        "failed_requests": fail_count,
        "throughput_rps": round(rps, 2),
        "total_time_sec": round(total_time, 2),
        "latency_p50_ms": round(float(p50), 2),
        "latency_p95_ms": round(float(p95), 2),
        "latency_p99_ms": round(float(p99), 2),
        "sla_met": bool(p95 < 50.0)
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Triton Concurrency & HPA Load Tester")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Triton or ALB Endpoint URL")
    parser.add_argument("--model", type=str, default="fraud_detection", help="Model Name")
    parser.add_argument("--concurrency", type=int, default=15, help="Concurrent workers")
    parser.add_argument("--requests", type=int, default=300, help="Total requests")
    parser.add_argument("--batch_size", type=int, default=1, help="Items per request")
    parser.add_argument("--output", type=str, default="artifacts/evaluation/load_test_results.json")
    args = parser.parse_args()

    report = run_load_test(
        base_url=args.url.rstrip("/"),
        model_name=args.model,
        concurrency=args.concurrency,
        total_requests=args.requests,
        batch_size=args.batch_size
    )

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {args.output}")


if __name__ == "__main__":
    main()
