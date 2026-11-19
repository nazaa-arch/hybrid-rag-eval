# MLOps Production Incident & Drift Response Runbook

## Service Level Objectives (SLOs) and Latency Budgets
All Tier-1 real-time inference and RAG APIs operate under strict availability and latency Service Level Agreements (SLAs):
- **Availability SLO**: 99.95% monthly uptime (maximum 21.9 minutes of unplanned downtime per month).
- **Inference Latency SLA (Tabular / Classic ML)**: p50 <= 35 ms, p95 <= 120 ms, p99 <= 250 ms.
- **RAG Query Latency SLA (Retrieval + Rerank)**: Retrieval stage p95 <= 150 ms; Time-to-First-Token (TTFT) <= 600 ms via Server-Sent Events (SSE) streaming; full end-to-end p95 <= 2,500 ms.
- **Error Rate Budget**: HTTP 5xx error rate must remain below **0.1%** over any rolling 15-minute window.

## Data Drift and Concept Drift Thresholds
Automated drift monitors run hourly on production feature distributions and embedding centroids using Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests:
- **PSI < 0.10 (Green / Nominal)**: No significant population shift; continue standard scheduled monitoring.
- **0.10 <= PSI < 0.25 (Amber / Warning)**: Moderate covariate shift detected. Alert `#mlops-alerts` on Slack, inspect upstream ETL schema changes, and schedule shadow evaluation on freshly labeled data within 48 hours.
- **PSI >= 0.25 (Red / Critical Drift)**: Severe distribution shift detected. Page the on-call ML Engineer immediately (P2 incident), freeze automated feature store backfills, trigger emergency data-quality validation, and prepare model retraining or rollback.
- **Retrieval Drift (RAG)**: Monitor daily mean top-1 rerank score and abstention rate. If abstention rate spikes by more than **15 percentage points** above the 7-day baseline, inspect corpus freshness, broken chunking pipelines, or embedding index corruption.

## Incident Severity Classification and Response Times
- **SEV1 (Critical)**: Production inference outage, silent NaN/corrupted predictions affecting >5% of traffic, or confirmed PII leakage. Acknowledge within **5 minutes**, establish incident bridge within **15 minutes**, executive status update every **30 minutes**.
- **SEV2 (High)**: p95 latency exceeds 2x SLA for >10 minutes, PSI >= 0.25 across top-3 features, or retrieval Recall drops >10% in canary checks. Acknowledge within **15 minutes**, mitigate within **2 hours**.
- **SEV3 (Moderate)**: Non-critical batch feature lag, amber PSI warning, or single-replica pod restart loop with healthy failover. Triage during business hours within **24 hours**.

## Canary Deployment and Rollback Protocol
Every model or retrieval index update uses progressive canary traffic shifting: **5% -> 25% -> 50% -> 100%** with a minimum 30-minute soak period per stage. Automatic rollback is triggered immediately if any of the following guardrails trip:
1. HTTP 5xx rate exceeds **0.5%** for 3 consecutive minutes.
2. p95 latency breaches **1.5x SLA** for 5 consecutive minutes.
3. Grounded citation rate falls below **92%** or abstention rate doubles in canary traffic.

To execute a manual rollback in under **2 minutes**:
1. Pin the serving router back to the last known-good immutable model artifact digest (`model_registry_sha`) and matching vector index snapshot ID.
2. Drain canary pods and verify p95 latency and error rate return to baseline.
3. Preserve failed pod logs, request payload hashes (PII-redacted), and Prometheus metrics snapshots for post-mortem analysis.

## Blameless Post-Mortem Requirements
Every SEV1 and SEV2 incident requires a blameless post-mortem document published within **72 hours** of resolution. The post-mortem must capture root cause timeline, blast radius, detection gap, and at least two concrete preventive action items (such as new regression tests in `eval/qa_dataset.json` or stricter CI schema assertions) tracked to completion within 14 days.
