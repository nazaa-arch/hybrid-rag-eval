# Standard Operating Procedure (SOP): AI Governance & Model Risk Management

## Purpose and Scope
This Standard Operating Procedure (SOP-AI-GOV-04) establishes the governance lifecycle, risk tiering, pre-deployment validation gates, and continuous accountability controls for all Machine Learning (ML) and Generative AI (GenAI) models deployed in production. It applies to internal engineering teams, vendor-hosted foundation model APIs, and Retrieval-Augmented Generation (RAG) assistants handling customer or enterprise data.

## Model Risk Tiering Framework
Every AI system must be registered in the Central Model Inventory and classified into one of three risk tiers before development reaches staging:
- **Tier 1 (High Risk)**: Models directly influencing credit underwriting, fraud blocking, healthcare triage, legal compliance, automated HR screening, or customer-facing GenAI agents with tool-execution authority. Requires independent Model Risk Committee sign-off, red-teaming, fairness audit, and quarterly review.
- **Tier 2 (Medium Risk)**: Internal copilots, document summarization tools, customer support draft assistants with mandatory human-in-the-loop (HITL) approval, and forecasting pipelines. Requires peer engineering review, automated evaluation suite pass, and semi-annual review.
- **Tier 3 (Low Risk)**: Non-critical internal productivity scripts, offline analytics clustering, and experimental prototypes restricted to synthetic or non-PII datasets. Requires lightweight model card documentation and annual check.

## Pre-Deployment Validation Gates
No Tier 1 or Tier 2 model may promote to production without passing four mandatory gates:
1. **Model Card & Data Lineage Gate**: Document training data provenance, known bias limitations, intended use boundaries, out-of-scope uses, and license compliance.
2. **Offline Evaluation & Faithfulness Gate**: GenAI and RAG applications must achieve at least **0.90 Citation Faithfulness**, **>= 0.85 Retrieval Recall@5**, and **>= 0.90 Abstention Accuracy** on the curated golden benchmark suite before release.
3. **Security & Red-Teaming Gate**: Verify prompt injection resistance, jailbreak defense, PII leakage prevention, and strict least-privilege scoping on any agent tool calls.
4. **Fairness & Bias Audit**: For classification or scoring models, disparate impact ratio across protected attributes must remain within the **0.80 to 1.25** four-fifths rule threshold.

## Human-in-the-Loop (HITL) and Override Controls
For high-impact decisions (Tier 1), the system must expose an explicit human override workflow and preserve an immutable decision audit log for a minimum of **5 years**. Generative AI systems must visibly indicate AI-generated content, attach verifiable source citations, and automatically abstain rather than fabricate answers when evidence confidence is below threshold.

## Third-Party Foundation Model & Data Privacy Controls
Sending production customer PII or confidential source code to external public LLM APIs without a signed Data Processing Agreement (DPA) and zero-data-retention guarantee is strictly prohibited. Sensitive workloads must use either:
- Self-hosted local models inside the private VPC (e.g., Ollama / vLLM cluster), or
- Enterprise private endpoints with zero-retention contractual clauses and encrypted transit (TLS 1.3).

## Incident Escalation and Model Decommissioning
Any confirmed hallucination causing financial/regulatory harm, unauthorized PII disclosure, or severe fairness regression must be classified as an **AI-SEV1** incident and reported to the AI Governance Officer and Security Operations Center (SOC) within **1 hour**. Models that fail consecutive quarterly audits or exceed their intended lifecycle must be formally deprecated with a 30-day consumer migration notice and archived weights.
