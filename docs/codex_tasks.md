# Codex Task Prompts

Use these prompts one by one in Codex.

## Task 1 — Project Skeleton
Create a Python package named `retail_ai` using the repository structure in `AGENTS.md`. Add `pyproject.toml` or `requirements.txt`, a demo script, and empty test files. Do not implement full ML yet.

## Task 2 — Schemas
Implement Pydantic schemas for SKU, ShelfSlot, POSSnapshot, CVCount, DemandForecast, SeverityResult, OrderDraft, HarnessResult, and DecisionCard. Add validation tests.

## Task 3 — Severity Scorer
Implement severity scoring with demand anomaly, shrinkage, theft-suspicion boost, and low-confidence penalty. Add tests for normal, shrinkage, demand spike, and theft-suspicion cases.

## Task 4 — Harness
Implement a deterministic harness that validates:
- SKU exists
- order quantity is positive
- order does not exceed daily budget
- order does not exceed storage capacity
- frozen SKU cannot be ordered
- schema is valid

Add tests for every fail case.

## Task 5 — Optimizer
Implement a simple order optimizer using reorder point logic first. LP can be added later. Inputs: current stock, forecast demand, safety stock, pack size, max capacity, budget. Output: OrderDraft.

## Task 6 — Decision Card
Generate a JSON and markdown decision card containing trigger source, scores, evidence, validation result, and final decision.

## Task 7 — Agent Stubs
Create agent interfaces that return structured JSON only. For now, implement deterministic placeholder logic so tests do not depend on external LLM APIs.

## Task 8 — Demo Scenario
Create a demo with 5 SKUs:
- normal case
- high demand case
- shrinkage case
- theft-suspicion case
- low-confidence CV case

Print decision cards for all cases.
