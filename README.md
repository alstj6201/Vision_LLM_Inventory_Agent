# Retail Inventory AI Prototype

This project prototypes an unmanned-store inventory intelligence system.

It combines:
- POS-based virtual inventory
- demand forecasting
- CCTV-based physical stock counting
- shrinkage detection
- severity triage
- safe order generation through a deterministic harness

## Why This Architecture
The main business problem is not just forecasting demand. The harder problem is detecting when POS inventory and real shelf inventory diverge, especially when the error is only one item.

Therefore, the system separates:
- probabilistic AI reasoning
- deterministic validation
- auditable execution

## MVP Flow
1. Load POS sales and physical stock counts.
2. Forecast next-day demand.
3. Compare expected stock with physical stock.
4. Compute shrinkage and severity.
5. Route normal cases to optimizer.
6. Route exception cases to agents.
7. Validate all order drafts through harness.
8. Generate decision card.

## Run Demo
```bash
pip install -r requirements.txt
python scripts/run_demo.py
```
