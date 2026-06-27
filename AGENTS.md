# AGENTS.md — Retail Inventory AI Project

## Project Goal

Build a prototype of an AI-powered inventory intelligence system for unmanned retail stores.

The system integrates:

- POS-based virtual inventory tracking
- Demand forecasting
- CCTV-based physical inventory counting
- Shrinkage detection
- Safe automatic replenishment

The highest priorities are:

- Accurate counting (including 1-item differences)
- Explainable decisions
- Deterministic safety
- Modular architecture

The MVP must be runnable entirely using synthetic data.

---

# Core Principles

## Safety First

LLM/VLM agents are reasoning components only.

They may:

- retrieve context
- summarize evidence
- explain anomalies
- draft order proposals

They must NEVER:

- execute write actions
- modify inventory
- call ordering APIs

Only the Deterministic Harness owns write permissions.

---

## Harness is Mandatory

Every order proposal—including completely normal automatic replenishment—must pass through the Deterministic Harness.

There are no exceptions.

Pipeline:

Demand/CV
      │
      ▼
LP Optimizer
      │
      ▼
Deterministic Harness
      │
 ┌────┴────┐
 │ Pass    │
 ▼         ▼
Execute   Block

If an anomaly exists, the Cognition Layer is inserted BEFORE the Harness.

Demand/CV
      │
      ▼
Agent Layer
      │
      ▼
LP Optimizer
      │
      ▼
Harness
      │
      ▼
Execute / Freeze

---

# System Architecture

## 1. Demand Trigger

Inputs

- POS sales
- Store calendar
- Promotion
- Weather
- Holidays

Outputs

- virtual_stock
- demand_forecast
- demand_anomaly_score

Implementation

Start simple.

Preferred order:

1. Moving Average
2. Weekday Average
3. LightGBM

Prophet is optional.

---

## 2. Supply Trigger

Inputs

- CCTV shelf snapshots
- Planogram
- POS sales since previous snapshot

Outputs

- cv_count
- count_confidence
- shrinkage_score

Vision Pipeline

Object Detection / Segmentation
        │
        ▼
Object Crop
        │
        ▼
DINOv2 / CLIP Embedding
        │
        ▼
FAISS Retrieval
        │
        ▼
Planogram Constraint
        │
        ▼
SKU + Confidence
        │
        ▼
Inventory Count

Notes

- Never use YOLO class IDs as final SKU labels.
- Count confidence is as important as raw count.
- Candidate SKU search must be restricted using the planogram.

---

## 3. Severity

severity =
w1*demand_anomaly
+w2*shrinkage
+w3*theft_indicator
+w4*low_confidence

Where

theft_indicator

=

Demand normal

AND

Physical inventory abnormal

---

## 4. Decision Routing

Low severity

↓

LP Optimizer

↓

Harness

↓

Automatic Order

Medium severity

↓

Decision Card

↓

Harness

↓

Human Review

High severity

↓

Freeze SKU

↓

Decision Card

↓

Human Review

↓

No automatic order

---

## 5. Cognition Layer

Only executed for abnormal cases.

### CausalContextAgent

Searches

- historical incidents
- weather
- promotions
- holidays
- events

using RAG.

---

### VisionGroundedAgent

Analyzes

- crop images
- masks
- shelf context
- confidence

Possible causes

- theft
- misplaced product
- occlusion
- CV error
- restocking delay
- planogram change

---

### OrderDraftingAgent

Creates

structured JSON

including

- evidence
- uncertainty
- recommendation

---

### SelfCritiqueAgent

Verifies

- reasoning
- evidence
- consistency

before sending to the Harness.

---

# Deterministic Harness

The Harness owns every write permission.

Checks

## 1. RBAC

Agents

Read Only

Harness

Read + Write

---

## 2. Semantic Validation

Pydantic validation

Check

- SKU
- supplier
- pack size
- schema
- required fields

---

## 3. Stock Auditor

Compare

POS virtual inventory

vs

Latest CV count

vs

Sales since snapshot

vs

Confidence

---

## 4. Constraint Checker

Verify

- budget
- storage capacity
- min order
- max order
- supplier lead time
- freeze flag

---

## 5. Retry

Maximum

3

retries.

Every retry must include evidence-based feedback.

Failed cases are stored for future RAG retrieval.

---

# Database Design

The MVP uses SQLite.

No external database is required.

Generate everything using synthetic data.

## Required Tables

stores

suppliers

skus

planogram_slots

pos_sales

stock_snapshots

cv_detections

demand_forecasts

incidents

order_drafts

orders

harness_results

decision_cards

---

## Synthetic Database

Create

scripts/generate_synthetic_db.py

Generate

- 1 demo store
- 30–50 SKUs
- 3 suppliers
- fixed planogram
- 90 days of POS history
- daily inventory snapshots

Inject anomalies

- demand spike
- shrinkage
- theft-like event
- low-confidence CV
- planogram mismatch
- restocking delay

Output

data/retail_ai_demo.sqlite

---

# Repository Layer

Never scatter SQL throughout the project.

Create

src/retail_ai/db.py

Responsibilities

- database connection
- repository classes
- CRUD methods

Return Pydantic models whenever possible.

---

# Decision Card

Every execution must generate

- trigger source
- SKU
- shelf slot
- virtual stock
- CV count
- forecast
- anomaly score
- shrinkage score
- severity
- agent summary
- harness result
- retry count
- final decision
- execution cost

---

# MVP Order

Implement in this exact order.

1.
Database schema

2.
Synthetic database generator

3.
Pydantic models

4.
Demand forecasting

5.
CV count simulator

6.
Shrinkage scorer

7.
Severity scorer

8.
LP optimizer

9.
Harness

10.
Decision Card

11.
Agent stubs

12.
Replace CV simulator with real vision pipeline

---

# Non Goals

Do NOT build

- real ordering API
- real payment integration
- multi-camera reconstruction
- real-time streaming
- theft accusation

---

# Coding Rules

Python 3.11+

Use

- Pydantic
- SQLAlchemy
- SQLite

Agent outputs

JSON only.

Every module requires unit tests.

No secrets inside source code.

No write action may bypass the Harness.

---

# Repository

retail_ai/

AGENTS.md

README.md

requirements.txt

data/

docs/

scripts/

tests/

src/

retail_ai/

db.py

schemas.py

synthetic_data.py

demand.py

vision_counting.py

shrinkage.py

severity.py

optimizer.py

agents.py

harness.py

decision_card.py
