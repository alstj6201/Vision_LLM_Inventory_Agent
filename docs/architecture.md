# Revised Architecture Notes

## Key Change 1: YOLO Alone Is Not Enough
YOLO can detect products or product-like objects, but SKU-level recognition such as “Pepero Original” vs another Pepero flavor is fragile without training data. The recommended design is:

1. Detect object/slot region.
2. Crop product image.
3. Match crop against a SKU reference gallery using embeddings.
4. Apply planogram constraints to reduce false matches.

## Key Change 2: Treat 1-Item Difference as a Confidence Problem
A one-item mismatch can be theft, CV miss, occlusion, restocking delay, or POS timing issue. Therefore, every count should include confidence.

## Key Change 3: Agents Should Handle Exceptions Only
Most normal cases should not call expensive LLM/VLM agents. Agents are valuable for ambiguous, high-severity, or audit-sensitive cases.

## Key Change 4: Harness Owns Write Permission
The LLM should never place orders directly. It drafts; the harness validates and executes.

## Final Flow
Demand trigger and supply trigger both feed a severity scorer. Normal, high-confidence cases go to LP optimization. Exception cases go through cognition agents and then deterministic harness validation.
