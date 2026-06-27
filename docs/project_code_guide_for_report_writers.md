# 프로젝트 코드 설명서: 보고서 작성자용

이 문서는 Python 경험이 많지 않은 보고서 작성자가 현재 구현된 Retail Inventory AI Prototype을 이해하고 설명할 수 있도록 작성한 코드 가이드입니다. 내용은 현재 저장소의 실제 코드, SQLite 테이블, 결과 폴더를 확인한 기준입니다. 구현되지 않은 기능은 "현재 구현 단계에서는 제외" 또는 "향후 확장 예정"으로 구분했습니다.

## 1. 프로젝트 개요

이 프로젝트는 무인 매장 또는 소형 리테일 매장에서 발생할 수 있는 재고 불일치 문제를 탐지하고, 안전하게 발주 의사결정을 돕는 AI 기반 재고 지능 시스템 프로토타입입니다.

해결하려는 문제는 단순히 "내일 얼마나 팔릴지"를 예측하는 것이 아닙니다. POS 판매 기록상으로 계산한 가상 재고와 CCTV 기반 실제 선반 재고가 다를 때, 그 차이가 수요 증가인지, 재고 손실인지, 컴퓨터 비전 오류인지, 보충 지연인지 구분하고 안전한 의사결정을 내리는 것이 핵심입니다.

전체 시스템은 다음 흐름으로 구성됩니다.

```text
Reference Gallery / POS / Synthetic DB
-> Demand Forecasting
-> CCTV Vision Counting
-> Shrinkage and Severity Scoring
-> Multi-Agent Reasoning for Exceptions
-> Deterministic Harness Validation
-> MILP Optimizer
-> Decision Card and Dashboard
```

중요한 설계 원칙은 LLM 또는 VLM Agent가 직접 발주를 실행하지 않는다는 점입니다. Agent는 원인 추정과 발주 초안 작성까지만 담당하고, 실제 승인 여부는 Deterministic Harness가 검증합니다.

## 2. 프로젝트 구현 과정

### 2.1 상품 이미지 준비와 Reference Gallery 구축

`products_image/` 폴더에는 SKU별 상품 이미지가 저장되어 있습니다. 폴더명은 `{sku_id}_{product_name}` 형식이고, 이미지 파일명은 `{sku_id}_{height}_s_{angle}.jpg` 형식입니다.

이 이미지는 CCTV detector 검증용이 아니라 SKU embedding database 구축용입니다. 즉, 실제 선반에서 탐지된 crop 이미지가 어떤 SKU와 유사한지 찾기 위한 기준 이미지입니다.

생성 결과:

- `data/embeddings/embeddings.npy`
- `data/embeddings/metadata.csv`
- `data/embeddings/faiss.index`

### 2.2 DINOv2 embedding과 FAISS index 구축

`src/embedding.py`와 `tools/build_embeddings.py`는 Reference Gallery 이미지를 DINOv2 embedding으로 변환합니다. embedding은 L2 normalize되어 FAISS `IndexFlatIP`에 저장됩니다. normalize된 embedding에서는 inner product를 cosine similarity처럼 사용할 수 있습니다.

보고서 문장 예시:

> Reference Gallery의 각 상품 이미지는 DINOv2 embedding으로 변환되며, FAISS index는 CCTV crop 이미지와 가장 유사한 SKU 후보를 빠르게 검색하기 위해 사용된다.

### 2.3 merge_dataset.csv와 SQLite Runtime DB

`data/products/merge_dataset.csv`는 POS 판매량, 가격, 이벤트, SNAP 정보, 상품 정보, 이미지 메타데이터를 포함하는 원천 데이터입니다. 이 파일은 직접 수정하지 않고, synthetic runtime database를 만들 때 참고합니다.

현재 SQLite DB는 다음 경로에 있습니다.

```text
synthetic_retail_company_dataset/retail_inventory.sqlite
```

SQLite를 사용하는 이유는 MVP가 외부 서버 없이 한 파일로 실행될 수 있어야 하기 때문입니다. 보고서/데모 단계에서 재현성이 높고, CSV보다 관계형 테이블 간 연결을 설명하기 쉽습니다.

### 2.4 Demand Forecasting

`src/retail_ai/demand_forecasting.py`는 다음날 SKU별 판매량을 예측합니다. 실제 구현은 Moving Average, Weekday Average, LightGBM을 비교하고 validation 성능이 가장 좋은 모델을 선택합니다. 현재 결과에서는 LightGBM이 최종 선택되었습니다.

주요 feature:

- 요일, 월
- 가격
- lag feature: `lag_1`, `lag_7`, `lag_14`, `lag_28`
- rolling mean/std/max/min
- expanding mean
- days since last sale
- price change rate
- event count
- SNAP 정보

현재 결과 파일:

- `results/demand_forecasting/forecast_predictions.csv`
- `results/demand_forecasting/sku_metrics.csv`
- `results/demand_forecasting/overall_metrics.json`
- `results/demand_forecasting/demand_anomaly_scores.csv`
- `results/demand_forecasting/model_comparison.csv`
- `results/demand_forecasting/improvement_report.md`

### 2.5 Vision Pipeline

Vision pipeline은 다음 순서로 동작합니다.

```text
CCTV shelf image
-> Detector Under Evaluation
-> Bounding boxes
-> Crop
-> DINOv2 embedding
-> FAISS retrieval
-> SKU prediction
-> Count aggregation
```

현재 detector는 고정된 한 모델이 아니라 benchmark 대상입니다. `src/retail_ai/detector_benchmark.py`와 `tools/benchmark_detectors.py`가 YOLO와 OWL-ViT를 같은 이미지에서 비교합니다. 최근 benchmark 결과에서는 `yolo @ confidence 0.2`가 추천 detector로 기록되었습니다.

주의할 점:

- YOLO class label은 최종 SKU로 사용하지 않습니다.
- OWL-ViT label도 최종 SKU로 사용하지 않습니다.
- Detector는 "여기에 상품 후보가 있다"는 bounding box만 제공합니다.
- 최종 SKU는 DINOv2 embedding + FAISS retrieval로 결정합니다.

### 2.6 Shrinkage와 Severity / Triage

`src/retail_ai/severity.py`는 수요 이상과 재고 차이를 결합해 SKU별 위험도를 계산합니다.

주요 계산:

- `shrinkage_score`: expected stock과 CV count 차이 기반
- `demand_anomaly_score`: 실제 판매량과 예측 판매량 차이 기반
- `low_confidence_penalty`: CV count confidence가 낮을 때 penalty
- `theft_suspicion_indicator`: 수요는 정상인데 실물 재고만 크게 부족할 때 표시
- `severity`: 위 요소를 가중합한 0~1 위험도

라우팅:

- `normal`
- `requires_review`
- `freeze_and_alert`

### 2.7 Multi-Agent Cognition Layer

`src/retail_ai/agents.py`는 예외 케이스에 대해서만 Agent reasoning을 실행합니다.

구현된 Agent:

- `CausalContextAgent`: 과거 사례, 수요, 이벤트, 재고 정보를 바탕으로 원인 후보 요약
- `VisionGroundedAgent`: CV count와 detection metadata 기반 시각적 원인 후보 요약
- `OrderDraftingAgent`: 발주 초안 JSON 생성
- `SelfCritiqueAgent`: 초안 검토 및 pass/revise/block 판단

`src/retail_ai/llm_client.py`는 OpenAI API client와 rule-based dry-run client를 제공합니다. OpenAI quota 또는 네트워크 오류가 있으면 데모에서는 dry-run fallback이 사용됩니다.

중요:

> Agent는 발주 API를 호출하지 않으며, order draft와 reasoning만 저장한다.

### 2.8 Deterministic Harness

`src/retail_ai/harness.py`는 Agent draft를 그대로 실행하지 않고 검증하는 안전 계층입니다.

Harness 단계:

1. RBAC Gateway
2. Semantic Validation
3. Stock Auditor
4. Constraint Checker
5. Retry / Self-Healing reason 기록
6. 최종 상태 결정

최종 상태:

- `approved`
- `requires_manual_review`
- `blocked`

Harness는 `order_history`, `harness_results`, `decision_cards`를 업데이트합니다.

### 2.9 PuLP MILP Optimizer

`src/retail_ai/optimizer.py`는 PuLP 기반 MILP optimizer입니다. Greedy가 아니라 정수계획 모델로 발주량을 계산합니다.

두 가지 mode:

- `production`: Harness 결과가 `approved`인 SKU만 자동 최적화
- `simulation`: `approved`와 `requires_manual_review` SKU를 포함해 "사람이 승인한다면" 시나리오 계산

제약조건:

- 예산
- 창고 용량
- 최소 발주량
- 최대 발주량
- pack size 배수
- blocked SKU 제외
- production mode에서 manual review SKU 제외

### 2.10 End-to-End Demo

`src/retail_ai/demo_runner.py`와 `tools/run_end_to_end_demo.py`는 구현된 모듈을 하나의 발표용 pipeline으로 연결합니다.

생성 결과:

- `results/demo/demo_dashboard.html`
- `results/demo/demo_summary.json`
- `results/demo/demo_report.md`
- `results/demo/decision_cards_demo.csv`
- `results/demo/alerts.csv`
- `results/demo/figures/*`
- `results/demo/vision/*`

## 3. 프로젝트 폴더 구조

### `src/`

핵심 Python 모듈이 들어 있습니다. 보고서 관점에서는 "실제 알고리즘과 비즈니스 로직이 있는 곳"입니다.

### `tools/`

명령줄에서 실행하는 스크립트입니다. 예측 학습, triage 실행, agent 실행, harness 실행, optimizer 실행, demo 실행을 담당합니다.

### `tests/`

unit test가 있습니다. 실제 모델 다운로드나 OpenAI API 호출이 필요한 부분은 mock 또는 dry-run으로 검증합니다.

### `data/`

원천 CSV와 embedding 결과가 들어 있습니다. `data/products/merge_dataset.csv`와 `data/embeddings/*`가 중요합니다.

### `products_image/`

SKU별 Reference Gallery 이미지가 저장되어 있습니다. CCTV detector 검증용이 아니라 embedding database 구축용입니다.

### `synthetic_retail_company_dataset/`

SQLite runtime DB와 CSV export가 있습니다. MVP는 이 synthetic dataset으로 전체 pipeline을 재현합니다.

### `results/`

각 단계 실행 결과가 저장됩니다. 보고서 그림, CSV, JSON, dashboard가 이 폴더에 있습니다.

### `docs/`

문서와 아키텍처 설명이 들어갑니다.

## 4. SQLite 테이블 설명

현재 DB 경로:

```text
synthetic_retail_company_dataset/retail_inventory.sqlite
```

주요 테이블:

| 테이블 | 비즈니스 의미 |
|---|---|
| `sku_master` | SKU 기본 정보. 상품명, 카테고리, 공급사, 가격, pack size, 발주 제약을 저장한다. |
| `sku_images` | SKU별 reference image 목록. DINOv2/FAISS embedding database와 연결되는 1:N 이미지 정보다. |
| `inventory_snapshot` | 날짜별 SKU의 opening stock, 판매량, restock, closing stock, expected stock을 저장한다. |
| `cv_count_log` | CCTV 또는 CV count 결과를 저장한다. expected stock, cv_count, count_confidence가 포함된다. |
| `demand_forecasts` | SKU별 예측 판매량을 저장한다. 현재 schema는 `date`, `sku_id`, `forecast_quantity` 중심이다. |
| `anomaly_cases` | severity/triage 결과 중 검토 또는 알림이 필요한 이상 케이스를 저장한다. |
| `order_drafts` | Agent가 만든 발주 초안이다. 실행 결과가 아니라 Harness 검증 전 제안이다. |
| `harness_results` | Harness validation 결과를 저장한다. semantic, stock audit, constraint 결과가 포함된다. |
| `order_history` | Harness 또는 optimizer 결과가 반영된 주문 기록성 테이블이다. 실제 외부 발주 API 호출은 현재 구현 단계에서는 제외다. |
| `decision_cards` | triage, agent, harness, optimizer 결과를 SKU별 decision card 형태로 모은다. |

## 5. 주요 코드 파일 설명

### Demand Forecasting: `src/retail_ai/demand_forecasting.py`

- 목적: SKU별 다음날 판매량 예측
- 입력: `data/products/merge_dataset.csv`
- 처리: feature engineering, time-series split, Moving Average/Weekday Average/LightGBM 비교
- 출력: forecast CSV, metric JSON/CSV, anomaly score, figure
- 보고서 문장: "수요 예측 모듈은 단일 모델에 의존하지 않고 여러 baseline과 LightGBM을 비교해 validation 기준으로 최종 모델을 선택한다."

### Vision Pipeline: `src/retail_ai/vision_counting.py`

- 목적: CCTV 이미지에서 상품 후보 box를 crop하고 SKU를 식별해 수량 집계
- 입력: CCTV image, FAISS index, metadata CSV
- 처리: detector -> crop -> DINOv2 embedding -> FAISS top-k retrieval -> SKU aggregation
- 출력: `CountResult`
- 보고서 문장: "Detector는 SKU 분류기가 아니라 상품 후보 영역 생성기이며, 최종 SKU 식별은 embedding retrieval 단계에서 수행된다."

### Detector Benchmark: `src/retail_ai/detector_benchmark.py`

- 목적: YOLO, OWL-ViT 등 detector 후보를 같은 이미지에서 비교
- 입력: CCTV image list, detector list, confidence thresholds
- 처리: detector별 bbox 생성, JSON/시각화 저장, 추천 detector 선택
- 출력: `results/detector_benchmark/benchmark_summary.csv`, `benchmark_summary.json`
- 보고서 문장: "Detector는 교체 가능한 계층으로 구현되어, 동일한 downstream DINOv2/FAISS pipeline 앞단에서 후보 모델을 비교할 수 있다."

### Severity: `src/retail_ai/severity.py`

- 목적: demand anomaly와 shrinkage를 결합해 위험도 계산
- 입력: demand forecasts, inventory snapshot, cv count log
- 처리: shrinkage score, demand anomaly score, severity score, routing
- 출력: triage results, anomaly cases, decision cards
- 보고서 문장: "Severity module은 수요 이상과 물리 재고 이상을 분리해 해석하고, 자동 처리와 검토 필요 케이스를 구분한다."

### Multi-Agent: `src/retail_ai/agents.py`, `src/retail_ai/llm_client.py`

- 목적: 예외 케이스에 대한 원인 추정과 발주 초안 생성
- 입력: anomaly cases, decision cards, demand forecasts, inventory, CV metadata
- 처리: causal context, vision grounded reasoning, order drafting, self critique
- 출력: agent_outputs JSON, order_drafts, agent_summary
- 보고서 문장: "Agent layer는 reasoning과 draft 생성에 한정되며, 실행 권한은 없다."

### Harness: `src/retail_ai/harness.py`

- 목적: Agent draft를 deterministic rule로 검증
- 입력: order_drafts, sku_master, inventory, forecasts, CV counts, decision cards
- 처리: RBAC, semantic validation, stock audit, constraint check, retry count
- 출력: harness_results, order_history, final decision cards
- 보고서 문장: "Harness는 LLM output을 직접 실행하지 않고, 정형 validation을 통과한 경우에만 다음 단계로 전달한다."

### MILP Optimizer: `src/retail_ai/optimizer.py`

- 목적: 안전 제약 안에서 발주량 최적화
- 입력: sku constraints, forecasts, harness results, order drafts
- 처리: PuLP MILP, integer pack count, budget/storage/min/max/pack constraints
- 출력: optimized_orders CSV, optimizer summary, order_history update
- 보고서 문장: "Optimizer는 재고 부족 위험이 큰 SKU에 우선순위를 주되, 예산과 창고 용량 및 pack size 제약을 동시에 만족하는 정수 발주량을 계산한다."

### End-to-End Demo: `src/retail_ai/demo_runner.py`

- 목적: 전체 pipeline을 발표용으로 한 번에 실행
- 입력: date, CCTV images, SQLite DB, detector mode
- 처리: vision, triage, agents, harness, optimizer, decision card, dashboard 생성
- 출력: demo dashboard, summary, alerts, figures
- 보고서 문장: "End-to-End Demo Runner는 새 알고리즘을 구현하지 않고 기존 모듈을 orchestration하여 발표 가능한 결과물을 생성한다."

## 6. 실행 스크립트 설명

### `tools/rebuild_synthetic_dataset.py`

- 언제 실행: synthetic dataset과 SQLite DB를 다시 만들 때
- 입력: 내부 synthetic generation 로직과 원천 CSV
- 출력: `synthetic_retail_company_dataset/retail_inventory.sqlite`, CSV export

### `tools/train_demand_forecast.py`

- 언제 실행: demand forecasting 결과를 새로 만들 때
- 입력: `data/products/merge_dataset.csv`
- 출력: `results/demand_forecasting/*`, SQLite `demand_forecasts`

### `tools/run_triage.py`

- 언제 실행: demand/CV 결과를 바탕으로 severity routing을 계산할 때
- 입력: SQLite DB, date
- 출력: `results/triage/*`, `anomaly_cases`, `decision_cards`

### `tools/run_agents.py`

- 언제 실행: `requires_review` 또는 `freeze_and_alert` 케이스에 Agent reasoning을 붙일 때
- 입력: SQLite DB, date, provider, dry-run option
- 출력: `results/agents/*`, `order_drafts`, `decision_cards.agent_summary`

### `tools/run_harness.py`

- 언제 실행: Agent order draft를 검증할 때
- 입력: SQLite DB, date
- 출력: `results/harness/*`, `harness_results`, `order_history`, `decision_cards`

### `tools/run_optimizer.py`

- 언제 실행: Harness 결과를 바탕으로 최적 발주량을 계산할 때
- 입력: SQLite DB, date, mode, budget, storage capacity
- 출력: `results/optimizer/*`, `order_history`, `decision_cards.optimized_qty`

### `tools/run_end_to_end_demo.py`

- 언제 실행: 발표/보고서용 전체 데모 결과를 생성할 때
- 입력: date, mode, morning/evening image, SQLite DB, detector option
- 출력: `results/demo/*`

## 7. 데이터 흐름

```text
POS data
-> Demand Forecast
-> demand_forecasts / demand_anomaly_scores

CCTV image
-> Detector Under Evaluation
-> crop images
-> DINOv2 embedding
-> FAISS retrieval
-> SKU count

Forecast + Inventory + CV Count
-> Shrinkage Score
-> Severity Score
-> normal / requires_review / freeze_and_alert

Exception Cases
-> Multi-Agent reasoning
-> order_drafts

Order Drafts
-> Deterministic Harness
-> approved / requires_manual_review / blocked

Harness Results
-> MILP Optimizer
-> optimized order quantities

All Results
-> Decision Cards
-> Dashboard and report figures
```

## 8. 주요 수식의 비즈니스 의미

### Demand Anomaly Score

예측 판매량과 실제 판매량의 차이를 0~1 범위로 표현합니다. 값이 클수록 평소 수요 패턴과 다르게 팔렸다는 뜻입니다.

### Shrinkage Score

기대 재고보다 실제 CV count가 적을 때 커지는 점수입니다. 값이 클수록 실물 재고 부족이 크다는 뜻입니다.

### Severity Score

수요 이상, 재고 부족, 도난 의심, CV confidence를 하나의 위험도로 결합합니다. 이 점수로 정상 처리, 검토 필요, 동결 및 알림을 구분합니다.

### MILP Optimizer

여러 SKU가 동시에 발주 후보일 때 예산, 창고 용량, pack size, 최소/최대 발주량을 지키면서 우선순위가 높은 SKU에 발주량을 배정합니다.

## 9. `results/` 폴더 설명

| 폴더 | 설명 |
|---|---|
| `results/demand_forecasting/` | 예측 결과, 모델 비교, metric, anomaly score, forecast figure |
| `results/triage/` | severity 결과, routing count, severity distribution |
| `results/agents/` | Agent JSON output, order draft CSV, agent summary |
| `results/harness/` | Harness validation result, final decision card CSV/Markdown |
| `results/optimizer/` | MILP optimized order result, optimizer summary, figures |
| `results/detector_benchmark/` | YOLO/OWL-ViT detector 비교 결과와 bbox 시각화 |
| `results/demo/` | End-to-End dashboard, alert, decision card, figures |

## 10. 보고서에 바로 사용할 수 있는 문장

- "본 프로젝트는 POS 기반 가상 재고와 CCTV 기반 실물 재고를 결합하여 무인 매장의 재고 불일치 위험을 탐지하는 프로토타입이다."
- "수요 예측은 Moving Average, Weekday Average, LightGBM을 비교한 뒤 validation 성능을 기준으로 최종 모델을 선택한다."
- "Vision pipeline은 detector를 SKU 분류기로 사용하지 않고, 상품 후보 영역을 생성하는 모듈로만 사용한다."
- "최종 SKU 식별은 CCTV crop image를 DINOv2 embedding으로 변환한 뒤 FAISS reference gallery에서 유사 이미지를 검색하는 방식으로 수행된다."
- "Agent layer는 이상 상황의 원인과 발주 초안을 설명하지만, 직접 발주를 실행하지 않는다."
- "Deterministic Harness는 semantic validation, stock audit, constraint check를 통해 모든 발주 초안을 검증한다."
- "MILP Optimizer는 예산, 창고 용량, 최소/최대 발주량, pack size 제약을 만족하는 정수 발주량을 계산한다."
- "End-to-End Demo Runner는 기존 모듈을 하나의 발표용 pipeline으로 연결하고, dashboard와 decision card를 자동 생성한다."

## 11. 현재 구현 한계

- 현재 구현은 synthetic data 기반 실험이다.
- 실제 발주 API 연동은 현재 구현 단계에서는 제외다.
- Agent는 order draft와 reasoning만 저장하며, 직접 write action을 수행하지 않는다.
- Detector는 현재 CCTV simulation image 기준으로 평가 중이며, 실제 매장 CCTV 환경에서는 추가 검증이 필요하다.
- OpenAI API는 quota나 network 문제 발생 시 dry-run fallback으로 대체된다.
- 실시간 streaming CCTV 처리는 현재 구현 단계에서는 제외다.
- 다중 매장 운영과 외부 POS API 연동은 향후 확장 예정이다.

## 12. 향후 개선 방향

- 실제 CCTV 데이터 확보 및 annotation
- detector 후보 추가 평가: YOLO-World, GroundingDINO, Florence-2, Grounded-SAM
- planogram constraint를 Vision SKU selection에 더 강하게 반영
- 실제 POS API 연동
- 실제 발주 API 연동 단, Harness 이후에만 허용
- 실시간 streaming 처리
- 다중 매장 / 다중 카메라 확장
- Agent reasoning 결과의 human feedback 기반 개선

## 13. 실행 순서 요약

보고서 작성자가 전체 결과를 재현하고 싶을 때는 다음 순서로 이해하면 됩니다.

```bash
python tools/build_embeddings.py
python tools/train_demand_forecast.py
python tools/benchmark_detectors.py --images data/simulation/Morning.png data/simulation/Evening.png --detectors yolo owlvit
python tools/run_triage.py --date latest
python tools/run_agents.py --date latest --dry-run
python tools/run_harness.py --date latest
python tools/run_optimizer.py --date latest --mode simulation
python tools/run_end_to_end_demo.py --date 2026-05-21 --mode simulation --detector auto
```

일반 보고서에서는 모든 단계를 직접 실행하기보다 `results/` 폴더의 산출물을 근거로 설명하면 됩니다.
