# Synthetic Retail Company Dataset

가상의 편의점 체인 5개 매장, 100개 SKU, 180일치 데이터입니다.

## 핵심 전제
- YOLO/DETR 파인튜닝 없음
- Vision Layer는 GroundingDINO zero-shot + DINOv2/FAISS reference matching으로 가정
- POS, 재고, CV 실측, 발주, 이상사례, RAG case가 동일한 `store_id`, `sku_id`, `date`로 연결됨

## 파일
- `sku_master.csv`: SKU 기준정보, slot_id, reference image placeholder
- `store_master.csv`: 매장 기준정보
- `pos_transactions.csv`: POS 거래 로그
- `inventory_snapshot.csv`: 일별 시스템 재고와 CV 폐점 실측 재고
- `cv_count_log.csv`: 하루 3회 CV SKU count 결과
- `order_history.csv`: 발주 추천/실행 로그
- `promotion_calendar.csv`: 프로모션 일정
- `weather_holiday.csv`: 날씨/휴일 feature
- `anomaly_cases.csv`: 재고손실/오검출/진열변경 이상사례
- `rag_case_library.jsonl`: Causal Context Agent용 RAG 사례문서
- `vision_detections_sample.jsonl`: VLM/Vision-grounded Agent 데모용 탐지 결과 샘플

## 주요 Join Key
- `sku_id`
- `store_id`
- `date`
- `image_id`
- `slot_id`

## 추천 사용법
1. 수요예측: `pos_transactions.csv` + `promotion_calendar.csv` + `weather_holiday.csv`
2. 재고손실 탐지: `inventory_snapshot.csv` 또는 `cv_count_log.csv`
3. RAG: `rag_case_library.jsonl`
4. Agent Harness 검증: `anomaly_cases.csv` + `order_history.csv`
