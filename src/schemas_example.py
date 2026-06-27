from pydantic import BaseModel, Field
from typing import Literal, list

class SKU(BaseModel):
    sku_id: str
    name: str
    pack_size: int = Field(gt=0)
    unit_cost: float = Field(gt=0)
    max_capacity: int = Field(gt=0)

class CVCount(BaseModel):
    sku_id: str
    shelf_slot_id: str
    count: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)

class SeverityResult(BaseModel):
    sku_id: str
    demand_anomaly_score: float = Field(ge=0, le=1)
    shrinkage_score: float = Field(ge=0, le=1)
    low_confidence_penalty: float = Field(ge=0, le=1)
    theft_suspicion: bool
    severity: float = Field(ge=0)
    route: Literal["auto_order", "review", "freeze_and_alert"]
