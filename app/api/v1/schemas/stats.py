from typing import List, Dict
from pydantic import BaseModel

class CountItem(BaseModel):
    label: str
    count: int

class SeriesPoint(BaseModel):
    date: str   # "YYYY-MM-DD"
    count: int

class ChartData(BaseModel):
    labels: List[str]
    datasets: List[Dict]  # ex: [{ "label": "...", "data": [..] }]

class StatsResponse(BaseModel):
    total: int
    distinct_patients: int | None = None
    avg_confidence: float | None = None

    by_birads: List[CountItem]
    by_label: List[CountItem]

    last_30d_series: List[SeriesPoint]
    confidence_histogram: ChartData  # bins 0-10%, 10-20%, ...
