from typing import List, Dict
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from datetime import date, timedelta

from app.db.database import get_db
from app.api.v1.schemas.stats import CountItem, SeriesPoint, ChartData, StatsResponse
from app.api.v1.models.enums import BiradsCategory
from app.api.v1.models.image_analysis import ImageAnalysis as ImageAnalysisModel

router = APIRouter(prefix="/stats", tags=["Statistiques et rapports"])

LABEL_FROM_BIRADS = {
    BiradsCategory.BI_RADS_1: "Normal",
    BiradsCategory.BI_RADS_2: "Benign",
    BiradsCategory.BI_RADS_5: "Malignant",
}

def _label_from_birads(b: BiradsCategory | None) -> str:
    if b is None:
        return "Unknown"
    return LABEL_FROM_BIRADS.get(b, "Unknown")

@router.get("/summary", response_model=StatsResponse)
def stats_summary(db: Session = Depends(get_db)) -> StatsResponse:
    IA = ImageAnalysisModel
    q_total = select(func.count(IA.id), func.avg(IA.confidence))
    total, avg_conf = db.execute(q_total).one()
    distinct_patients = None
    q_birads = select(IA.result_class, func.count()).group_by(IA.result_class)
    rows_b = db.execute(q_birads).all()
    by_birads = [CountItem(label=(rc.value if rc else "Unknown"), count=int(c)) for rc, c in rows_b]
    agg: Dict[str, int] = {}
    for rc, c in rows_b:
        lbl = _label_from_birads(rc)
        agg[lbl] = agg.get(lbl, 0) + int(c)
    by_label = [CountItem(label=k, count=v) for k, v in sorted(agg.items())]
    today = date.today()
    start = today - timedelta(days=29)
    q_ts = (
        select(func.date(IA.submitted_at).label("d"), func.count(IA.id))
        .where(func.date(IA.submitted_at) >= start.isoformat())
        .group_by("d").order_by("d")
    )
    rows_ts = db.execute(q_ts).all()
    by_date = {str(d): int(c) for d, c in rows_ts}
    series: List[SeriesPoint] = []
    for i in range(30):
        d = start + timedelta(days=i)
        ds = d.isoformat()
        series.append(SeriesPoint(date=ds, count=by_date.get(ds, 0)))
    bins = [(i/10, (i+1)/10) for i in range(10)]
    counts = []
    for i, (lo, hi) in enumerate(bins):
        if i < 9:
            q = select(func.count()).where((IA.confidence >= lo) & (IA.confidence < hi))
        else:
            q = select(func.count()).where((IA.confidence >= lo) & (IA.confidence <= hi))
        counts.append(int(db.execute(q).scalar_one()))
    hist_labels = [f"{int(lo*100)}–{int(hi*100)}%" for lo, hi in bins]
    hist = ChartData(labels=hist_labels, datasets=[{"label": "Confidences", "data": counts}])
    return StatsResponse(
        total=int(total or 0),
        distinct_patients=None,
        avg_confidence=float(avg_conf) if avg_conf is not None else None,
        by_birads=by_birads,
        by_label=by_label,
        last_30d_series=series,
        confidence_histogram=hist,
    )
