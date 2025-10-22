# app/api/v1/models/enums.py

from enum import Enum

class BiradsCategory(str, Enum):
    BI_RADS_0 = "0 - Examen incomplet"
    BI_RADS_1 = "1 - Normal"
    BI_RADS_2 = "2 - Bénin"
    BI_RADS_3 = "3 - Probablement bénin"
    BI_RADS_4A = "4A - Suspicion faible"
    BI_RADS_4B = "4B - Suspicion intermédiaire"
    BI_RADS_4C = "4C - Suspicion forte"
    BI_RADS_5 = "5 - Évocateur de cancer"
    BI_RADS_6 = "6 - Cancer prouvé"


class BreastDensity(str, Enum):
    A = "A - Presque entièrement graisseux (<25%)"
    B = "B - Densité fibreuse éparse (25-50%)"
    C = "C - Hétérogènement dense (51-75%)"
    D = "D - Extrêmement dense (>75%)"
    
class AnalysisStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"