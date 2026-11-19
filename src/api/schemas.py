
"""
Pydantic schemas for the Diabetes prediction API.

Defines the input model (DiabetesFeatures) and the output model (PredictionResponse)
used by FastAPI to validate requests and serialise responses automatically.
"""

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Feature columns — ordered list that matches train_features_v1.csv (no target)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "bp",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "bmi_age_interaction",
    "bp_bmi_interaction",
    "s1_s2_ratio",
]

# ---------------------------------------------------------------------------
# Schemas — request body (input) and response body (output)
# ---------------------------------------------------------------------------


class DiabetesFeatures(BaseModel):
    """Input: the 13 engineered feature values for one patient (excluding target)."""

    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    bmi_age_interaction: float
    bp_bmi_interaction: float
    s1_s2_ratio: float


class PredictionResponse(BaseModel):
    """Output: predicted diabetes disease progression and the model version used."""

    prediction: float
    model_version: str
