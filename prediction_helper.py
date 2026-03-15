import pandas as pd
import joblib
import os

# ── Artifact paths (absolute — works from any working directory) ───────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_ART  = os.path.join(_BASE, "artifacts")

model_young  = joblib.load(os.path.join(_ART, "model_young.joblib"))
model_rest   = joblib.load(os.path.join(_ART, "model_rest.joblib"))
scaler_young = joblib.load(os.path.join(_ART, "scaler_young.joblib"))
scaler_rest  = joblib.load(os.path.join(_ART, "scaler_rest.joblib"))


def calculate_normalized_risk(medical_history: str) -> float:
    """
    Convert a medical-history string into a 0-1 normalised risk score.

    Scoring table
    -------------
    No Disease / None     →  0
    Thyroid               →  5
    Diabetes              →  6
    High blood pressure   →  6
    Heart disease         →  8
    Max possible          → 14  (heart disease + diabetes or HBP)
    """
    risk_scores = {
        "diabetes":            6,
        "heart disease":       8,
        "high blood pressure": 6,
        "thyroid":             5,
        "no disease":          0,
        "none":                0,
    }
    diseases    = medical_history.lower().split(" & ")
    total_score = sum(risk_scores.get(d.strip(), 0) for d in diseases)
    return total_score / 14          # normalise to [0, 1]


def preprocess_input(input_dict: dict) -> pd.DataFrame:
    """
    Map the raw UI input dictionary to the feature DataFrame
    expected by the trained scikit-learn models.
    """
    expected_columns = [
        "age", "number_of_dependants", "income_lakhs",
        "insurance_plan", "genetical_risk", "normalized_risk_score",
        "gender_Male",
        "region_Northwest", "region_Southeast", "region_Southwest",
        "marital_status_Unmarried",
        "bmi_category_Obesity", "bmi_category_Overweight", "bmi_category_Underweight",
        "smoking_status_Occasional", "smoking_status_Regular",
        "employment_status_Salaried", "employment_status_Self-Employed",
    ]

    plan_enc = {"Bronze": 1, "Silver": 2, "Gold": 3}
    df       = pd.DataFrame(0, columns=expected_columns, index=[0])

    for key, value in input_dict.items():
        if key == "Gender" and value == "Male":
            df["gender_Male"] = 1
        elif key == "Region":
            if value == "Northwest":   df["region_Northwest"]  = 1
            elif value == "Southeast": df["region_Southeast"]  = 1
            elif value == "Southwest": df["region_Southwest"]  = 1
        elif key == "Marital Status" and value == "Unmarried":
            df["marital_status_Unmarried"] = 1
        elif key == "BMI Category":
            if value == "Obesity":       df["bmi_category_Obesity"]     = 1
            elif value == "Overweight":  df["bmi_category_Overweight"]  = 1
            elif value == "Underweight": df["bmi_category_Underweight"] = 1
        elif key == "Smoking Status":
            if value == "Occasional": df["smoking_status_Occasional"] = 1
            elif value == "Regular":  df["smoking_status_Regular"]    = 1
        elif key == "Employment Status":
            if value == "Salaried":       df["employment_status_Salaried"]      = 1
            elif value == "Self-Employed":df["employment_status_Self-Employed"] = 1
        elif key == "Insurance Plan":
            df["insurance_plan"] = plan_enc.get(value, 1)
        elif key == "Age":
            df["age"] = value
        elif key == "Number of Dependants":
            df["number_of_dependants"] = value
        elif key == "Income in Lakhs":
            df["income_lakhs"] = value
        elif key == "Genetical Risk":
            df["genetical_risk"] = value

    df["normalized_risk_score"] = calculate_normalized_risk(input_dict["Medical History"])
    df = _scale(input_dict["Age"], df)
    return df


def _scale(age: int, df: pd.DataFrame) -> pd.DataFrame:
    """Apply the age-appropriate scaler to the numeric columns."""
    scaler_obj    = scaler_young if age <= 25 else scaler_rest
    cols_to_scale = scaler_obj["cols_to_scale"]
    scaler        = scaler_obj["scaler"]

    # The scaler was fitted with an extra 'income_level' column — supply a
    # placeholder so transform() doesn't raise a shape/column error.
    df["income_level"]   = 0
    df[cols_to_scale]    = scaler.transform(df[cols_to_scale])
    df.drop("income_level", axis="columns", inplace=True)
    return df


def predict(input_dict: dict) -> int:
    """
    Return the predicted annual health-insurance premium (₹) as an integer.
    Routes to model_young for age ≤ 25, model_rest otherwise.
    """
    processed = preprocess_input(input_dict)
    model     = model_young if input_dict["Age"] <= 25 else model_rest
    return int(model.predict(processed)[0])
