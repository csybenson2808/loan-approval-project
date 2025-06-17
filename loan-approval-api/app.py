from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("loan_model.joblib")
features = joblib.load("features_list.joblib")

def calculate_monthly_repayment(principal, annual_interest_rate, months):
    r = annual_interest_rate / 100 / 12
    if r == 0:
        return principal / months
    repayment = (r * principal) / (1 - (1 + r) ** (-months))
    return repayment

class LoanApplication(BaseModel):
    age: int
    income: float
    credit_score: int
    loan_amount: float
    loan_term_months: int
    interest_rate: float
    num_existing_loans: int
    has_criminal_record: bool

@app.post("/predict")
async def predict(application: LoanApplication):
    input_dict = application.dict()

    # Calculate existing loans repayment estimate
    existing_loans_repayment = input_dict["num_existing_loans"] * 300  # fixed estimate

    # Calculate new loan monthly repayment
    new_loan_repayment = calculate_monthly_repayment(
        input_dict["loan_amount"], input_dict["interest_rate"], input_dict["loan_term_months"])

    # Calculate DTI
    income = input_dict["income"]
    dti = (existing_loans_repayment + new_loan_repayment) / income if income > 0 else 10.0

    input_dict["dti"] = dti

    # Prepare input for model
    input_filtered = {feature: input_dict[feature] for feature in features}
    input_df = pd.DataFrame([input_filtered])

# Get prediction (0 or 1)
    prediction = model.predict(input_df)[0]

    # Get prediction probabilities for both classes [prob_of_0, prob_of_1]
    prediction_proba = model.predict_proba(input_df)[0]
    approval_prob = prediction_proba[1]  # Probability loan is approved

    return {
        "approved": bool(prediction),
        "approval_probability": round(approval_prob, 4),  # Rounded to 4 decimals
        "message": "Loan approved" if prediction == 1 else "Loan rejected",
        "dti": round(dti, 3)    
    }
