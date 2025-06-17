import pandas as pd
import numpy as np

def calculate_monthly_repayment(principal, annual_interest_rate, months):
    # monthly interest rate
    r = annual_interest_rate / 100 / 12
    if r == 0:
        return principal / months
    repayment = (r * principal) / (1 - (1 + r) ** (-months))
    return repayment

def generate_data(num_samples=1000):
    data = []
    for _ in range(num_samples):
        age = np.random.randint(18, 70)
        income = np.random.randint(1000, 15000)  # monthly income
        credit_score = np.random.randint(300, 850)
        loan_amount = np.random.randint(5000, 500000)
        loan_term_months = np.random.choice([12, 24, 36, 48, 60])
        interest_rate = np.random.uniform(3, 20)
        num_existing_loans = np.random.randint(0, 4)
        has_criminal_record = np.random.choice([0, 1])

        # Estimate existing loans monthly repayment (simplify: each existing loan repays 300 monthly)
        existing_loans_repayment = num_existing_loans * 300

        # Calculate new loan monthly repayment
        new_loan_repayment = calculate_monthly_repayment(loan_amount, interest_rate, loan_term_months)

        # Calculate DTI = (existing loans + new loan) / income
        dti = (existing_loans_repayment + new_loan_repayment) / income if income > 0 else 10.0  # large number if income=0

        # Simple rule for approval: approve if DTI < 0.36, credit_score > 600, no criminal record, age 21-65
        approved = int((dti < 0.36) and (credit_score > 600) and (has_criminal_record == 0) and (21 <= age <= 65))

        data.append({
            "age": age,
            "income": income,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "interest_rate": interest_rate,
            "num_existing_loans": num_existing_loans,
            "has_criminal_record": has_criminal_record,
            "dti": dti,
            "approved": approved
        })

    df = pd.DataFrame(data)
    df.to_csv("loan_data.csv", index=False)
    print("Data generated and saved to loan_data.csv")

if __name__ == "__main__":
    generate_data()
