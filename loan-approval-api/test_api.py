import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "age": 52,
    "income": 20000,
    "credit_score": 500,
    "loan_amount": 100000,
    "loan_term_months": 24,
    "interest_rate": 5.5,
    "num_existing_loans": 1,
    "has_criminal_record": False
}

response = requests.post(url, json=data)
print("Response status code:", response.status_code)
print("Response JSON:", response.json())
