# Loan Approval Prediction API

A deployable machine learning API that predicts loan approvals based on applicant financial data. Built with **FastAPI**, trained using **scikit-learn**, and containerized using **Docker**, this project demonstrates an end-to-end ML deployment pipeline.

---

## 🚀 Features

- Predict loan approvals using a trained RandomForestClassifier
- Calculates and considers **Debt-to-Income (DTI)** automatically
- Exposes a `/predict` endpoint via FastAPI
- Dockerized and deployable to AWS EC2
- Returns both prediction and approval probability

---

## 📂 Project Structure

```
loan-approval-api/
├── app.py                  # FastAPI app
├── train_model.py          # Script to train and save the ML model
├── loan_data_generator.py  # Script to generate synthetic loan data
├── loan_data.csv           # Dataset (generated)
├── loan_model.joblib       # Trained model
├── features_list.joblib    # List of model features
├── requirements.txt        # Python dependencies
└── Dockerfile              # Container instructions
```

---

## 🔮 Prediction Input

POST `/predict`

### Example JSON:

```json
{
  "age": 52,
  "income": 15000,
  "credit_score": 650,
  "loan_amount": 100000,
  "loan_term_months": 24,
  "interest_rate": 3,
  "num_existing_loans": 1,
  "has_criminal_record": false
}
```

### Response:

```json
{
  "approved": false,
  "approval_probability": 0.42,
  "message": "Loan rejected",
  "dti": 0.307
}
```

---

## 🧪 Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Train model (if needed)
python train_model.py

# Run API
uvicorn app:app --reload

# Access docs
http://localhost:8000/docs
```

---

## 🐳 Docker Usage

### Build the image

```bash
docker build -t loan-api .
```

### Run the container

```bash
docker run -d -p 8000:8000 loan-api
```

### Access the API

```
http://<your-ec2-ip>:8000/docs
```

---

## ☁️ AWS Deployment Steps

1. Launch an EC2 instance (Amazon Linux or Ubuntu)
2. Open ports 22 (SSH) and 8000 (API) in security group
3. SSH into instance and install Docker
4. Copy project to EC2 using `scp`
5. Build and run Docker container

---

## 📈 Model Training Info

- **Algorithm**: RandomForestClassifier
- **Features**: Includes computed DTI (not user-input)
- **Accuracy**: \~98.5%
- **ROC-AUC**: 0.9948
- **Balanced for class imbalance** using `class_weight='balanced'`

---

## 📘 Author

Built by Benson Chua as part of a self-directed AI infrastructure learning journey.

---

## 📜 License

MIT License

