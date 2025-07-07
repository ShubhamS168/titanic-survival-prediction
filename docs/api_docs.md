
# ⚙️ API Documentation (Future Scope)

This section describes a hypothetical REST API for programmatic access to the Titanic Survival Prediction System.

## 🔁 Predict Endpoint

Simulate a POST request to submit passenger data and receive a prediction.

### 🔗 Endpoint

```
POST /predict
Host: http://localhost:8501
Content-Type: application/json
```

## 📥 Request Body (Example Input)

```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 28,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}
```

## 📤 Response Body (Example Output)

```json
{
  "prediction": 0,
  "probability": {
    "died": 0.72,
    "survived": 0.28
  },
  "confidence": 0.72,
  "model_version": "2.1_fixed"
}
```

## 📌 Example curl Request

```bash
curl -X POST http://localhost:8501/predict   -H "Content-Type: application/json"   -d '{"Pclass": 3, "Sex": "male", "Age": 28, "SibSp": 0, "Parch": 0, "Fare": 7.25, "Embarked": "S"}'
```
