# 💡 Knowledge Base

## ❓ Common Issues

### ❌ Model File Not Found
**Error**: `FileNotFoundError: models/best_model.pkl not found`  
**Solution**: Run the following command to generate model artifacts:  
```bash
python scripts/train_high_performance.py
```

### ❌ Prediction Error – Feature Mismatch
**Error**: Feature names should match those passed during fit…  
**Solution**:

- Ensure the input features match those used during training.
- Make sure `data_processor.py` uses the same logic and feature names as `feature_engineering.py`.
- Load and align features using `models/feature_names.pkl`.

### ❌ Categorical Encoding Error
**Error**: could not convert string to float: 'male'  
**Solution**:

- Update `process_user_input()` to encode categorical inputs like `Sex` and `Embarked` before calling the model.

---

## 🧠 How to Retrain

Update your training data in:

```bash
data/train.csv
```

Run the training script:

```bash
python scripts/train_high_performance.py
```

This will update:

- `models/best_model.pkl`
- `models/feature_names.pkl`
- `models/model_metadata.json`

Restart your Streamlit app:

```bash
streamlit run app.py
```
