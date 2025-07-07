# 🧠 Model Cards

Documenting the models used in the Titanic prediction system.

---

## ✅ LogisticRegression_Fixed

- **Accuracy**: 82.1%
- **Tuned with**: Optuna
- **Features used**: ['Pclass', 'Sex', 'Age', 'Fare', ..., 'IsChild']
- **Pros**: Fast, interpretable
- **Cons**: Sensitive to linearity

---

## 🌲 RandomForest_Fixed

- **Accuracy**: 81.0%
- **CV Score**: 0.7471 ± 0.0296
- **Notes**: Stable performance, resistant to overfitting

---

## ⚡ XGBoost_Fixed

- **Accuracy**: 81.6%
- **CV Score**: 0.7345 ± 0.0161
- **Notes**: Handles non-linear relationships better, slightly more complex

---

## 🤝 Ensemble (Voting Classifier)

- **Components**: LogisticRegression, RandomForest, XGBoost
- **Test Accuracy**: 84.3%
- **CV Mean Accuracy**: 83.5% ± 1.2%
- **Advantages**: Combines strengths, improved generalization
- **Limitations**: Slightly higher latency

---

## 📦 Model Artifacts

- `models/best_model.pkl`
- `models/feature_names.pkl`
- `models/model_metadata.json`