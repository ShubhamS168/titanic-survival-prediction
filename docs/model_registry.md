# 🗂️ Model Registry

## 📁 Stored Models

| File Name            | Description                          |
|----------------------|--------------------------------------|
| `best_model.pkl`     | Serialized top-performing model      |
| `feature_names.pkl`  | Ordered list of features used        |
| `model_metadata.json`| Info about training, accuracy, algorithm |

## 🔄 Versioning Notes

- **v1.0**: Initial ensemble model.
- **v2.1_fixed**: Improved feature selection and tuned parameters.
- **v3.0_high_performance**: Optuna-based tuning and updated artifacts.

> ⏱ **Last Updated**: Automatically updated during each save operation by `train_high_performance.py`.