# 🚢 Deploying Machine Learning Models with Streamlit – Titanic Survival Prediction

## 📘 Overview

This project demonstrates a **production-ready machine learning deployment pipeline** using **Streamlit** to predict Titanic passenger survival based on engineered features and optimized models. The app enables real-time predictions, model explanations via SHAP, and intuitive dashboards for users.

The solution showcases:

- ✅ Real-time prediction with model confidence
- 📊 Model explainability using SHAP
- 📈 Visual dashboards and performance insights
- 🛠️ Modular code for training, evaluation, and deployment
- ⚙️ Fast and high-accuracy (~80%+) models via Optuna tuning

🔗 **Hosted Web App**: *[Add your deployed Streamlit app link here]*

## 📁 Project Structure

```
Assignment/
└── week 7/
    ├── .streamlit/                   # Ensure consistent and reliable app behavior across environments.
    │   └── config.toml
    ├── app.py                        # Entry point for Streamlit app
    ├── requirements.txt              # Project dependencies
    │
    ├── data/                         # Datasets
    │   ├── train.csv                 # Primary training dataset
    │   └── titanic_sample.csv        # Backup synthetic dataset
    │
    ├── docs/                         # Documentation files
    │   ├── user_guide.md
    │   ├── api_docs.md
    │   ├── model_cards.md
    │   ├── deployment_guide.md
    │   ├── model_registry.md
    │   ├── monitoring_dashboard.md
    │   └── knowledge_base.md
    │
    ├── models/                       # Trained models and metadata
    │   ├── best_model.pkl
    │   ├── feature_names.pkl
    │   └── model_metadata.json
    │
    ├── pages/                        # Streamlit multipage setup
    │   ├── 1_Home.py
    │   ├── 2_Prediction.py
    │   ├── 3_Model_Insights.py
    │   └── 4_About.py
    │
    ├── scripts/                      # Training workflows
    │   └── train_high_performance.py
    │
    ├── src/                          # Modular codebase
    │   ├── data_loader.py
    │   ├── feature_engineering.py
    │   ├── data_processor.py
    │   ├── model_loader.py
    │   ├── prediction_utils.py
    │   └── visualization_utils.py
    │
    └── README.md 
```

## ⚙️ `.streamlit/config.toml` – App Configuration

- This directory contains configuration files that define how the Streamlit app behaves across different environments.

The `config.toml` file ensures:

- **Consistent appearance** (theme, layout, sidebar settings)
- **Reliable behavior** across local and cloud deployments
- **Custom branding** (e.g., title, favicon)

###  📌 Tip

Including `.streamlit/config.toml` in your repo guarantees the app will look and behave the same on every machine or platform it's deployed to.


## 📊 Dataset Sources

- [`train.csv`](https://www.kaggle.com/datasets/yasserh/titanic-dataset) – Kaggle Titanic dataset  
- [`titanic_sample.csv`](https://github.com/beginerSE/titanic_sample/blob/main/titanic_train_mod.csva) – Realistic synthetic fallback dataset  
- [Public fallback](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) – Auto-download source if local files are missing  

## ✅ Model Training Summary

This section summarizes results from running:

```bash
python scripts/train_high_performance.py
```

### ⚙️ Feature Engineering

```
🚀 FIXED High-Performance Training for 80%+ Accuracy
============================================================
✅ Loaded existing dataset: (891, 12)
🔧 Applying feature engineering...
Starting enhanced feature engineering...
✓ Title extraction completed
✓ Family features created
✓ Fare features created
✓ Age features created
✓ Cabin features created
✓ Ticket features created
✓ Name features created
✓ Interaction features created
✓ Converting data types for compatibility...
✓ Enhanced feature engineering completed! New shape: (891, 62)
✅ Features created: (891, 62)
🔧 Selected 12 high-impact features:
['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'FarePerPerson', 'CabinLetter', 'HasCabin', 'IsChild']
📊 Final training features: 12
📊 Feature names: [...]
```

### 🤖 Model Results

| Model                    | Accuracy | CV Mean | CV Std |
|--------------------------|----------|---------|--------|
| RandomForest       | 81.01%   | 0.7471  | ±0.0296 |
| XGBoost            | 81.56%   | 0.7345  | ±0.0161 |
| LogisticRegression | **82.12%** | **0.7598**  | ±0.0194 |

🏆 **Best Model**: `LogisticRegression`  
🎯 **Target Achieved**: ✅ 80%+ Accuracy  
💾 **Saved To**: `models/best_model.pkl` + metadata + feature names

## ▶️ Running the Web App

### 1. Clone the Repository

```bash
git clone https://github.com/ShubhamS168/Celebal-CSI-Data-Science/tree/main
cd Assignment/Week 7/
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv titanic_env
# Windows
titanic_env\Scripts\activate
# Linux/macOS
source titanic_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the App

```bash
streamlit run app.py
```

📍 Navigate to [http://localhost:8501](http://localhost:8501)

## 📌 Pages & Functionality

| Page                | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| 🏠 Home             | Dataset summary, model overview, key performance metrics                   |
| 🎯 Make Prediction  | User input for passenger details, real-time prediction, confidence gauge    |
| 📊 Model Insights   | Feature importance, SHAP explanations, model comparison charts              |
| ℹ️ About            | Project overview, training workflow, customization & deployment steps       |

## 💡 Customization Guidelines

- 🔄 **Data**: Replace `data/train.csv` with your own dataset  
- 🛠️ **Model Tuning**: Adjust `train_high_performance.py` for Optuna ranges  
- 📐 **Feature Engineering**: Edit `src/feature_engineering.py` for new ideas  
- 🎨 **UI Styling**: Customize CSS in `app.py`  
- 🌐 **Deployment**: Upload to Streamlit Community Cloud, Heroku, or Docker  

## 📬 Credits

- **Author**: *Shubham Sourav*  
- **Dataset**: Kaggle Titanic, GitHub Synthetic  
- **Resources Used**:
  - [Streamlit Docs](https://docs.streamlit.io/)
  - [Machine Learning Mastery – Streamlit Guide](https://machinelearningmastery.com/how-to-quickly-deploy-machine-learning-models-streamlit/)

## 🪪 License

Distributed under the MIT License.  
© 2025 Shubham Sourav. All rights reserved.