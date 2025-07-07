# 🧭 User Guide

This guide explains how to use the **Titanic Survival Prediction System** Streamlit app.

---

## 🚀 Launching the App

1. Open a terminal in the project root directory.
2. Activate your virtual environment (if using one):

```bash
# For macOS/Linux
source venv/bin/activate

# For Windows
venv\Scripts\activate
```

3. Install dependencies (only needed once):

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run app.py
```

---

## 🧭 Navigating the App

- **🏠 Home**: Overview of the project, dataset preview, and key metrics.
- **🎯 Make Prediction**: Enter passenger details to receive a survival prediction.
- **📊 Model Insights**: Visual insights into model performance and feature importance.
- **ℹ️ About**: Project background, dataset origin, and deployment details.

---

## ✍️ Input Fields

| Field       | Type        | Description                                 |
|-------------|-------------|---------------------------------------------|
| `Pclass`    | 1, 2, 3     | Passenger class (ticket class)              |
| `Sex`       | male/female | Gender of the passenger                     |
| `Age`       | 0–100       | Passenger age in years                      |
| `SibSp`     | 0–10        | Number of siblings/spouses aboard          |
| `Parch`     | 0–10        | Number of parents/children aboard          |
| `Fare`      | float       | Ticket price in 1912 British Pounds         |
| `Embarked`  | S, C, Q     | Port of Embarkation                         |
| `Cabin`     | text (opt.) | Cabin number (optional)                     |
| `Ticket`    | text (opt.) | Ticket number (optional)                    |
| `Name`      | text (opt.) | Passenger full name (optional)             |

---

## ✅ Example Prediction Workflow

1. Navigate to the **🎯 Make Prediction** page.
2. Fill out all relevant fields.
3. Click on **🎯 Predict Survival**.
4. View the results:

   - 🟩 **Survived** or 🟥 **Did Not Survive**
   - 📊 Probability gauge and confidence level
   - 📈 Feature impact bar chart
   - 📚 Historical comparison (if enabled)

---

## 📖 Interpreting the Results

- **Prediction**:  
  `1` = Survived  
  `0` = Did not survive

- **Probability**:  
  Softmax probability output from the model.

- **Confidence**:  
  The class with the highest prediction probability.

- **Feature Impact**:  
  Visual explanation of which features influenced the prediction most (using SHAP or bar charts).

---

> For further details, visit the [Model Cards](model_cards.md) and [Model Insights](../pages/3_Model_Insights.py) pages.