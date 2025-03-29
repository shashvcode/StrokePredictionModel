# Stroke Risk Prediction Model

A machine learning pipeline to predict stroke risk percentage based on patient symptoms and demographic data.

---

## What This Project Does

This project uses:
- Realistic patient symptom data
- Binary and numeric features (e.g., chest pain, age, gender)
- Preprocessing pipelines with scaling and encoding
- GridSearchCV to tune models
- XGBoost as the final model

The model predicts a patient’s **stroke risk (%)** based on 17 symptoms and demographics using a clean, fast interface.

---

## Tools Used

- Python 
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn for EDA
- Joblib for saving/loading model
- Jupyter for development

---

## Steps

1. **EDA**: Explored feature distributions, correlations, and rare conditions
2. **Feature Engineering**: Created interaction features and combined risk flags
3. **Preprocessing**: Used `StandardScaler` and `OneHotEncoder` via `ColumnTransformer`
4. **Model Selection**: Tried Ridge, Lasso, Random Forest, and XGBoost
5. **Tuning**: Performed Grid Search with 5-fold cross-validation
6. **Evaluation**: Achieved MAE ~0.9 and R² ~0.997 on test data
7. **Interface**: Created CLI-style user input for real-time stroke risk prediction

---

## 🔄 Continuous Improvement

- Currently investigating model bias caused by over-reliance on age as a predictor. 
- Although age is medically correlated with stroke risk, the model tends to overestimate risk in healthy older individuals due to skewed training data. 
-  Exploring partial dependence plots and stratified sampling to rebalance feature influence and calibrate predictions more accurately.



## How to Run

```bash
pip install -r requirements.txt
python predict.py