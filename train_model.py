import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

print("Loading historical CRM data...")
df = pd.read_csv('historical_crm_data.csv')

# 1. Define Features (X) and Target (y)
# We map our target 'Status' to binary numbers: Won = 1, Lost = 0
df['Target'] = df['Status'].apply(lambda x: 1 if x == 'Won' else 0)

feature_cols = [
    'Company_Size', 
    'Lead_Source', 
    'Deal_Value_USD', 
    'Days_In_Pipeline', 
    'Follow_Up_Calls', 
    'Emails_Sent', 
    'Decision_Maker_Engaged'
]

X = df[feature_cols]
y = df['Target']

# 2. Split Data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Preprocessing Pipeline
# Categorical columns need One-Hot Encoding; Numerical columns need Scaling
categorical_cols = ['Company_Size', 'Lead_Source']
numerical_cols = ['Deal_Value_USD', 'Days_In_Pipeline', 'Follow_Up_Calls', 'Emails_Sent', 'Decision_Maker_Engaged']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# 4. Build and Train Logistic Regression Pipeline
log_reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

print("\nTraining Logistic Regression Engine...")
log_reg_pipeline.fit(X_train, y_train)
log_preds = log_reg_pipeline.predict(X_test)
log_probs = log_reg_pipeline.predict_proba(X_test)[:, 1]

print("--- Logistic Regression Performance ---")
print(classification_report(y_test, log_preds, target_names=['Lost (Churn)', 'Won']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, log_probs):.4f}")

# 5. Build and Train Random Forest Pipeline (For Comparison)
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("\nTraining Random Forest Engine...")
rf_pipeline.fit(X_train, y_train)
rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
print(f"Random Forest ROC-AUC Score: {roc_auc_score(y_test, rf_probs):.4f}")

# 6. Save the Logistic Regression pipeline for our Streamlit app
# We choose Logistic Regression as our primary app engine because its smooth probability gradients 
# respond beautifully to our interactive "What-If" sliders!
model_filename = 'churn_pipeline_model.pkl'
joblib.dump(log_reg_pipeline, model_filename)
print(f"\nSuccess! Model saved to {model_filename}")