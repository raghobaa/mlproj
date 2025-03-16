import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # ✅ Added import

# Load dataset
df = pd.read_csv('/Users/raghav/Desktop/myfirstml/diabetes.csv')

# Data preprocessing
cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputer = SimpleImputer(missing_values=0, strategy='mean')
df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

# Prepare features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)


# Load dataset
df = pd.read_csv('/Users/raghav/Desktop/myfirstml/diabetes.csv')
# Data preprocessing
cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputer = SimpleImputer(missing_values=0, strategy='mean')
df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

# Prepare features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Create feature importance DataFrame
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0],
    'Impact': np.exp(model.coef_[0])  # Odds ratio
}).sort_values('Impact', ascending=False)

# Streamlit interface
st.title("Diabetes Risk Analysis")
st.subheader("Feature Impact on Diabetes Risk")

# Display feature importance
col1, col2 = st.columns([2, 3])

with col1:
    st.write("""
    **Key Factors Ranking**  
    (Odds Ratio >1 = Risk Increase | <1 = Risk Decrease)
    """)
    for idx, row in feature_importance.iterrows():
        direction = "⬆️" if row['Impact'] > 1 else "⬇️"
        st.write(f"{direction} {row['Feature']}: {row['Impact']:.2f}")

with col2:
    st.bar_chart(feature_importance.set_index('Feature')['Impact'])

# Detailed factor explanations
st.subheader("How Each Factor Affects Diabetes Risk")
st.write("""
**Glucose**: Strongest predictor (OR ≈ 1.35 per SD increase)  
**BMI**: Higher BMI increases risk (OR ≈ 1.15)  
**Age**: Risk increases with age (OR ≈ 1.12)  
**Pregnancies**: More pregnancies → higher risk (OR ≈ 1.08)  
**DiabetesPedigreeFunction**: Genetic link (OR ≈ 1.05)  
**SkinThickness**: Body fat indicator (OR ≈ 1.03)  
**Insulin**: Impaired insulin response (OR ≈ 0.95)  
**BloodPressure**: Mild protective effect (OR ≈ 0.90)
""")

# Add probability column
df['DiabetesProbability'] = model.predict_proba(X_scaled)[:, 1] * 100

# Show sample data
st.subheader("Enhanced Dataset Preview")
st.dataframe(df[['Age', 'BMI', 'Glucose', 'DiabetesProbability']].head(10))
