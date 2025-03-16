import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Set page config first
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_preprocess_data():
    # Load dataset (replace with your actual data loading)
    df = pd.read_csv("diabetes.csv")
    
    # Handle missing values (0s in medical measurements)
    cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    imputer = SimpleImputer(missing_values=0, strategy='mean')
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])
    
    return df

@st.cache_data
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    return model, scaler

# Main application
def main():
    st.title("Diabetes Risk Assessment Tool")
    st.markdown("""
    This tool predicts diabetes risk based on health parameters and explains how each factor contributes to the prediction.
    **Note:** This is not medical advice. Always consult a healthcare professional.
    """)

    # Load data and train model
    df = load_and_preprocess_data()
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    model, scaler = train_model(X, y)

    # User input section
    with st.sidebar:
        st.header("Health Parameters")
        pregnancies = st.slider("Pregnancies", 0, 17, 0)
        glucose = st.number_input("Glucose (mg/dL)", 50, 200, 100)
        bp = st.number_input("Blood Pressure (mmHg)", 20, 130, 70)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 99, 20)
        insulin = st.number_input("Insulin (μU/ml)", 0, 846, 79)
        bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.08, 2.5, 0.37)
        age = st.slider("Age", 20, 100, 30)

    # Create input array
    input_data = pd.DataFrame([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]],
                              columns=X.columns)

    if st.button("Assess Diabetes Risk"):
        try:
            # Preprocess and predict
            scaled_input = scaler.transform(input_data)
            probability = model.predict_proba(scaled_input)[0][1] * 100
            risk_level = "High Risk" if probability >= 50 else "Moderate Risk" if probability >= 30 else "Low Risk"

            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Risk Assessment")
                st.metric("Diabetes Probability", f"{probability:.1f}%")
                st.progress(probability/100)
                
                # Risk level indicator
                color = "red" if risk_level == "High Risk" else "orange" if risk_level == "Moderate Risk" else "green"
                st.markdown(f"<h3 style='color:{color}'>{risk_level}</h3>", unsafe_allow_html=True)
                
                # Recommendations
                st.subheader("Recommendations")
                if risk_level == "High Risk":
                    st.error("""
                    - Consult a healthcare professional immediately
                    - Regular blood sugar monitoring
                    - Adopt low glycemic index diet
                    - 150 mins/week moderate exercise
                    """)
                elif risk_level == "Moderate Risk":
                    st.warning("""
                    - Regular health checkups
                    - Maintain healthy weight
                    - Reduce processed sugar intake
                    - Stress management techniques
                    """)
                else:
                    st.success("""
                    - Maintain healthy lifestyle
                    - Annual health checkups
                    - Balanced diet with whole foods
                    - Regular physical activity
                    """)

            with col2:
                st.subheader("Key Contributing Factors")
                
                # Feature importance analysis
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Impact': np.exp(model.coef_[0])  # Odds ratio
                }).sort_values('Impact', ascending=False)
                
                # Visualize feature impacts
                st.bar_chart(feature_importance.set_index('Feature')['Impact'])
                
                # Detailed factor explanations
                st.subheader("Factor Explanations")
                st.write("""
                - **Glucose**: Blood sugar levels (most significant predictor)
                - **BMI**: Body mass index (obesity correlation)
                - **Age**: Risk increases with age
                - **DiabetesPedigreeFunction**: Genetic predisposition
                - **Pregnancies**: Gestational diabetes history
                - **BloodPressure**: Cardiovascular health indicator
                - **SkinThickness**: Body fat distribution
                - **Insulin**: Insulin resistance marker
                """)

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    main()
