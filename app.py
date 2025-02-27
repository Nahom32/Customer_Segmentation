import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

def handle_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    df_filled = dataframe.copy()

    for column in df_filled.columns:
        if df_filled[column].dtype == 'object':  # For categorical columns
            df_filled[column] = df_filled[column].replace(',,', df_filled[column].mode()[0]).fillna(df_filled[column].mode()[0])
        else:  # For numerical columns
            df_filled[column] = df_filled[column].replace(',,', df_filled[column].mean()).fillna(df_filled[column].mean())

    # Final check and reporting
    missing_cols = df_filled.columns[df_filled.isna().sum() > 0].tolist()
    
    if missing_cols:
        print(f"Warning: Some NaN values remain in the following columns: {missing_cols}")
    else:
        print("✅ All missing values successfully handled!")

    return df_filled

def encode_categorical_columns(df: pd.DataFrame, encoder: dict) -> pd.DataFrame:
    df_encoded = df.copy()
    for col in df.columns:
        if col in encoder:  
            df_encoded[col] = df_encoded[col].astype(str)  # Convert to string for consistency
            df_encoded[col] = df_encoded[col].map(lambda x: encoder[col].transform([x])[0] if x in encoder[col].classes_ else -1)
    
    return df_encoded

# # Load models & encoders
model = pickle.load(open("models/loan_model_rf.pkl", "rb"))
encoder = pickle.load(open("assets/label_encoders.pkl", "rb"))
scaler = pickle.load(open("assets/scaler.pkl", "rb"))

# # CSV File Path
CSV_FILE = "data/loan_eligibility_sample.csv"

# Load Data
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
else:
    df = pd.DataFrame(columns=["user_id", "loan_amount", "income", "credit_score", "loan_purpose", "status"])

# Save Data Function
def save_data():
    df.to_csv(CSV_FILE, index=False)

# Streamlit App Layout
st.set_page_config(page_title="Bank B", layout="wide")

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "Bank B",
        ["🏠 Home", "📊 Loan Analytics", "📋Loan Eligible Applicants", "📢 Explainability"],
        icons=["house", "bar-chart", "list", "info-circle"],
        menu_icon="cast", default_index=0
    )

# Home Page
if selected == "🏠 Home":
    st.title("🏦 Welcome to Bank B's Loan Dashboard")
    st.write("Analyze loan eligibility trends and gain insights into applicant approvals.")

# Loan Analytics Page
elif selected == "📊 Loan Analytics":
    st.title("📊 Loan Eligibility Statistics")
    
    if not df.empty:
        # Loan Eligibility Count
        loan_counts = df['Status'].value_counts()
        fig1 = px.pie(loan_counts, names=loan_counts.index, values=loan_counts.values, title="Loan Approval Distribution")
        st.plotly_chart(fig1)
        
        # Loan Eligibility by Income
        fig2 = px.histogram(df, x="income", color="Status", title="Loan Eligibility by Income")
        st.plotly_chart(fig2)
        
        # Loan Eligibility by Credit Score
        fig3 = px.box(df, x="Status", y="Credit_Score", title="Loan Eligibility by Credit Score")
        st.plotly_chart(fig3)

        fig4 = px.box(df, x="Status", y="loan_amount", title="Loan Eligibility by Loan Amount Requested")
        st.plotly_chart(fig4)
    else:
        st.warning("No data available.")

# Eligible Applicants Page
elif selected == "📋Loan Eligible Applicants":
    st.title("📋 List of Eligible Applicants")
    eligible_df = df[df['Status'] == 1]
    # st.dataframe(eligible_df[['ID', 'loan_amount', 'income', 'loan_purpose']])
    # print(eligible_df)
    # if not eligible_df.empty:
    #     income_range = st.slider("Filter by Income Range", min_value=int(eligible_df['income'].min()), max_value=int(eligible_df['income'].max()), value=(int(eligible_df['income'].min()), int(eligible_df['income'].max())))
    #     filtered_df = eligible_df[(eligible_df['income'] >= income_range[0]) & (eligible_df['income'] <= income_range[1])]
    #     st.dataframe(filtered_df)
    # else:
    #     st.warning("No eligible applicants found.")

    if not eligible_df.empty:
        # Income filter
        income_range = st.slider("Filter by Income Range", min_value=int(eligible_df['income'].min()), max_value=int(eligible_df['income'].max()), value=(int(eligible_df['income'].min()), int(eligible_df['income'].max())))
        
        # Credit score filter
        credit_score_range = st.slider("Filter by Credit Score Range", min_value=int(eligible_df['Credit_Score'].min()), max_value=int(eligible_df['Credit_Score'].max()), value=(int(eligible_df['Credit_Score'].min()), int(eligible_df['Credit_Score'].max())))

        # Loan amount filter
        loan_amount_range = st.slider("Filter by Loan Amount Range", min_value=int(eligible_df['loan_amount'].min()), max_value=int(eligible_df['loan_amount'].max()), value=(int(eligible_df['loan_amount'].min()), int(eligible_df['loan_amount'].max())))

        # Apply all filters
        filtered_df = eligible_df[
            (eligible_df['income'] >= income_range[0]) & (eligible_df['income'] <= income_range[1]) &
            (eligible_df['Credit_Score'] >= credit_score_range[0]) & (eligible_df['Credit_Score'] <= credit_score_range[1]) &
            (eligible_df['loan_amount'] >= loan_amount_range[0]) & (eligible_df['loan_amount'] <= loan_amount_range[1])
        ]
        st.dataframe(filtered_df)
    else:
        st.warning("No eligible applicants found.")

# Explainability Page
elif selected == "📢 Explainability":
    st.title("📢 Why Was This Applicant Approved or Denied?")
    
    applicant_id = st.number_input("Enter Applicant ID", min_value=1, step=1)
    applicant_df = df[df['ID'] == applicant_id]
    
    if not applicant_df.empty:
        app_df = handle_missing_values(applicant_df)
        data = encode_categorical_columns(app_df, encoder)
        features = data.drop(columns=['Status', 'ID']).values
        features_scaled = scaler.transform(features)

        st.write("🔍 SHAP Decision Tree Explanation")
        generate_shap_summary_plot(app_df, model, scaler, encoder)
        

        st.write("Decision Reasoning")
        prediction = model.predict(features_scaled)
        st.success("✅ Loan Approved!" if prediction[0] == 1 else "❌ Loan Denied")

    else:
        st.warning("No applicant found with this ID.")
st.sidebar.success("Select a page above ⬆️")