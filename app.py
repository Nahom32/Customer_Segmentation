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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

def predict_loan_status(data: pd.DataFrame, model: RandomForestClassifier, scaler: StandardScaler, encoder: dict) -> np.ndarray:
    data_clean = handle_missing_values(data)
    
    for col in data_clean.columns:
        if col in encoder:  
            data_clean[col] = data_clean[col].astype(str)  # Convert to string for consistency
            data_clean[col] = data_clean[col].map(lambda x: encoder[col].transform([x])[0] if x in encoder[col].classes_ else -1)
    features = data_clean.drop(columns=['ID']).values
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return prediction

def predict_attrition_status(data: pd.DataFrame, model: RandomForestClassifier, scaler: StandardScaler, encoder: dict) -> np.ndarray:
    data_clean = handle_missing_values(data)
    for col in data_clean.columns:
        if col in encoder:  
           
            data_clean[col] = data_clean[col].astype(str)  # Convert to string for consistency
            data_clean[col] = data_clean[col].map(lambda x: encoder[col].transform([x])[0] if x in encoder[col].classes_ else -1)
    features = data_clean
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return prediction

# # Load models & encoders
model = pickle.load(open("models/loan_model_rf.pkl", "rb"))
encoder = pickle.load(open("assets/label_encoders.pkl", "rb"))
churn_encoder = pickle.load(open("churn_model/label_encoder_churn.pkl", "rb"))
scaler = pickle.load(open("assets/scaler.pkl", "rb"))
churn_scaler = pickle.load(open("churn_model/scaler_churn.pkl", "rb"))
attrition_model = pickle.load(open("churn_model/churn_model_rf.pkl", "rb"))

# # CSV File Path
# CSV_FILE = "data/loan_eligibility_sample.csv"
# ATTRITION_CSV = "data/churn.csv"
USER_CSV = "data/the_client_data.csv"

def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame()

# df = load_data(CSV_FILE)
# attrition_df = load_data(ATTRITION_CSV)
user_df = load_data(USER_CSV)

loan_selected_features = [
    "ID",
    "year",
    "loan_limit",
    "Gender",
    "approv_in_adv",
    "loan_type",
    "loan_purpose",
    "Credit_Worthiness",
    "open_credit",
    "business_or_commercial",
    "loan_amount",
    "rate_of_interest",
    "interest_rate_spread",
    "upfront_charges",
    "term",
    "Neg_ammortization",
    "interest_only",
    "lump_sum_payment",
    "property_value",
    "construction_type",
    "occupancy_type",
    "Secured_by",
    "total_units",
    "income",
    "credit_type",
    "Credit_Score",
    "co-applicant_credit_type",
    "age",
    "submission_of_application",
    "LTV",
    "Region",
    "Security_Type",
    "dtir1",
    "Status"
]
loan_df_selected = user_df[loan_selected_features]
credit_selected_features = [
    "ID",
    "age",
    "Gender",
    "Dependent_count",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
    "Attrition_Flag"
]
credit_df_features = user_df[credit_selected_features]

# Save Data Function
def save_data(user_df):
    user_df.to_csv(USER_CSV, index=False)

# Streamlit App Layout
st.set_page_config(page_title="Bank B", layout="wide")

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "Finance Dashboard",
        ["🏠 Home", "📊 Loan Analytics", "📋 Loan Eligible Applicants", "💳 Credit Card Attrition", "📈 Invest Recommendations", "📝 Add Client"],
        icons=["house", "bar-chart", "list", "credit-card", "chart-line", "person-add"],
        menu_icon="cast", default_index=0
    )

if selected == "🏠 Home":
    st.title("🏦 Welcome to Bank B's Loan Dashboard")
    
    st.write("""
    **Welcome to Bank X's Financial Dashboard!** 🎉

    Our dashboard provides key insights into loan eligibility, financial health, and customer retention. Explore the features below:

    - **📊 Loan Analytics**: Analyze trends in loan approvals based on income, credit score, and loan amounts.
    
    - **📋 Loan Eligible Applicants**: View and filter eligible loan applicants by income, credit score, and loan amount.

    - **💳 Credit Card Attrition**: Examine trends in customer retention and predict attrition with advanced analytics.

    - **📈 Investment Recommendations**: Get personalized investment plans for customers based on their financial profiles.

    - **📝 Add Client**: Easily add new clients to the system with a simple form. Input important details to expand your client base.

    **Key Features:**
    - 📊 Interactive visualizations: Track loan eligibility trends.
    - 🔍 Customizable filters: Narrow down applicants by key factors.
    - 🔮 Predictive models: Forecast customer retention and suggest investments.
    - 📂 Easy client management: Add and manage clients with ease.

    Use the sidebar to explore each section and gain valuable insights! 🚀
    """)

# Loan Analytics Page
elif selected == "📊 Loan Analytics":
    st.title("📊 Loan Eligibility Statistics")
    
    if not loan_df_selected.empty:
        # Loan Eligibility Count
        loan_counts = loan_df_selected['Status'].value_counts()
        fig1 = px.pie(loan_counts, names=loan_counts.index, values=loan_counts.values, title="Loan Approval Distribution")
        st.plotly_chart(fig1)
        
        # Loan Eligibility by Income
        fig2 = px.histogram(loan_df_selected, x="income", color="Status", title="Loan Eligibility by Income")
        st.plotly_chart(fig2)
        
        # Loan Eligibility by Credit Score
        fig3 = px.box(loan_df_selected, x="Status", y="Credit_Score", title="Loan Eligibility by Credit Score")
        st.plotly_chart(fig3)

        fig4 = px.box(loan_df_selected, x="Status", y="loan_amount", title="Loan Eligibility by Loan Amount Requested")
        st.plotly_chart(fig4)
    else:
        st.warning("No data available.")

# Eligible Applicants Page
elif selected == "📋 Loan Eligible Applicants":
    st.title("📋 List of Eligible Applicants")
    eligible_df = loan_df_selected[loan_df_selected['Status'] == 1]
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

elif selected == "💳 Credit Card Attrition":
    st.title("💳 Credit Card Attrition Analysis")
    if not credit_df_features.empty:
        st.write("Explore trends and predictions related to customer retention.")
        attrition_counts = credit_df_features['Attrition_Flag'].value_counts()
        fig = px.pie(attrition_counts, names=attrition_counts.index, values=attrition_counts.values, title="Attrition Distribution")
        st.plotly_chart(fig)

        customer_id = st.number_input("Enter Customer ID", min_value=1, step=1)
        customer_df = credit_df_features[credit_df_features['ID'] == customer_id]
        if not customer_df.empty:
            st.write(customer_df["Attrition_Flag"])
            val = customer_df["Attrition_Flag"].values[0]
            st.success("✅ Customer Retained" if val == 0 else "❌ Customer Attrited")
        else:
            st.warning("No customer found with this ID.")
    else:
        st.warning("No credit card attrition data available.")

elif selected == "📈 Invest Recommendations":
    st.title("📈 Personalized Invest Recommendations")
    if not loan_df_selected.empty:
        dfi = loan_df_selected.copy()
        st.write("Clustering customers into investment groups based on financial behavior.")
        # st.write("Available Columns in Data:", dfi.columns)

        # Selecting relevant columns for clustering
        cluster_features = ['ID', 'Credit_Score', "property_value"]

        investment_df = dfi[cluster_features].dropna()
        print(investment_df)
        # Standardizing the features
        u_scaler = StandardScaler()
        investment_scaled = u_scaler.fit_transform(investment_df)
        print(investment_scaled)
        # Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        investment_df.loc[investment_df.index, 'Investment_Cluster'] = kmeans.fit_predict(investment_scaled)
        print(investment_df)

        investment_mapping = {
            0: "Conservative: Fixed deposits, Bonds, Retirement plans",
            1: "Balanced: Mutual funds, ETFs, Diversified investments",
            2: "Aggressive: Stocks, Cryptocurrencies, High-risk assets"
        }
        investment_df['Investment_Recommendation'] = investment_df['Investment_Cluster'].map(investment_mapping)

        fig = px.scatter(investment_df, x='Credit_Score', y='property_value', color='Investment_Cluster', title='Investment Segmentation')
        st.plotly_chart(fig)
        

        customer_id = st.number_input("Enter Customer ID", min_value=1, step=1)
        if customer_id in investment_df['ID'].values:
            recommendation = investment_df[investment_df['ID'] == customer_id]['Investment_Recommendation'].values[0]
            st.success(f"Recommended Investment Plan: {recommendation}")
        else:
            st.warning("No customer found with this ID.")
    else:
        st.warning("No data available for investment recommendations.")
        
elif selected == "📝 Add Client":
    st.title("📝 Add a New Client")
    with st.form("client_form"):
        col1, col2 = st.columns(2)
        with col1:
            id = st.number_input("Client Number", min_value=1, step=1)
            age = st.number_input("Age",min_value=18, max_value=100, step=1)
            age_range = st.selectbox("Age Range", ['25-34' ,'55-64', '35-44', '45-54', '65-74', '>74' ,'<25'])
            gender = st.selectbox("Gender", ["Male", "Female"])
            year = st.number_input("Year", min_value=2000, max_value=2022, step=1)

            income = st.number_input("Income ($)", min_value=0.0, step=1000.0)
            credit_type = st.selectbox("Credit Type", ["EXP", "EQUI", "CRIF", "CIB"])
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)

            loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, step=1000.0)
            # loan_purpose = st.selectbox("Loan Purpose", ["p1", "p2", "p3", "p4"])
            loan_type = st.selectbox("Loan Type", ["type1", "type2", "type3"])
            credit_worthiness = st.selectbox("Credit Worthiness", ["I1", "I2"])

            loan_limit = st.selectbox("Loan Limit", ["cf", "ncf"])
            approv_in_adv = st.selectbox("Approval in Advance", ["pre", "nopre"])
            rate_of_interest = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, step=0.1)

            interest_rate_spread = st.number_input("Interest Rate Spread (%)", min_value=0.0, max_value=30.0, step=0.1)
            upfront_charges = st.number_input("Upfront Charges ($)", min_value=0.0, step=100.0)
            term = st.number_input("Term (Years)", min_value=1, step=1)

            property_value = st.number_input("Property Value ($)", min_value=0.0, step=1000.0)
            construction_type = st.selectbox("Construction Type", ["sb", "mh"])
            occupancy_type = st.selectbox("Occupancy Type", ["pr", "sr", "ir"])
            loan_purpose = st.selectbox("Loan Purpose", ["p1", "p2", "p3", "p4"])

            secured_by = st.text_input("Secured By (Home/Land)", max_chars=100)
            total_units = st.selectbox("Total Units", ["1U", "2U", "3U", "4U"])
            neg_ammortization = st.selectbox("Negative Amortization", ["not_neg", "neg_amm"])
            security_type = st.selectbox("Security Type", ["direct", "indirect"])  

        
            interest_only = st.selectbox("Interest Only Period", ["int_only", "not_int"])
            lump_sum_payment = st.selectbox("Lump Sum Payment", ["lpsm", "not_lpsm"])
            submission_of_application = st.selectbox("Submission of Application", ["to_inst", "not_inst"])

        with col2:
            region = st.selectbox("Region", ["south", "North", "central", "North-East"])
            business_or_commercial = st.selectbox("Business/Commercial", ["b/c", "nob/c"])
            co_applicant_credit_type = st.selectbox("Co-Applicant Credit Type", ["CIB", "EXP"])

            co_applicant_credit_score = st.number_input("Co-Applicant Credit Score", min_value=300, max_value=900, step=1)
            dtir1 = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, step=0.01)
            open_credit = st.selectbox("Open Credit", ["opc", "nopc"])
            ltv = st.number_input("Loan-to-Value Ratio", min_value=0.0, max_value=1.0, step=0.01)

            Dependent_Count = st.number_input("Dependent Count", min_value=0, step=1)
            Educational_Level = st.selectbox("Educational Level", ["High School", "Graduate", "Uneducated", "College", "Post-Graduate", "Doctorate"])
            Martial_Status = st.selectbox("Martial Status", ["Married", "Single", "Divorced", "Unknown"])
            Income_Category = st.selectbox("Income Category", ["$60K - $80K", "Less than $40K", "$80K - $120K", "$40K - $60K", "$120K +"])
            Card_Category = st.selectbox("Card Category", ["Blue", "Gold", "Silver", "Platinum"])
            
            Months_on_book = st.number_input("Months on Book", min_value=0, step=1)
            Total_Relationship_Count = st.number_input("Total Relationship Count", min_value=0, step=1)
            Months_Inactive_12_mon = st.number_input("Months Inactive (12 months)", min_value=0, step=1)
            Contacts_Count_12_mon = st.number_input("Contacts Count (12 months)", min_value=0, step=1)
        
            Credit_Limit = st.number_input("Credit Limit", min_value=0.0, step=1000.0)
            Total_Revolving_Bal = st.number_input("Total Revolving Balance", min_value=0.0, step=1000.0)
            Avg_Open_To_Buy = st.number_input("Average Open to Buy", min_value=0.0, step=1000.0)
            Total_Amt_Chng_Q4_Q1 = st.number_input("Total Amount Change Q4-Q1", min_value=0.0, step=0.01)
            
            Total_Trans_Amt = st.number_input("Total Transaction Amount", min_value=0.0, step=1000.0)
            Total_Trans_Ct = st.number_input("Total Transaction Count", min_value=0, step=1)
            Total_Ct_Chng_Q4_Q1 = st.number_input("Total Count Change Q4-Q1", min_value=0.0, step=0.01)
            Avg_Utilization_Ratio = st.number_input("Average Utilization Ratio", min_value=0.0, max_value = 1.0, step=0.01)
    
        submitted = st.form_submit_button("Add Client")
    
        if submitted:
            new_client = pd.DataFrame({
                "ID": [id],
                "year": [year],
                "loan_limit": [loan_limit],
                "Gender": [gender],
                "approv_in_adv": [approv_in_adv],
                "loan_type": [loan_type],
                "loan_purpose": [loan_purpose],
                "Credit_Worthiness": [credit_worthiness],
                "open_credit": [open_credit],
                "business_or_commercial": [business_or_commercial],
                "loan_amount": [loan_amount],
                "rate_of_interest": [rate_of_interest],
                "interest_rate_spread": [interest_rate_spread],
                "upfront_charges": [upfront_charges],
                "term": [term],
                "Neg_ammortization": [neg_ammortization],
                "interest_only": [interest_only],
                "lump_sum_payment": [lump_sum_payment],
                "property_value": [property_value],
                "construction_type": [construction_type],
                "occupancy_type": [occupancy_type],
                "Secured_by": [secured_by],
                "total_units": [total_units],
                "income": [income],
                "credit_type": [credit_type],
                "Credit_Score": [credit_score],
                "co-applicant_credit_type": [co_applicant_credit_type],
                "age": [age_range],
                "submission_of_application": [submission_of_application],
                "LTV": [ltv],
                "Region": [region],
                "Security_Type": [security_type],
                "dtir1": [dtir1],

                "age": [age],
                "Dependent_count": [Dependent_Count],
                "Education_Level": [Educational_Level],
                "Marital_Status": [Martial_Status],
                "Income_Category": [Income_Category],
                "Card_Category": [Card_Category],
                "Months_on_book": [Months_on_book],
                "Total_Relationship_Count": [Total_Relationship_Count],
                "Months_Inactive_12_mon": [Months_Inactive_12_mon],
                "Contacts_Count_12_mon": [Contacts_Count_12_mon],
                "Credit_Limit": [Credit_Limit],
                "Total_Revolving_Bal": [Total_Revolving_Bal],
                "Avg_Open_To_Buy": [Avg_Open_To_Buy],
                "Total_Amt_Chng_Q4_Q1": [Total_Amt_Chng_Q4_Q1],
                "Total_Trans_Amt": [Total_Trans_Amt],
                "Total_Trans_Ct": [Total_Trans_Ct],
                "Total_Ct_Chng_Q4_Q1": [Total_Ct_Chng_Q4_Q1],
                "Avg_Utilization_Ratio": [Avg_Utilization_Ratio]
            })

            loan = pd.DataFrame({
                "ID": [id],
                "year": [year],
                "loan_limit": [loan_limit],
                "Gender": [gender],
                "approv_in_adv": [approv_in_adv],
                "loan_type": [loan_type],
                "loan_purpose": [loan_purpose],
                "Credit_Worthiness": [credit_worthiness],
                "open_credit": [open_credit],
                "business_or_commercial": [business_or_commercial],
                "loan_amount": [loan_amount],
                "rate_of_interest": [rate_of_interest],
                "interest_rate_spread": [interest_rate_spread],
                "upfront_charges": [upfront_charges],
                "term": [term],
                "Neg_ammortization": [neg_ammortization],
                "interest_only": [interest_only],
                "lump_sum_payment": [lump_sum_payment],
                "property_value": [property_value],
                "construction_type": [construction_type],
                "occupancy_type": [occupancy_type],
                "Secured_by": [secured_by],
                "total_units": [total_units],
                "income": [income],
                "credit_type": [credit_type],
                "Credit_Score": [credit_score],
                "co-applicant_credit_type": [co_applicant_credit_type],
                "age": [age_range],
                "submission_of_application": [submission_of_application],
                "LTV": [ltv],
                "Region": [region],
                "Security_Type": [security_type],
                "dtir1": [dtir1],

            })

            credit = pd.DataFrame({
                "age": [age],
                "Gender": [gender],
                "Dependent_count": [Dependent_Count],
                "Education_Level": [Educational_Level],
                "Marital_Status": [Martial_Status],
                "Income_Category": [Income_Category],
                "Card_Category": [Card_Category],
                "Months_on_book": [Months_on_book],
                "Total_Relationship_Count": [Total_Relationship_Count],
                "Months_Inactive_12_mon": [Months_Inactive_12_mon],
                "Contacts_Count_12_mon": [Contacts_Count_12_mon],
                "Credit_Limit": [Credit_Limit],
                "Total_Revolving_Bal": [Total_Revolving_Bal],
                "Avg_Open_To_Buy": [Avg_Open_To_Buy],
                "Total_Amt_Chng_Q4_Q1": [Total_Amt_Chng_Q4_Q1],
                "Total_Trans_Amt": [Total_Trans_Amt],
                "Total_Trans_Ct": [Total_Trans_Ct],
                "Total_Ct_Chng_Q4_Q1": [Total_Ct_Chng_Q4_Q1],
                "Avg_Utilization_Ratio": [Avg_Utilization_Ratio]

            })
            status = predict_loan_status(loan, model, scaler, encoder)
            new_client['Status'] = status

            attrition_flag = predict_attrition_status(credit, attrition_model, churn_scaler, churn_encoder)
            new_client['Attrition_Flag'] = attrition_flag

            user_df = pd.concat([user_df, new_client], ignore_index=True)
            
            save_data(user_df)
            st.success("Client added successfully!")

st.sidebar.success("Select a page above ⬆️")