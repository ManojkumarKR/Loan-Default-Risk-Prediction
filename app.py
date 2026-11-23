import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


# Full Width Dashboard

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        max-width: 100% !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

if"prediction_history" not in st.session_state:
  st.session_state.prediction_history=[]

#load trained model

loaded_data = joblib.load("loan_default_model.pkl")

loaded_model = loaded_data["model"]
loaded_ct = loaded_data["ct"]
loaded_scaler = loaded_data["scaler"]


st.markdown("""
<div style="padding:20px; background-color:#172554;
border-radius:12px; margin-bottom:20px;">

<h1 style="color:white; margin-bottom:5px;">
Loan Default Risk Prediction System
</h1>

<p style="color:#bfdbfe; font-size:16px;">
AI-Powered Loan Risk Assessment and Analytics Dashboard
</p>

</div>
""", unsafe_allow_html=True)

############

st.caption("AI / ML Based Loan Risk Assessment System")

st.divider()

st.header("1. Applicant Prediction")

st.subheader("Applicant Information")

input_col1, input_col2 = st.columns(2)

with input_col1:

    age = st.number_input("Age", min_value=18, max_value=100, value=30)

    annual_income_inr = st.number_input("Annual Income (INR)", min_value=0, value=500000)

    employment_years = st.number_input("Employment Years", min_value=0.0, value=5.0)

    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)

    loan_amount_inr = st.number_input("Loan Amount (INR)", min_value=0, value=200000)

    loan_term_months = st.number_input("Loan Term (Months)", min_value=1, value=36)

    interest_rate_pct = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0)

with input_col2:

    existing_loans_count = st.number_input("Existing Loans Count", min_value=0, value=1)

    missed_payments_last_2y = st.number_input("Missed Payments in Last 2 Years", min_value=0, value=0)

    debt_to_income_ratio = st.number_input("Debt to Income Ratio", min_value=0.0, value=0.138)

    dependents = st.number_input("Number of Dependents", min_value=0, value=2)

    home_ownership = st.text_input("Home Ownership", value="Family")

    loan_purpose = st.text_input("Loan Purpose", value="Vehicle")

    employment_type = st.text_input("Employment Type", value="Salaried")


new_applicant = pd.DataFrame({
    "age": [age],
    "annual_income_inr": [annual_income_inr],
    "employment_years": [employment_years],
    "credit_score": [credit_score],
    "loan_amount_inr": [loan_amount_inr],
    "loan_term_months": [loan_term_months],
    "interest_rate_pct": [interest_rate_pct],
    "existing_loans_count": [existing_loans_count],
    "missed_payments_last_2y": [missed_payments_last_2y],
    "debt_to_income_ratio": [debt_to_income_ratio],
    "dependents": [dependents],
    "home_ownership": [home_ownership],
    "loan_purpose": [loan_purpose],
    "employment_type": [employment_type]
})

# Prediction Button
if st.button("Predict Loan Default Risk", type="primary"):

    new_applicant_ct = loaded_ct.transform(new_applicant)

    new_applicant_scaled = loaded_scaler.transform(new_applicant_ct)

    prediction = loaded_model.predict(new_applicant_scaled)

    probability = loaded_model.predict_proba(new_applicant_scaled)

    default_probability = probability[0][1] * 100

    if prediction[0] == 1:

        st.error("Loan Default Risk: HIGH")
        risk = "High"

    else:

        st.success("Loan Default Risk: LOW")
        risk = "Low"

    st.metric(
        "Default Probability",
        f"{default_probability:.2f}%"
    )

    st.session_state.prediction_history.append({

        "prediction": risk,
        "default probability": round(default_probability, 2)

    })


st.divider()


# ==================================
# 2. DASHBOARD STATISTICS
# ==================================

st.header("2. Loan Risk Dashboard")

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    total_predictions = len(history_df)

    high_risk_count = (
        history_df["prediction"].str.lower() == "high"
    ).sum()

    low_risk_count = (
        history_df["prediction"].str.lower() == "low"
    ).sum()

    high_risk_percentage = (
        high_risk_count / total_predictions
    ) * 100


    # Four Dashboard Cards

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Predictions", total_predictions)

    with col2:
        st.metric("High Risk", int(high_risk_count))

    with col3:
        st.metric("Low Risk", int(low_risk_count))

    with col4:
        st.metric("High Risk %", f"{high_risk_percentage:.2f}%")


    st.divider()


    # ==================================
    # 3. DASHBOARD CHARTS
    # ==================================

    st.header("3. Loan Risk Analytics")

    chart_col1, chart_col2 = st.columns(2)


    # Loan Risk Distribution

    with chart_col1:

        st.subheader("Loan Risk Distribution")

        risk_counts = history_df["prediction"].value_counts()

        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="High Risk vs Low Risk",
            hole=0.4
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Default Probability Analysis

    with chart_col2:

        st.subheader("Applicant Default Probability")

        probability_df = history_df.copy()

        probability_df["Applicant"] = [
            f"Applicant {i+1}"
            for i in range(len(probability_df))
        ]

        st.bar_chart(
            probability_df.set_index("Applicant")[
                "default probability"
            ],
            y_label="Default Probability (%)"
        )


    st.divider()


    # ==================================
    # 4. PREDICTION HISTORY
    # ==================================

    st.header("4. Prediction History")

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )
    
    
else:
  st.info("No prediction history available yet.")

  st.divider()

  st.header("3. Loan Risk Analytics")

  st.info("Make predictions to view dashboard charts.")

  st.header("4. Prediction History")

  st.info("Your prediction history will appear here.")


# 5 Excel Batch Prediction

st.divider()
st.header("5. Excel Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload Applicant Excel File",
    type=["xlsx"]
)

if uploaded_file is not None:

    bulk_data = pd.read_excel(uploaded_file)

    required_columns = [
        "age",
        "annual_income_inr",
        "employment_years",
        "credit_score",
        "loan_amount_inr",
        "loan_term_months",
        "interest_rate_pct",
        "existing_loans_count",
        "missed_payments_last_2y",
        "debt_to_income_ratio",
        "dependents",
        "home_ownership",
        "loan_purpose",
        "employment_type"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in bulk_data.columns
    ]

    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")

    else:
        bulk_data = bulk_data[required_columns]

        st.success("Excel file validated successfully!")

        st.write("Total applicants:", len(bulk_data))

        st.subheader("Uploaded Applicant Data")

        st.dataframe(bulk_data, use_container_width=True)

        if st.button("Predict All Applicants"):
          X_bulk = loaded_ct.transform(bulk_data)
          X_bulk_scaled = loaded_scaler.transform(X_bulk)
          bulk_predictions = loaded_model.predict(X_bulk_scaled)
          bulk_probabilities = loaded_model.predict_proba(X_bulk_scaled)[:, 1]
          bulk_data["Prediction"] = bulk_predictions
          bulk_data["Default Probability (%)"] = (bulk_probabilities * 100).round(2)
          bulk_data["Risk Level"] = ["High Risk" if p >= 0.30 else "Low Risk"
          for p in bulk_probabilities]


          # Applicant Review Recommendation

          bulk_data["Recommendation"] = [
              "Manual Review Required" if p >= 0.30
              else "Standard Credit Review"
              for p in bulk_probabilities
          ]

          # Batch Prediction Summary

          total_applicants = len(bulk_data)

          high_risk = (
              bulk_data["Risk Level"] == "High Risk"
          ).sum()

          low_risk = (
              bulk_data["Risk Level"] == "Low Risk"
          ).sum()

          high_risk_percentage = (
              high_risk / total_applicants
          ) * 100 if total_applicants > 0 else 0

          st.subheader("Batch Prediction Summary")

          col1, col2, col3, col4 = st.columns(4)

          with col1:
              st.metric("Total Applicants", total_applicants)

          with col2:
              st.metric("High Risk", int(high_risk))

          with col3:
              st.metric("Low Risk", int(low_risk))

          with col4:
              st.metric("High Risk %", f"{high_risk_percentage:.2f}%")


              # Batch Risk Distribution Chart

          st.subheader("Batch Risk Distribution")

          risk_counts = bulk_data["Risk Level"].value_counts()

          fig = px.pie(
              values=risk_counts.values,
              names=risk_counts.index,
              title="High Risk vs Low Risk Applicants",
              hole=0.4
          )

          st.plotly_chart(
              fig,
              use_container_width=True
          )

          # High Risk Applicant Table

          st.subheader("High Risk Applicants")

          high_risk_applicants = bulk_data[
              bulk_data["Risk Level"] == "High Risk"
          ]

          st.write(
              "Total High Risk Applicants:",
              len(high_risk_applicants)
          )

          st.dataframe(
              high_risk_applicants,
              use_container_width=True,
              hide_index=True
          )

          st.download_button(
              label="Download High Risk Applicants",
              data=high_risk_applicants.to_csv(index=False).encode("utf-8"),
              file_name="high_risk_applicants.csv",
              mime="text/csv"
          )


          # Low Risk Applicant Table

          st.subheader("Low Risk Applicants")

          low_risk_applicants = bulk_data[
              bulk_data["Risk Level"] == "Low Risk"
          ]

          st.write(
              "Total Low Risk Applicants:",
              len(low_risk_applicants)
          )

          st.dataframe(
              low_risk_applicants,
              use_container_width=True,
              hide_index=True
          )
          st.download_button(
              label="Download Low Risk Applicants",
              data=low_risk_applicants.to_csv(index=False).encode("utf-8"),
              file_name="low_risk_applicants.csv",
              mime="text/csv"
          )



          st.success("Excel predictions completed!")
          st.subheader("Batch Prediction Results")
          st.dataframe(bulk_data, use_container_width=True)
          st.download_button(
              label="Download Predicted File",
              data=bulk_data.to_csv(index=False).encode("utf-8"),
              file_name="loan_predictions.csv",
              mime="text/csv"
          )


          # Default Probability Distribution

          st.subheader("Default Probability Distribution")

          probability_bins = pd.cut(
              bulk_data["Default Probability (%)"],
              bins=[0, 30, 50, 70, 100],
              labels=[
                  "0-30%",
                  "30-50%",
                  "50-70%",
                  "70-100%"
              ],
              include_lowest=True
          )

          probability_counts = probability_bins.value_counts().sort_index()

          fig_probability = px.bar(
              x=probability_counts.index.astype(str),
              y=probability_counts.values,
              labels={
                  "x": "Default Probability Range",
                  "y": "Number of Applicants"
              },
              title="Applicant Default Probability Analysis",
              text=probability_counts.values
          )

          st.plotly_chart(
              fig_probability,
              use_container_width=True
          )

          # Applicant Risk Recommendations

          st.subheader("Applicant Risk Recommendations")

          if high_risk > 0:

              st.warning(
                  f"{high_risk} applicants need additional credit review."
              )

          if low_risk > 0:

              st.success(
                  f"{low_risk} applicants are classified as low risk."
              )

          st.info(
              "Recommendation: Review income, credit score, "
              "existing loans and repayment history before "
              "making final loan approval decisions."
          )




  

          
# ==================================
# 5. CLEAR HISTORY
# ==================================

if st.button("Clear Prediction History"):

    st.session_state.prediction_history = []

    st.rerun()
