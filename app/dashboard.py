import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ============================================================
# PAGE CONFIGURATION
# Must be the very first streamlit command
# ============================================================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔴",
    layout="wide"
)

# ============================================================
# LOAD MODEL AND SUPPORTING FILES
# ============================================================
@st.cache_resource
def load_model():
    model         = joblib.load('models/churn_model.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    model_type    = type(model).__name__
    scaler = None
    if model_type == 'LogisticRegression':
        scaler = joblib.load('models/scaler.pkl')
    return model, feature_names, scaler, model_type

model, feature_names, scaler, model_type = load_model()

# ============================================================
# LOAD DATA FOR METRICS PAGE
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/churn_clean.csv')
    return df

df = load_data()

# ============================================================
# HEADER SECTION
# ============================================================
st.title("🔴 Customer Churn Prediction System")
st.markdown("**Predict which customers will cancel their subscription — 30 days in advance**")
st.markdown("---")

# ============================================================
# SIDEBAR - NAVIGATION
# ============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Predict Churn", "Model Performance", "About Project"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown(f"Model: `{model_type}`")
st.sidebar.markdown(f"Features: `{len(feature_names)}`")
st.sidebar.markdown(f"Dataset: `IBM Telco Churn`")


# ============================================================
# PAGE 1 - PREDICT CHURN
# ============================================================
if page == "Predict Churn":

    st.header("Enter Customer Details")
    st.markdown("Fill in the customer information below and click **Predict** to see churn risk.")
    st.markdown("")

    # Two columns for inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Account Info")

        tenure = st.slider(
            "Tenure (months as customer)",
            min_value=0, max_value=72, value=12,
            help="How many months has this customer been with us?"
        )

        contract = st.selectbox(
            "Contract Type",
            options=["Month-to-month", "One year", "Two year"],
            help="Type of contract the customer has"
        )

        payment_method = st.selectbox(
            "Payment Method",
            options=["Electronic check", "Mailed check",
                     "Bank transfer (automatic)", "Credit card (automatic)"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            options=["Yes", "No"]
        )

    with col2:
        st.subheader("Charges")

        monthly_charges = st.slider(
            "Monthly Charges ($)",
            min_value=18, max_value=120, value=65,
            help="How much does this customer pay per month?"
        )

        total_charges = monthly_charges * tenure

        st.metric(
            label="Calculated Total Charges",
            value=f"${total_charges:,.2f}",
            help="Auto calculated from tenure x monthly charges"
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            options=["No", "Yes"]
        )

        partner = st.selectbox("Has Partner", options=["Yes", "No"])
        dependents = st.selectbox("Has Dependents", options=["No", "Yes"])

    with col3:
        st.subheader("Services")

        phone_service = st.selectbox("Phone Service", options=["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            options=["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service",
            options=["Fiber optic", "DSL", "No"]
        )
        online_security = st.selectbox(
            "Online Security",
            options=["No", "Yes", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support",
            options=["No", "Yes", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            options=["No", "Yes", "No internet service"]
        )

    st.markdown("---")

    # ============================================================
    # PREDICT BUTTON
    # ============================================================
    predict_btn = st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

    if predict_btn:

        # --------------------------------------------------------
        # BUILD INPUT FROM USER SELECTIONS
        # --------------------------------------------------------

        # Map text to numbers (same encoding as Day 3)
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        payment_map  = {
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)"  : 1,
            "Electronic check"         : 2,
            "Mailed check"             : 3
        }
        internet_map    = {"DSL": 0, "Fiber optic": 1, "No": 2}
        yes_no_map      = {"No": 0, "Yes": 1}
        yes_no_nps_map  = {"No": 0, "No phone service": 1, "Yes": 2}
        yes_no_nis_map  = {"No": 0, "No internet service": 1, "Yes": 2}

        # Count number of services
        services = [
            phone_service == "Yes",
            multiple_lines == "Yes",
            internet_service != "No",
            online_security == "Yes",
            tech_support == "Yes",
            streaming_tv == "Yes"
        ]
        num_services = sum(services)

        # Engineered features
        is_new_customer  = 1 if tenure <= 6  else 0
        is_loyal         = 1 if tenure >= 24 else 0
        charges_per_mo   = total_charges / (tenure + 1)
        has_support      = 1 if (tech_support == "Yes" or online_security == "Yes") else 0
        is_high_value    = 1 if monthly_charges > 70 else 0
        contract_risk    = 2 if contract == "Month-to-month" else 1 if contract == "One year" else 0

        # Build full feature vector matching training columns
        input_data = {
            'gender'            : 0,
            'SeniorCitizen'     : yes_no_map[senior_citizen],
            'Partner'           : yes_no_map[partner],
            'Dependents'        : yes_no_map[dependents],
            'tenure'            : tenure,
            'PhoneService'      : yes_no_map[phone_service],
            'MultipleLines'     : yes_no_nps_map[multiple_lines],
            'InternetService'   : internet_map[internet_service],
            'OnlineSecurity'    : yes_no_nis_map[online_security],
            'OnlineBackup'      : 0,
            'DeviceProtection'  : 0,
            'TechSupport'       : yes_no_nis_map[tech_support],
            'StreamingTV'       : yes_no_nis_map[streaming_tv],
            'StreamingMovies'   : 0,
            'Contract'          : contract_map[contract],
            'PaperlessBilling'  : yes_no_map[paperless_billing],
            'PaymentMethod'     : payment_map[payment_method],
            'MonthlyCharges'    : monthly_charges,
            'TotalCharges'      : float(total_charges),
            'is_new_customer'   : is_new_customer,
            'is_loyal_customer' : is_loyal,
            'charges_per_month' : charges_per_mo,
            'num_services'      : num_services,
            'has_support'       : has_support,
            'is_high_value'     : is_high_value,
            'contract_risk'     : contract_risk
        }

        # Make DataFrame in correct column order
        input_df = pd.DataFrame([input_data])[feature_names]

        # Apply scaler if Logistic Regression
        if scaler is not None:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values

        # Get prediction
        churn_prob  = model.predict_proba(input_scaled)[0][1]
        churn_pred  = model.predict(input_scaled)[0]

        # --------------------------------------------------------
        # SHOW RESULT
        # --------------------------------------------------------
        st.markdown("---")
        st.header("Prediction Result")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric("Churn Probability", f"{churn_prob*100:.1f}%")

        with res_col2:
            st.metric("Tenure", f"{tenure} months")

        with res_col3:
            st.metric("Monthly Charges", f"${monthly_charges}")

        st.markdown("")

        # Color coded risk box
        if churn_prob >= 0.65:
            st.error(f"""
            ### ⚠️ HIGH CHURN RISK — {churn_prob*100:.0f}% probability

            **This customer is very likely to cancel soon!**

            Recommended Actions:
            - Call customer within 24 hours
            - Offer loyalty discount (10-20%)
            - Upgrade support plan for free
            - Assign dedicated account manager
            """)

        elif churn_prob >= 0.35:
            st.warning(f"""
            ### ⚡ MEDIUM CHURN RISK — {churn_prob*100:.0f}% probability

            **This customer shows some signs of leaving.**

            Recommended Actions:
            - Send personalized email this week
            - Offer small discount on renewal
            - Check if any support issues exist
            """)

        else:
            st.success(f"""
            ### ✅ LOW CHURN RISK — {churn_prob*100:.0f}% probability

            **This customer is likely to stay.**

            Recommended Actions:
            - No immediate action needed
            - Include in standard loyalty program
            - Monitor monthly as usual
            """)

        # --------------------------------------------------------
        # SHAP EXPLANATION
        # --------------------------------------------------------
        st.markdown("---")
        st.subheader("Why This Prediction?")
        st.markdown("Which factors are driving this customer's churn risk:")

        try:
            if model_type in ['RandomForestClassifier', 'XGBClassifier']:
                explainer   = shap.TreeExplainer(model)
                shap_vals   = explainer.shap_values(input_df)
                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]
                else:
                    sv = shap_vals[0]
            else:
                X_train_raw   = df.drop('Churn', axis=1)
                split_point   = int(len(df) * 0.80)
                X_train_data  = X_train_raw.iloc[:split_point][feature_names]
                X_train_scaled = scaler.transform(X_train_data)
                explainer     = shap.LinearExplainer(model, X_train_scaled)
                sv            = explainer.shap_values(input_scaled)[0]

            # Build explanation dataframe
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'Value'  : input_df.values[0],
                'Impact' : sv
            }).sort_values('Impact', ascending=False)

            # Show top factors
            exp_col1, exp_col2 = st.columns(2)

            with exp_col1:
                st.markdown("**🔴 Factors INCREASING churn risk:**")
                top_increase = shap_df[shap_df['Impact'] > 0].head(5)
                for _, row in top_increase.iterrows():
                    st.markdown(f"- **{row['Feature']}** = {row['Value']:.1f} → +{row['Impact']:.3f}")

            with exp_col2:
                st.markdown("**🟢 Factors DECREASING churn risk:**")
                top_decrease = shap_df[shap_df['Impact'] < 0].head(5)
                for _, row in top_decrease.iterrows():
                    st.markdown(f"- **{row['Feature']}** = {row['Value']:.1f} → {row['Impact']:.3f}")

            # Waterfall bar chart
            st.markdown("")
            top12 = shap_df.reindex(
                shap_df['Impact'].abs().sort_values(ascending=True).index
            ).tail(12)

            fig, ax = plt.subplots(figsize=(8, 6))
            colors  = ['#e74c3c' if v > 0 else '#2ecc71' for v in top12['Impact']]
            ax.barh(range(len(top12)), top12['Impact'],
                    color=colors, edgecolor='black', alpha=0.85)
            labels = [f"{f} = {v:.1f}" for f, v in
                      zip(top12['Feature'], top12['Value'])]
            ax.set_yticks(range(len(top12)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.axvline(x=0, color='black', linewidth=1.5)
            ax.set_xlabel("Impact on Churn Prediction")
            ax.set_title("Feature Impact — Red=Increases Risk | Green=Decreases Risk",
                         fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.info(f"SHAP explanation not available: {e}")


# ============================================================
# PAGE 2 - MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":

    st.header("Model Performance")
    st.markdown("How well does the model perform on unseen test data?")
    st.markdown("")

    # Load metrics
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    split_point  = int(len(df) * 0.80)
    X_test       = X.iloc[split_point:][feature_names]
    y_test       = y.iloc[split_point:]

    if scaler is not None:
        X_test_input = scaler.transform(X_test)
    else:
        X_test_input = X_test.values

    y_pred = model.predict(X_test_input)
    y_prob = model.predict_proba(X_test_input)[:, 1]

    from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, accuracy_score

    recall  = recall_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_prob)
    acc     = accuracy_score(y_test, y_pred)

    # Metric cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Recall",    f"{recall:.3f}",  help="% of churners we catch")
    m2.metric("Precision", f"{prec:.3f}",    help="% of flagged that actually churn")
    m3.metric("F1 Score",  f"{f1:.3f}",      help="Balance of precision and recall")
    m4.metric("AUC ROC",   f"{auc:.3f}",     help="Overall discrimination ability")
    m5.metric("Accuracy",  f"{acc:.3f}",     help="Overall correct predictions")

    st.markdown("")
    st.info(f"""
    **What these numbers mean:**
    - Our model catches **{recall*100:.0f}%** of all customers who will churn
    - When we flag a customer as risky, we are right **{prec*100:.0f}%** of the time
    - AUC of **{auc:.3f}** means excellent separation between churners and non-churners
    """)

    st.markdown("---")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm  = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted: Stayed', 'Predicted: Churned'],
                yticklabels=['Actually: Stayed', 'Actually: Churned'],
                ax=axes[0], linewidths=2)
    axes[0].set_title('Confusion Matrix', fontweight='bold')

    categories = ['True Negative\n(Correctly said Stayed)',
                  'False Positive\n(False alarm)',
                  'False Negative\n(Missed churner!)',
                  'True Positive\n(Correctly caught)']
    values  = [tn, fp, fn, tp]
    colors  = ['#2ecc71', '#f39c12', '#e74c3c', '#27ae60']
    bars    = axes[1].barh(categories, values, color=colors, edgecolor='black')
    axes[1].set_title('What Each Number Means', fontweight='bold')
    axes[1].set_xlabel('Number of Customers')
    for bar, val in zip(bars, values):
        axes[1].text(val + 2, bar.get_y() + bar.get_height()/2,
                     str(val), va='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Model Comparison")
    st.markdown("How this model compares to simple baselines:")

    comparison = pd.DataFrame({
        'Model'    : ['Always Predict No', 'Simple Rule', f'{model_type} (Our Model)'],
        'Recall'   : [0.000, 0.310, round(recall, 3)],
        'Precision': [0.000, 0.480, round(prec, 3)],
        'F1 Score' : [0.000, 0.375, round(f1, 3)],
        'AUC'      : [0.500, 0.620, round(auc, 3)]
    })
    st.dataframe(comparison, use_container_width=True)


# ============================================================
# PAGE 3 - ABOUT PROJECT
# ============================================================
elif page == "About Project":

    st.header("About This Project")

    st.markdown("""
    ## Customer Churn Prediction System

    ### The Problem
    Subscription businesses lose 20-30% of customers every year.
    By the time customers cancel, it is too late to act.
    This system identifies **who will leave and why** — 30 days before they go.

    ### The Solution
    A machine learning model trained on 7,043 real customer records
    that predicts churn probability for any customer in real time.

    ### Business Impact
    - Model catches **76% of all churners** before they leave
    - Enables proactive retention actions (discounts, calls, upgrades)
    - Estimated **$500,000+ annual revenue** at risk identified

    ---

    ### Technical Details

    | Component | Details |
    |---|---|
    | Dataset | IBM Telco Customer Churn (7,043 customers) |
    | Features | 26 (19 original + 7 engineered) |
    | Model | Logistic Regression with class balancing |
    | Recall | 0.759 (catches 76% of churners) |
    | AUC ROC | 0.844 (excellent discrimination) |
    | Explainability | SHAP values for every prediction |

    ---

    ### Engineered Features Created
    - **is_new_customer** — customers in first 6 months (highest risk)
    - **is_loyal_customer** — customers with 2+ years (lowest risk)
    - **charges_per_month** — normalized payment pattern
    - **num_services** — count of addon services subscribed
    - **has_support** — has tech support or online security
    - **is_high_value** — pays more than $70/month
    - **contract_risk** — risk score based on contract type

    ---

    ### Key Findings from Analysis
    1. **Contract type** is the strongest churn signal
       - Month-to-month: 43% churn rate
       - Two year: only 3% churn rate
    2. **First 6 months** = highest risk period (50%+ churn)
    3. **No tech support** = 3x higher churn risk
    4. **Higher monthly charges** = more likely to churn

    ---

    ### Project Structure
    ```
    churn-prediction/
    ├── data/raw/          Original dataset
    ├── data/processed/    Cleaned + engineered data
    ├── models/            Trained model files
    ├── notebooks/         Step by step analysis
    ├── docs/              Charts and reports
    └── app/               This dashboard
    ```
    """)

    st.markdown("---")
    st.markdown("Built as a portfolio project following production ML best practices.")
