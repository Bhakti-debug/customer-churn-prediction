import streamlit as st
import pandas as pd
import joblib
import pickle
import os

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
    }

    .result-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        margin: 1rem 0 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .result-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.45rem;
    }

    .result-status {
        font-size: 1.7rem;
        font-weight: 650;
        margin-bottom: 0.3rem;
    }

    .result-description {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 0;
    }

    .status-safe {
        color: #15803d;
    }

    .status-risk {
        color: #dc2626;
    }

    .comparison-header {
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }

    .comparison-description {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 1.2rem;
    }

    .model-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        min-height: 105px;
        box-shadow: 0 1px 5px rgba(0, 0, 0, 0.03);
    }

    .model-name {
        font-size: 0.82rem;
        color: #6b7280;
        margin-bottom: 0.45rem;
    }

    .model-result {
        font-size: 1rem;
        font-weight: 600;
    }

    .model-probability {
        font-size: 0.82rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }

    .model-safe {
        color: #15803d;
    }

    .model-churn {
        color: #dc2626;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    .sidebar-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 2.7rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def preprocess_input(input_data, scaler, feature_names):
    # Convert input data into a DataFrame
    df = pd.DataFrame([input_data])

    # Encode binary categorical features
    binary_map = {
        'Yes': 1,
        'No': 0,
        'Male': 1,
        'Female': 0
    }

    binary_cols = [
        'gender',
        'Partner',
        'Dependents',
        'PhoneService',
        'PaperlessBilling'
    ]

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    # One-hot encode multi-category features
    multi_cat_cols = [
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaymentMethod'
    ]

    df = pd.get_dummies(
        df,
        columns=multi_cat_cols
    )

    # Scale numerical features using the training scaler
    numeric_cols = [
        'tenure',
        'MonthlyCharges',
        'TotalCharges'
    ]

    df[numeric_cols] = scaler.transform(
        df[numeric_cols]
    )

    # Add missing features as zero
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Match the feature order used during training
    df = df[feature_names]

    return df


def load_models():
    # Get the project directory
    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    models_dir = os.path.join(
        base_dir,
        'models'
    )

    if not os.path.exists(models_dir):
        return None, None, None, False

    try:
        # Load trained machine learning models
        models = {
            'Logistic Regression': joblib.load(
                os.path.join(
                    models_dir,
                    'logistic_regression.joblib'
                )
            ),
            'Random Forest': joblib.load(
                os.path.join(
                    models_dir,
                    'random_forest.joblib'
                )
            ),
            'SVM': joblib.load(
                os.path.join(
                    models_dir,
                    'svm.joblib'
                )
            ),
            'KNN': joblib.load(
                os.path.join(
                    models_dir,
                    'knn.joblib'
                )
            )
        }

        # Load preprocessing objects
        scaler = joblib.load(
            os.path.join(
                models_dir,
                'scaler.joblib'
            )
        )

        with open(
            os.path.join(
                models_dir,
                'feature_names.pkl'
            ),
            'rb'
        ) as f:
            feature_names = pickle.load(f)

        return models, scaler, feature_names, True

    except FileNotFoundError:
        return None, None, None, False


def main():
    st.title("Telco Customer Churn Prediction")

    st.markdown(
        '<div class="app-subtitle">'
        'Estimate customer churn risk using machine learning '
        'models based on customer account and service information.'
        '</div>',
        unsafe_allow_html=True
    )

    models, scaler, feature_names, models_loaded = load_models()

    if not models_loaded:
        st.warning(
            "Model files not found. Please make sure the "
            "`models` directory is present."
        )
        return

    # Customer information inputs
    st.sidebar.markdown(
        '<div class="sidebar-title">'
        'Customer Information'
        '</div>',
        unsafe_allow_html=True
    )

    gender = st.sidebar.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.sidebar.checkbox(
        "Senior Citizen"
    )

    partner = st.sidebar.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.sidebar.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    phone_service = st.sidebar.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.sidebar.selectbox(
        "Multiple Lines",
        [
            "No phone service",
            "No",
            "Yes"
        ]
    )

    internet_service = st.sidebar.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    online_security = st.sidebar.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    online_backup = st.sidebar.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    device_protection = st.sidebar.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    tech_support = st.sidebar.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_tv = st.sidebar.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_movies = st.sidebar.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    contract = st.sidebar.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.sidebar.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        '<div class="sidebar-title">'
        'Account Details'
        '</div>',
        unsafe_allow_html=True
    )

    tenure = st.sidebar.slider(
        "Tenure (months)",
        0,
        72,
        1
    )

    monthly_charges = st.sidebar.slider(
        "Monthly Charges ($)",
        0.0,
        120.0,
        50.0
    )

    total_charges = st.sidebar.slider(
        "Total Charges ($)",
        0.0,
        10000.0,
        50.0
    )

    # Prepare the customer's input data
    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    st.subheader("Prediction")

    selected_model_name = st.selectbox(
        "Select Model",
        list(models.keys())
    )

    if st.button(
        "Predict Churn Risk",
        type="primary"
    ):
        # Preprocess the customer input
        df_processed = preprocess_input(
            input_data,
            scaler,
            feature_names
        )

        selected_model = models[selected_model_name]

        # Generate prediction
        prediction = selected_model.predict(
            df_processed
        )[0]

        has_proba = hasattr(
            selected_model,
            "predict_proba"
        )

        if has_proba:
            proba = selected_model.predict_proba(
                df_processed
            )[0]

            churn_prob = proba[1]

        # Display the prediction result
        if prediction == 1:
            status_class = "status-risk"
            status_text = "LIKELY TO CHURN"
            description = (
                "The model predicts that this customer "
                "is at risk of leaving."
            )
        else:
            status_class = "status-safe"
            status_text = "LIKELY TO STAY"
            description = (
                "The model predicts that this customer "
                "is likely to remain with the service."
            )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    Prediction Result
                </div>
                <div class="result-status {status_class}">
                    {status_text}
                </div>
                <p class="result-description">
                    {description}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display prediction metrics
        if has_proba:
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Churn Probability",
                    f"{churn_prob:.1%}"
                )

            with col2:
                st.metric(
                    "Selected Model",
                    selected_model_name
                )

        # Compare predictions from all models
        st.markdown(
            '<div class="comparison-header">'
            '<h3>Model Comparison</h3>'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="comparison-description">'
            'Prediction results from all trained models for the same customer.'
            '</div>',
            unsafe_allow_html=True
        )

        cols = st.columns(len(models))

        for col, (name, model) in zip(
            cols,
            models.items()
        ):
            pred = model.predict(
                df_processed
            )[0]

            if pred == 1:
                status = "Churn"
                status_class = "model-churn"
            else:
                status = "Stay"
                status_class = "model-safe"

            probability_text = ""

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(
                    df_processed
                )[0][1]

                probability_text = (
                    f"{prob:.1%} churn probability"
                )

            with col:
                st.markdown(
                    f"""
                    <div class="model-card">
                        <div class="model-name">
                            {name}
                        </div>
                        <div class="model-result {status_class}">
                            {status}
                        </div>
                        <div class="model-probability">
                            {probability_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.divider()

    st.caption(
        "Customer Churn Prediction • Machine Learning Project"
    )


if __name__ == "__main__":
    main()