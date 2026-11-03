
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    auc
)

from utils import validate_transaction_data


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = os.path.join(
    "artifacts", "fraud_model.pkl"
)


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


st.sidebar.title("🛡️ FraudGuard")
st.sidebar.caption("Machine Learning Security System")

if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model not found. Run "
        "'python train_model.py' first."
    )
    st.stop()

artifact = load_artifact()

model = artifact["model"]
features = artifact["features"]
metrics = artifact["metrics"]
model_name = artifact["model_name"]



page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Fraud Prediction",
        "Batch Prediction",
        "Model Evaluation",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Academic ML Project")
st.sidebar.caption("Predictions are not confirmed fraud findings.")


def get_fraud_probability(data):
    """Return probability of the fraud class (Class = 1)."""

    probabilities = model.predict_proba(data)
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("The trained model has no fraud class.")

    fraud_index = classes.index(1)
    return probabilities[:, fraud_index]


def predict_transactions(data):
    """Predict classes and fraud probabilities."""

    valid, message = validate_transaction_data(
        data, features
    )

    if not valid:
        raise ValueError(message)

    clean_data = data[features].copy()

    # Ensure numeric values and original feature order.
    for column in features:
        clean_data[column] = pd.to_numeric(
            clean_data[column], errors="raise"
        )

    predictions = model.predict(clean_data)
    probabilities = get_fraud_probability(clean_data)

    return predictions, probabilities


def show_metrics_cards(values):
    """Display key metrics in four columns."""

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Precision",
        f"{values['precision']:.2%}"
    )
    col2.metric(
        "Recall",
        f"{values['recall']:.2%}"
    )
    col3.metric(
        "F1 Score",
        f"{values['f1_score']:.2%}"
    )
    col4.metric(
        "ROC-AUC",
        f"{values['roc_auc']:.2%}"
    )


if page == "Dashboard":

    st.title("🛡️ Credit Card Fraud Detection")
    st.write(
        "Machine learning system for identifying "
        "potentially fraudulent credit card transactions."
    )

    st.divider()

    counts = artifact["class_counts"]
    legitimate = counts.get("0", 0)
    fraud = counts.get("1", 0)
    total = legitimate + fraud

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Legitimate Transactions", f"{legitimate:,}")
    col3.metric("Fraud Transactions", f"{fraud:,}")
    col4.metric(
        "Fraud Percentage",
        f"{(fraud / total * 100) if total else 0:.3f}%"
    )

    st.subheader("Dataset Class Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        x=["Legitimate", "Fraud"],
        y=[legitimate, fraud],
        hue=["Legitimate", "Fraud"],
        legend=False,
        ax=ax
    )

    ax.set_ylabel("Number of Transactions")
    ax.set_title("Legitimate vs Fraudulent Transactions")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Model Performance")

    show_metrics_cards(metrics)

    st.info(
        f"Selected algorithm: {model_name}. "
        "Performance figures are calculated on the held-out "
        "test dataset."
    )


# ==================================================
# 2. SINGLE TRANSACTION PREDICTION
# ==================================================

elif page == "Fraud Prediction":

    st.title("🔍 Fraud Prediction")
    st.write(
        "Enter transaction features to estimate the "
        "probability of fraud."
    )

    st.warning(
        "This model expects the anonymized "
        "features from the Kaggle dataset. The inputs are "
        "not real credit card details."
    )

    with st.form("transaction_form"):

        st.subheader("Transaction Information")

        values = {}
        columns = st.columns(3)

        for index, feature in enumerate(features):

            with columns[index % 3]:

                if feature == "Amount":
                    values[feature] = st.number_input(
                        "Transaction Amount",
                        min_value=0.0,
                        value=100.0,
                        step=10.0,
                        key="amount_input"
                    )

                elif feature == "Time":
                    values[feature] = st.number_input(
                        "Transaction Time (seconds)",
                        min_value=0.0,
                        value=0.0,
                        step=1.0,
                        key="time_input"
                    )

                else:
                    values[feature] = st.number_input(
                        feature,
                        value=0.0,
                        step=0.1,
                        format="%.6f",
                        key=f"input_{feature}"
                    )

        submitted = st.form_submit_button(
            "Analyze Transaction",
            use_container_width=True
        )

    if submitted:

        transaction = pd.DataFrame(
            [[values[name] for name in features]],
            columns=features
        )

        prediction, probability = predict_transactions(
            transaction
        )

        prediction = int(prediction[0])
        fraud_probability = float(probability[0])

        st.divider()
        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric(
            "Estimated Fraud Probability",
            f"{fraud_probability:.2%}"
        )

        col2.metric(
            "Model Classification",
            "Fraud" if prediction == 1 else "Legitimate"
        )

        st.progress(fraud_probability)

        if prediction == 1:
            st.error(
                "⚠️ Potential Fraud Detected. "
                "This transaction has been classified as fraud."
            )
        else:
            st.success(
                "✅ Classified as Legitimate by the model."
            )

        st.caption(
            "A model prediction is not proof of fraud. "
            "This demonstration does not connect to a bank "
            "or authorize real transactions."
        )



elif page == "Batch Prediction":

    st.title("📂 Batch Transaction Prediction")

    st.write(
        "Upload a CSV containing multiple transactions. "
        "The application will classify each row."
    )

    st.caption(
        "Required columns are Time, Amount and V1–V28 "
        "for the standard Kaggle dataset."
    )

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            uploaded_data = pd.read_csv(uploaded_file)

            # Do not use actual target labels as model inputs.
            if "Class" in uploaded_data.columns:
                uploaded_data = uploaded_data.drop(
                    columns=["Class"]
                )

            st.subheader("Uploaded Data Preview")
            st.dataframe(
                uploaded_data.head(10),
                use_container_width=True
            )

            st.write(
                f"Rows uploaded: {len(uploaded_data):,}"
            )

            if st.button(
                "Predict All Transactions",
                use_container_width=True
            ):

                predictions, probabilities = (
                    predict_transactions(uploaded_data)
                )

                results = uploaded_data.copy()

                results["Predicted_Class"] = predictions
                results["Fraud_Probability"] = probabilities

                results["Prediction_Label"] = [
                    "Fraud" if value == 1 else "Legitimate"
                    for value in predictions
                ]

                st.subheader("Prediction Results")

                fraud_count = int(
                    (predictions == 1).sum()
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Analyzed",
                    f"{len(results):,}"
                )

                col2.metric(
                    "Potential Fraud",
                    f"{fraud_count:,}"
                )

                col3.metric(
                    "Legitimate",
                    f"{len(results) - fraud_count:,}"
                )

                st.dataframe(
                    results,
                    use_container_width=True
                )

                csv_output = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Prediction Results",
                    data=csv_output,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.success(
                    "Predictions completed. "
                    "Download the results for further analysis."
                )

        except Exception as error:
            st.error(f"Unable to process CSV: {error}")


elif page == "Model Evaluation":
    st.title("📊 Model Evaluation")

    st.write(
        "Evaluate the selected model on the unseen test set."
    )

    show_metrics_cards(metrics)

    st.divider()

    # Confusion matrix
    st.subheader("Confusion Matrix")

    cm = metrics["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)

    # Classification report
    st.subheader("Classification Report")

    report = classification_report(
        artifact["y_test"],
        artifact["y_pred"],
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )

    # Precision-recall curve
    st.subheader("Precision-Recall Curve")

    precision, recall, _ = precision_recall_curve(
        artifact["y_test"],
        artifact["y_score"]
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)

    # ROC curve
    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(
        artifact["y_test"],
        artifact["y_score"]
    )

    roc_area = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_area:.4f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)

    # Validation model comparison
    st.subheader("Algorithm Comparison")

    comparison = pd.DataFrame(
        artifact["validation_results"]
    ).T

    st.dataframe(
        comparison.round(4),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    comparison[["Precision", "Recall", "F1 Score"]].plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Validation Performance Comparison")
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", rotation=0)

    st.pyplot(fig)
    plt.close(fig)

elif page == "About Project":

    st.title("ℹ️ About the Project")

    st.subheader("Credit Card Fraud Detection System")

    st.write(
        "In this project, we used supervised machine learning "
        "to classify credit card transactions as legitimate "
        "or potentially fraudulent."
    )

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Streamlit
    - Matplotlib and Seaborn
    - Joblib
    """)

    st.subheader("Machine Learning Algorithms")

    st.markdown("""
    - Logistic Regression
    - Random Forest Classifier
    """)

    st.subheader("Project Workflow")

    st.markdown("""
    1. Load the historical dataset.
    2. Inspect and prepare transaction data.
    3. Split data into training, validation and test sets.
    4. Train and compare machine learning algorithms.
    5. Select a model using validation performance.
    6. Evaluate it on unseen test transactions.
    7. Deploy predictions through a Streamlit website.
    """)

    st.subheader("Limitations")

    st.markdown("""
    - The dataset contains anonymized features.
    - Historical performance may not reflect future transactions.
    - Fraud probability is a model estimate, not certainty.
    - The application is an academic demonstration, not a
      production banking or payment-processing system.
    """)

    st.info(
        f"Currently loaded model: {model_name}"
    )

    st.caption(
        "Credit Card Fraud Detection | Machine Learning Project"
    )
