import os
import joblib

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from prepare_data import load_data, preprocess_data

# Constants
MODEL_PATH = "model/model.pkl"
TEST_DATA_PATH = "data/test_hospital_readmissions.csv"
RESULTS_DIR = "results"


def test():
    """Test the logistic regression model on the test dataset."""
    # Load model and data
    model = joblib.load(MODEL_PATH)
    df = load_data(TEST_DATA_PATH)

    # Split features and target
    y_true = (
        df["readmitted"].map({"no": 0, "yes": 1})
        if df["readmitted"].dtype == object
        else df["readmitted"]
    )
    X = preprocess_data(df.drop(columns=["readmitted"]), is_train=False)

    # Predict
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Metrics
    print("🔍 Evaluation Metrics:")
    print(f" - Accuracy:        {accuracy_score(y_true, y_pred):.4f}")
    print(f" - Precision:       {precision_score(y_true, y_pred):.4f}")
    print(f" - Recall:          {recall_score(y_true, y_pred):.4f}")
    print(f" - F1 Score:        {f1_score(y_true, y_pred):.4f}")
    print(f" - ROC-AUC Score:   {roc_auc_score(y_true, y_proba):.4f}")
    print(f" - Log Loss:        {log_loss(y_true, y_proba):.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred))

    # Save metrics results
    evaluation_results = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
                "Log Loss",
            ],
            "Value": [
                accuracy_score(y_true, y_pred),
                precision_score(y_true, y_pred),
                recall_score(y_true, y_pred),
                f1_score(y_true, y_pred),
                roc_auc_score(y_true, y_proba),
                log_loss(y_true, y_proba),
            ],
        }
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm)
    print("🧾 Confusion Matrix:")
    print(cm_df)

    ######## SAVING RESULTS #########
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    #  Save evaluation results to CSV
    evaluation_results.to_csv("results/evaluation_results.csv", index=False)
    print("✅ Evaluation results saved as evaluation_results.csv")

    # Save confusion matrix to CSV
    cm_df.to_csv("results/confusion_matrix.csv", index=False)
    print("✅ Confusion matrix saved as confusion_matrix.csv")

    # Save predictions to csv file
    predictions = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    predictions.to_csv("results/predictions.csv", index=False)
    print("✅ Predictions saved as predictions.csv")

    # Save probabilities to csv file (with the score bertween 0 and 1)
    results_df = df.copy()
    results_df["readmission_score"] = y_proba  # probability between 0 and 1
    results_df.to_csv("results/test_predictions_with_scores.csv", index=False)
    print(
        "\n✅ Saved predicted readmission scores to health_model/results/test_predictions_with_scores.csv"
    )

    ######## VISUALIZATION #########

    # Visualize confusion matrix using Scikit-learn
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    plt.show()
    print("✅ Confusion matrix visualization saved as confusion_matrix.png")

    # Visualize ROC Curve
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title("ROC Curve")
    plt.savefig("results/roc_curve.png")
    plt.show()
    print("✅ ROC curve visualization saved as roc_curve.png")


if __name__ == "__main__":
    test()
