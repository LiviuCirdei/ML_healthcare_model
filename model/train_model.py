import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

from prepare_data import load_data, preprocess_data

def train():
    """Train a logistic regression model with L1 regularization."""

    # Load and preprocess data
    df = load_data("data/train_hospital_readmissions.csv")
    X, y = preprocess_data(df, is_train=True)

    # Identify categorical columns and create preprocessor  
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough')


    # Create and train the pipeline
    # Lasso Logistic Regression
    # penalty="l1" → applies Lasso regularization (L1 norm)
    # solver="liblinear" → required for L1 penalty with logistic regression
    # LogisticRegression is a classification algorithm, even though it uses a regression-based approach behind the scenes
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000))
    ])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    y_pred = pipeline.predict(X_val)
    print("ROC-AUC:", roc_auc_score(y_val, y_proba))
    print(classification_report(y_val, y_pred))

    # Save the trained model
    joblib.dump(pipeline, "model/model.pkl")
    print("✅ Model saved to model/model.pkl")

if __name__ == "__main__":
    train()
