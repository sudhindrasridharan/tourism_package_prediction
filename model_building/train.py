
import os
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ---------------------------------------------------------
# 1. Define paths
# ---------------------------------------------------------

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "train.csv"
)

TEST_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "test.csv"
)

DEPLOYMENT_DIR = os.path.join(
    PROJECT_DIR,
    "deployment"
)

MODEL_PATH = os.path.join(
    DEPLOYMENT_DIR,
    "best_model.pkl"
)


# ---------------------------------------------------------
# 2. Load train and test data
# ---------------------------------------------------------

print("Loading training and testing data...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)


# ---------------------------------------------------------
# 3. Separate features and target
# ---------------------------------------------------------

TARGET = "ProdTaken"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


# ---------------------------------------------------------
# 4. Identify numerical and categorical columns
# ---------------------------------------------------------

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ---------------------------------------------------------
# 5. Create preprocessing pipelines
# ---------------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# ---------------------------------------------------------
# 6. Define model
# ---------------------------------------------------------

model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)


# ---------------------------------------------------------
# 7. Create complete ML pipeline
# ---------------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ]
)


# ---------------------------------------------------------
# 8. Define hyperparameters for tuning
# ---------------------------------------------------------

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}


# ---------------------------------------------------------
# 9. Start MLflow experiment
# ---------------------------------------------------------

mlflow.set_experiment("Tourism_Package_Prediction")


with mlflow.start_run():

    print("\nStarting hyperparameter tuning...")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)


    # -----------------------------------------------------
    # 10. Get best model and parameters
    # -----------------------------------------------------

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("\nBest Parameters:")
    print(best_params)

    print("\nBest Cross-Validation Score:")
    print(grid_search.best_score_)


    # -----------------------------------------------------
    # 11. Log tuned parameters
    # -----------------------------------------------------

    for parameter, value in best_params.items():
        mlflow.log_param(parameter, value)


    mlflow.log_metric(
        "best_cv_f1_score",
        grid_search.best_score_
    )


    # -----------------------------------------------------
    # 12. Evaluate model on test data
    # -----------------------------------------------------

    y_pred = best_model.predict(X_test)

    y_probability = best_model.predict_proba(X_test)[:, 1]


    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )


    # -----------------------------------------------------
    # 13. Log evaluation metrics
    # -----------------------------------------------------

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)
    mlflow.log_metric("test_roc_auc", roc_auc)


    # -----------------------------------------------------
    # 14. Display evaluation results
    # -----------------------------------------------------

    print("\n========== MODEL PERFORMANCE ==========")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")


    # -----------------------------------------------------
    # 15. Log best model in MLflow
    # -----------------------------------------------------

    mlflow.sklearn.log_model(
        best_model,
        "best_model"
    )


    # -----------------------------------------------------
    # 16. Save best model for deployment
    # -----------------------------------------------------

    os.makedirs(DEPLOYMENT_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(best_model, file)


    print("\nBest model saved successfully.")
    print("Model path:", MODEL_PATH)
