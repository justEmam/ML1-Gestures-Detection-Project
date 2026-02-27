import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH      = "data/hand_landmarks_data.csv"
MODELS_DIR     = "models"
MLFLOW_EXP     = "Hand Gesture Recognition"
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
VAL_TEST_SPLIT = 0.5

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path)
    print(f"Data loaded: {df.shape}")
    print(f"Classes: {df.iloc[:, -1].nunique()}")
    return df

# ── Log Data Info ─────────────────────────────────────────────────────────────
def log_data_info(df):
    mlflow.log_param("n_samples",  len(df))
    mlflow.log_param("n_features", df.shape[1] - 1)
    mlflow.log_param("n_classes",  df.iloc[:, -1].nunique())

    label_col    = df.columns[-1]
    class_counts = df[label_col].value_counts()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis', ax=ax)
    ax.set_title('Class Distribution')
    ax.set_xlabel('Gesture')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    mlflow.log_artifact("class_distribution.png")
    plt.close()

# ── Geometric Normalization ───────────────────────────────────────────────────
def geometric_normalization(df):
    x_cols = [col for col in df.columns if col.startswith('x')]
    y_cols = [col for col in df.columns if col.startswith('y')]

    wrist_x = df['x1'].copy()
    wrist_y = df['y1'].copy()

    df[x_cols] = df[x_cols].sub(wrist_x, axis=0)
    df[y_cols] = df[y_cols].sub(wrist_y, axis=0)

    scale = np.sqrt(df['x13']**2 + df['y13']**2).replace(0, 1.0)

    df[x_cols] = df[x_cols].div(scale, axis=0)
    df[y_cols] = df[y_cols].div(scale, axis=0)

    print(f"NaN count: {df.isna().sum().sum()}")
    print(f"Inf count: {np.isinf(df[x_cols + y_cols].values).sum()}")
    return df

# ── Split Data ────────────────────────────────────────────────────────────────
def split_data(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_valtest, y_train, y_valtest = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    X_val, X_test, y_val, y_test = train_test_split(
        X_valtest, y_valtest, test_size=VAL_TEST_SPLIT,
        random_state=RANDOM_STATE, stratify=y_valtest)

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    X_trainval = np.concatenate([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    test_fold  = np.concatenate([np.full(len(X_train), -1), np.zeros(len(X_val))])
    ps         = PredefinedSplit(test_fold)

    return X_train, X_val, X_test, y_train, y_val, y_test, X_trainval, y_trainval, ps

# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate_model(name, model, X_test, y_test):
    test_pred = model.predict(X_test)
    results = {
        'Model':          name,
        'Val Accuracy':   model.best_score_,
        'Test Accuracy':  accuracy_score(y_test, test_pred),
        'Test Precision': precision_score(y_test, test_pred, average='weighted', zero_division=0),
        'Test Recall':    recall_score(y_test, test_pred, average='weighted', zero_division=0),
        'Test F1':        f1_score(y_test, test_pred, average='weighted', zero_division=0)
    }
    return results, test_pred

# ── Confusion Matrix ──────────────────────────────────────────────────────────
def plot_confusion_matrix(name, y_test, test_pred):
    cm = confusion_matrix(y_test, test_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    path = f"confusion_matrix_{name}.png"
    plt.savefig(path)
    mlflow.log_artifact(path)
    plt.close()

# ── Performance Bar Chart ─────────────────────────────────────────────────────
def plot_performance(results_df):
    metrics = ['Val Accuracy', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1']
    x       = np.arange(len(metrics))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (model_name, row) in enumerate(results_df.iterrows()):
        ax.bar(x + i * width, row[metrics].values, width, label=model_name)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.legend()
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    mlflow.log_artifact("model_comparison.png")
    plt.close()

# ── Train & Log One Model ─────────────────────────────────────────────────────
def train_and_log(name, estimator, param_grid, X_trainval, y_trainval,
                  X_test, y_test, ps):

    with mlflow.start_run(run_name=name, nested=True):

        # Log hyperparameter search space
        mlflow.log_params({f"grid_{k}": str(v) for k, v in param_grid.items()})

        # Train
        gs = GridSearchCV(estimator, param_grid, cv=ps,
                          scoring='accuracy', n_jobs=-1, verbose=1)
        gs.fit(X_trainval, y_trainval)

        # Log best params found
        mlflow.log_params(gs.best_params_)

        # Evaluate
        results, test_pred = evaluate_model(name, gs, X_test, y_test)

        # Log metrics
        mlflow.log_metric("val_accuracy",   results['Val Accuracy'])
        mlflow.log_metric("test_accuracy",  results['Test Accuracy'])
        mlflow.log_metric("test_precision", results['Test Precision'])
        mlflow.log_metric("test_recall",    results['Test Recall'])
        mlflow.log_metric("test_f1",        results['Test F1'])

        # Log confusion matrix artifact
        plot_confusion_matrix(name, y_test, test_pred)

        # Log classification report as text artifact
        report      = classification_report(y_test, test_pred, zero_division=0)
        report_path = f"classification_report_{name}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Save model pkl locally
        model_path = f"{MODELS_DIR}/{name}_model.pkl"
        joblib.dump(gs.best_estimator_, model_path)

        # Log model to MLFlow — each model registered separately so you can compare in UI
        mlflow.sklearn.log_model(
            sk_model        = gs.best_estimator_,
            artifact_path   = f"model",
            registered_model_name = f"gesture_{name}"  # shows up in MLFlow Model Registry
        )
        mlflow.log_artifact(model_path)

        print(f"\n{name} Best Params:  {gs.best_params_}")
        print(f"{name} Val Accuracy:  {results['Val Accuracy']:.4f}")
        print(f"{name} Test Accuracy: {results['Test Accuracy']:.4f}")

        return gs, results, test_pred

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    mlflow.set_experiment(MLFLOW_EXP)

    # Load & normalize
    df = load_data(DATA_PATH)

    # Create MLFlow dataset object — shows beside run name in UI
    dataset = mlflow.data.from_pandas(
        df,
        source=DATA_PATH,
        name="hand_landmarks",
        targets=df.columns[-1]
    )

    # Normalize
    df = geometric_normalization(df)

    # Split
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     X_trainval, y_trainval, ps) = split_data(df)

    # Define models
    models = {
        'SVC': (
            SVC(),
            {'kernel': ['linear', 'rbf'], 'C': [1, 10, 20, 50, 130]}
        ),
        'LogisticRegression': (
            LogisticRegression(max_iter=10000),
            {'C': [0.01, 0.1, 1, 10, 100]}
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
        )
    }

    results_list = []
    predictions  = {}

    for name, (estimator, param_grid) in models.items():
        print(f"\n{'='*40}\nTraining {name}\n{'='*40}")

        with mlflow.start_run(run_name=name):

            # Log dataset — appears beside run name in UI
            mlflow.log_input(dataset, context="training")

            # Log pipeline config
            mlflow.log_params({
                "test_size":      TEST_SIZE,
                "val_test_split": VAL_TEST_SPLIT,
                "random_state":   RANDOM_STATE,
                "normalization":  "geometric_euclidean",
                "n_samples":      len(df),
                "n_features":     df.shape[1] - 1,
                "n_classes":      df.iloc[:, -1].nunique()
            })

            # Log hyperparameter search space
            mlflow.log_params({f"grid_{k}": str(v) for k, v in param_grid.items()})

            # Train
            gs = GridSearchCV(estimator, param_grid, cv=ps,
                              scoring='accuracy', n_jobs=-1, verbose=1)
            gs.fit(X_trainval, y_trainval)

            # Log best params
            mlflow.log_params(gs.best_params_)

            # Evaluate
            results, test_pred = evaluate_model(name, gs, X_test, y_test)
            results_list.append(results)
            predictions[name] = test_pred

            # Log metrics
            mlflow.log_metric("val_accuracy",   results['Val Accuracy'])
            mlflow.log_metric("test_accuracy",  results['Test Accuracy'])
            mlflow.log_metric("test_precision", results['Test Precision'])
            mlflow.log_metric("test_recall",    results['Test Recall'])
            mlflow.log_metric("test_f1",        results['Test F1'])

            # Artifacts
            plot_confusion_matrix(name, y_test, test_pred)

            report_path = f"classification_report_{name}.txt"
            with open(report_path, 'w') as f:
                f.write(classification_report(y_test, test_pred, zero_division=0))
            mlflow.log_artifact(report_path)

            # Save and log model
            model_path = f"{MODELS_DIR}/{name}_model.pkl"
            joblib.dump(gs.best_estimator_, model_path)
            mlflow.sklearn.log_model(
                sk_model=gs.best_estimator_,
                artifact_path="model",
                registered_model_name=f"gesture_{name}"
            )
            mlflow.log_artifact(model_path)

            print(f"{name} Best Params:  {gs.best_params_}")
            print(f"{name} Val Accuracy:  {results['Val Accuracy']:.4f}")
            print(f"{name} Test Accuracy: {results['Test Accuracy']:.4f}")



    print("\nDone. Run `mlflow ui` to compare runs.")

if __name__ == "__main__":
    main()