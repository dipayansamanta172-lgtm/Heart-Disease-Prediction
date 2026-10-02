import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from imblearn.over_sampling import SMOTE

DATA_FILE = "heart.csv"
GRAPHS_DIR = "graphs"
OUTPUT_DIR = "output"
TARGET_COLUMN = "Heart Disease Status"
RANDOM_STATE = 42
TEST_SIZE = 0.2

sns.set_style("whitegrid")

def create_folders():
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_dataset(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: '{path}' not found. Place heart.csv in the project folder.")
        sys.exit(1)
    except Exception as e:
        print(f"Error while loading dataset: {e}")
        sys.exit(1)

def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(os.path.join(GRAPHS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)

def inspect_dataset(df):
    print("Dataset Shape:", df.shape)
    print("\nFirst Five Rows:")
    print(df.head())
    print("\nDataset Information:")
    df.info()
    print("\nData Types:")
    print(df.dtypes)

def check_data_quality(df):
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())

def handle_missing_values(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if TARGET_COLUMN in categorical_cols:
        categorical_cols.remove(TARGET_COLUMN)

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

def encode_features(df):
    df = df.copy()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if TARGET_COLUMN in categorical_cols:
        categorical_cols.remove(TARGET_COLUMN)

    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))

    return df

def encode_target(df):
    df = df.copy()
    encoder = LabelEncoder()
    df[TARGET_COLUMN] = encoder.fit_transform(df[TARGET_COLUMN])
    return df

def plot_eda_collage(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Exploratory Data Analysis of Heart Disease Dataset", fontsize=18, fontweight="bold"
    )

    sns.histplot(df["Age"], kde=True, color="#4C72B0", ax=axes[0, 0])
    axes[0, 0].set_title("Age Distribution")
    axes[0, 0].set_xlabel("Age (years)")
    axes[0, 0].set_ylabel("Frequency")

    sns.histplot(df["Blood Pressure"], kde=True, color="#DD8452", ax=axes[0, 1])
    axes[0, 1].set_title("Blood Pressure Distribution")
    axes[0, 1].set_xlabel("Blood Pressure (mmHg)")
    axes[0, 1].set_ylabel("Frequency")

    sns.histplot(df["Cholesterol Level"], kde=True, color="#55A868", ax=axes[0, 2])
    axes[0, 2].set_title("Cholesterol Level Distribution")
    axes[0, 2].set_xlabel("Cholesterol Level (mg/dL)")
    axes[0, 2].set_ylabel("Frequency")

    sns.histplot(df["BMI"], kde=True, color="#C44E52", ax=axes[1, 0])
    axes[1, 0].set_title("BMI Distribution")
    axes[1, 0].set_xlabel("BMI")
    axes[1, 0].set_ylabel("Frequency")

    exercise_counts = df["Exercise Habits"].value_counts()
    sns.barplot(x=exercise_counts.index, y=exercise_counts.values, hue=exercise_counts.index,
                palette="viridis", legend=False, ax=axes[1, 1])
    axes[1, 1].set_title("Exercise Habits")
    axes[1, 1].set_xlabel("Exercise Habit Level")
    axes[1, 1].set_ylabel("Count")

    smoking_counts = df["Smoking"].value_counts()
    axes[1, 2].pie(smoking_counts.values, labels=smoking_counts.index, autopct="%1.1f%%",
                   colors=["#8172B2", "#CCB974"], startangle=90,
                   wedgeprops={"edgecolor": "white", "linewidth": 1})
    axes[1, 2].set_title("Smoking Status")

    save_figure(fig, "figure1_eda_collage.png")

def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 6))
    counts = df[TARGET_COLUMN].value_counts()

    bars = ax.bar(
        counts.index.astype(str), counts.values, color=["#4C72B0", "#C44E52"], edgecolor="black"
    )
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontweight="bold")

    ax.set_title("Dataset Class Distribution Before SMOTE")
    ax.set_xlabel("Heart Disease Status")
    ax.set_ylabel("Number of Samples")

    save_figure(fig, "figure2_class_distribution_before_smote.png")

def plot_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    if missing.empty:
        ax.text(0.5, 0.5, "No missing values found", ha="center", va="center", fontsize=14)
        ax.axis("off")
    else:
        ax.bar(missing.index, missing.values, color="#DD8452", edgecolor="black")
        ax.set_xticks(range(len(missing.index)))
        ax.set_xticklabels(missing.index, rotation=45, ha="right")
        ax.set_ylabel("Number of Missing Values")
        ax.set_xlabel("Column Name")

    ax.set_title("Missing Values per Column")
    save_figure(fig, "figure3_missing_values.png")

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, target_names=["No", "Yes"]),
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }
    return metrics

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    save_figure(fig, "figure4_confusion_matrix.png")

def plot_roc_curve(y_test, y_pred_proba, roc_auc):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#C44E52", linewidth=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5, label="Random Guess")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Random Forest")
    ax.legend(loc="lower right")

    save_figure(fig, "figure5_roc_curve.png")

def plot_performance_comparison(metrics):
    names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        names, values, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"], edgecolor="black"
    )

    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.3f}", xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontweight="bold")

    ax.set_ylim([0, 1.1])
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")

    save_figure(fig, "figure6_model_performance_comparison.png")

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
                linewidths=0.5, annot_kws={"size": 7}, cbar_kws={"shrink": 0.8}, ax=ax)

    ax.set_title("Correlation Heatmap (Numerical Features)", fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    save_figure(fig, "figure7_correlation_heatmap.png")

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.35)))
    ax.barh(
        importance_df["Feature"], importance_df["Importance"], color="#4C72B0", edgecolor="black"
    )

    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest Feature Importance")

    save_figure(fig, "figure8_feature_importance.png")

def save_results(metrics):
    path = os.path.join(OUTPUT_DIR, "results.txt")
    with open(path, "w") as f:
        f.write("Model Name: Random Forest\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        f.write(f"ROC-AUC Score: {metrics['roc_auc']:.4f}\n")

def main():
    create_folders()

    raw_df = load_dataset(DATA_FILE)

    inspect_dataset(raw_df)
    check_data_quality(raw_df)

    cleaned_df = handle_missing_values(raw_df)

    plot_eda_collage(cleaned_df)
    plot_class_distribution(cleaned_df)
    plot_missing_values(raw_df)

    encoded_df = encode_features(cleaned_df)
    encoded_df = encode_target(encoded_df)

    X = encoded_df.drop(columns=[TARGET_COLUMN])
    y = encoded_df[TARGET_COLUMN]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining set size before SMOTE:", X_train_scaled.shape[0])

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    print("Training set size after SMOTE:", X_train_balanced.shape[0])

    model = train_random_forest(X_train_balanced, y_train_balanced)
    metrics = evaluate_model(model, X_test_scaled, y_test)

    print("\nAccuracy:", round(metrics["accuracy"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1 Score:", round(metrics["f1_score"], 4))
    print("ROC-AUC Score:", round(metrics["roc_auc"], 4))
    print("\nClassification Report:")
    print(metrics["classification_report"])
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

    plot_confusion_matrix(metrics["confusion_matrix"])
    plot_roc_curve(y_test, metrics["y_pred_proba"], metrics["roc_auc"])
    plot_performance_comparison(metrics)
    plot_correlation_heatmap(encoded_df)
    plot_feature_importance(model, feature_names)

    save_results(metrics)

    print("\nProgram executed successfully.")

if __name__ == "__main__":
    main()