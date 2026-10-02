import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")



os.makedirs("graphs", exist_ok=True)
os.makedirs("output", exist_ok=True)



def load_data(filepath="heart.csv"):
    if not os.path.exists(filepath):
        print(f"Error: '{filepath}' not found. Place heart.csv in the same folder.")
        sys.exit(1)
    df = pd.read_csv(filepath)
    return df




def inspect_data(df):
    print("Dataset Shape:", df.shape)
    print("\nFirst Five Rows:")
    print(df.head())
    print("\nDataset Information:")
    print(df.info())
    print("\nData Types:")
    print(df.dtypes)




def clean_data(df):
    missing = df.isnull().sum()
    print("\nMissing Values Per Column:")
    print(missing[missing > 0])

    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")

    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Dropped {duplicates} duplicate rows.")

    return df




def impute_missing(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    
    target = "Heart Disease Status"
    if target in cat_cols:
        cat_cols.remove(target)
    if target in num_cols:
        num_cols.remove(target)

    if num_cols:
        num_imputer = SimpleImputer(strategy="mean")
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    return df




def encode_features(df):
    label_encoders = {}
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders




def plot_eda_collage(df_raw):
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Exploratory Data Analysis of Heart Disease Dataset", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig)

    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df_raw["Age"].dropna(), bins=30, color="#4C72B0", edgecolor="black")
    ax1.set_title("Age Distribution")
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Frequency")

    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df_raw["Blood Pressure"].dropna(), bins=30, color="#DD8452", edgecolor="black")
    ax2.set_title("Blood Pressure Distribution")
    ax2.set_xlabel("Blood Pressure")
    ax2.set_ylabel("Frequency")

    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df_raw["Cholesterol Level"].dropna(), bins=30, color="#55A868", edgecolor="black")
    ax3.set_title("Cholesterol Level Distribution")
    ax3.set_xlabel("Cholesterol Level")
    ax3.set_ylabel("Frequency")

    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(df_raw["BMI"].dropna(), bins=30, color="#C44E52", edgecolor="black")
    ax4.set_title("BMI Distribution")
    ax4.set_xlabel("BMI")
    ax4.set_ylabel("Frequency")

    
    ax5 = fig.add_subplot(gs[1, 1])
    exercise_counts = df_raw["Exercise Habits"].value_counts()
    ax5.bar(exercise_counts.index, exercise_counts.values, color="#8172B2", edgecolor="black")
    ax5.set_title("Exercise Habits")
    ax5.set_xlabel("Exercise Level")
    ax5.set_ylabel("Count")

    
    ax6 = fig.add_subplot(gs[1, 2])
    smoking_counts = df_raw["Smoking"].value_counts()
    ax6.pie(
        smoking_counts.values,
        labels=smoking_counts.index,
        autopct="%1.1f%%",
        colors=["#64B5CD", "#DA8BC3"],
        startangle=90
    )
    ax6.set_title("Smoking Status")

    plt.tight_layout()
    fig.savefig("graphs/figure1_eda_collage.png", dpi=300)
    plt.close(fig)
    print("Saved: graphs/figure1_eda_collage.png")


def plot_class_distribution(df_raw):
    
    fig, ax = plt.subplots(figsize=(7, 5))
    class_counts = df_raw["Heart Disease Status"].value_counts()
    colors = ["#4C72B0", "#DD8452"]
    ax.bar(class_counts.index, class_counts.values, color=colors, edgecolor="black")
    ax.set_title("Dataset Class Distribution Before SMOTE")
    ax.set_xlabel("Heart Disease Status")
    ax.set_ylabel("Count")
    for i, (label, val) in enumerate(class_counts.items()):
        ax.text(i, val + 40, str(val), ha="center", fontweight="bold")
    plt.tight_layout()
    fig.savefig("graphs/figure2_class_distribution.png", dpi=300)
    plt.close(fig)
    print("Saved: graphs/figure2_class_distribution.png")


def plot_missing_values(df_raw):
    
    missing = df_raw.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    if missing.empty:
        ax.text(0.5, 0.5, "No Missing Values Found", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        ax.set_title("Missing Values Per Column")
    else:
        ax.bar(missing.index, missing.values, color="#C44E52", edgecolor="black")
        ax.set_title("Missing Values Per Column")
        ax.set_xlabel("Column")
        ax.set_ylabel("Missing Count")
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig("graphs/figure3_missing_values.png", dpi=300)
    plt.close(fig)
    print("Saved: graphs/figure3_missing_values.png")




def plot_confusion_matrix(y_test, y_pred, class_names):
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    fig.savefig("graphs/figure4_confusion_matrix.png", dpi=300)
    plt.close(fig)
    print("Saved: graphs/figure4_confusion_matrix.png")


def plot_roc_curve(y_test, y_prob):
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC Curve (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    fig.savefig("graphs/figure5_roc_curve.png", dpi=300)
    plt.close(fig)
    print("Saved: graphs/figure5_roc_curve.png")


def plot_model_performance(metrics_dict):
    
    keys = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values = [metrics_dict[k] for k in keys]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(keys, values, color=colors, edgecolor="black")
    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    plt.tight_layout()
    fig.savefig("graphs/figure6_model_performance.png", dpi=300)
    plt.close(fig)
    print("Saved: graphs/figure6_model_performance.png")




def plot_correlation_heatmap(df_encoded):
    
    num_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df_encoded[num_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.3,
        annot_kws={"size": 7},
        ax=ax
    )
    ax.set_title("Correlation Heatmap (Numerical Features)")
    plt.tight_layout()
    fig.savefig("graphs/figure7_correlation_heatmap.png", dpi=300)
    plt.close(fig)
    print("Saved: graphs/figure7_correlation_heatmap.png")


def plot_feature_importance(model, feature_names):
    
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feat_df)))
    ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors, edgecolor="black")
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig("graphs/figure8_feature_importance.png", dpi=300)
    plt.close(fig)
    print("Saved: graphs/figure8_feature_importance.png")




def save_results(metrics_dict):
    lines = [
        "XGBClassifier",
        "",
        f"Accuracy: {metrics_dict['Accuracy']:.4f}",
        "",
        f"Precision: {metrics_dict['Precision']:.4f}",
        "",
        f"Recall: {metrics_dict['Recall']:.4f}",
        "",
        f"F1 Score: {metrics_dict['F1 Score']:.4f}",
        "",
        f"ROC-AUC Score: {metrics_dict['ROC-AUC Score']:.4f}",
    ]
    with open("output/results.txt", "w") as f:
        f.write("\n".join(lines))
    print("Saved: output/results.txt")




def main():
    
    df = load_data("heart.csv")

    
    df_raw = df.copy()

    
    inspect_data(df)

    
    df = clean_data(df)
    df_raw_clean = df.copy()  

    
    plot_eda_collage(df_raw_clean)
    plot_class_distribution(df_raw_clean)
    plot_missing_values(df_raw)

    
    df = impute_missing(df)

    
    df, label_encoders = encode_features(df)

    
    plot_correlation_heatmap(df)

    
    target_col = "Heart Disease Status"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_names = X.columns.tolist()

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    print(f"\nAfter SMOTE - Training set size: {X_train_resampled.shape[0]}")
    print(f"Class distribution after SMOTE: {np.bincount(y_train_resampled)}")

    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_resampled, y_train_resampled)

    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    metrics_dict = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC Score": roc_auc,
    }

    
    target_le = label_encoders.get(target_col)
    if target_le is not None:
        class_names = list(target_le.classes_)
    else:
        class_names = ["0", "1"]

    
    plot_confusion_matrix(y_test, y_pred, class_names)
    plot_roc_curve(y_test, y_prob)
    plot_model_performance(metrics_dict)

    
    plot_feature_importance(model, feature_names)

    
    save_results(metrics_dict)

    print("\nAll tasks completed successfully.")


if __name__ == "__main__":
    main()
