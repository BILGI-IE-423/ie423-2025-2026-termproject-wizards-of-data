import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
 
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    precision_recall_fscore_support
)
 
from IPython.display import display
 
# =========================================================
# LOAD SERIALIZED MODEL OUTPUTS
# =========================================================
stored_data = joblib.load("data/model_outputs.joblib")
 
y_train_before = stored_data["y_train_before"]
y_train_after = stored_data["y_train_after"]
model_results = stored_data["model_results"]
y_test = stored_data["y_test"]
 
absa_missing_series = stored_data["absa_missing_series"]
absa_flag_distributions = stored_data["absa_flag_distributions"]
 
# =========================================================
# MAIN EVALUATION FUNCTION
# =========================================================
def run_ultimate_evaluation_with_all_visuals(
    y_train_before,
    y_train_after,
    model_results,
    y_test,
    absa_missing,
    absa_flags,
    data_path="data/raw_original_data.csv"
):
 
    # =========================================================
    # VISUAL OUTPUT PATHS
    # =========================================================
    figures_dir = "visuals/figures"
    tables_dir = "visuals/tables"
 
    print("\n" + "="*60)
    print("PART 1: PROJECT SUMMARY AND PERFORMANCE TABLES")
    print("="*60)
 
    # =========================================================
    # TABLE 0
    # =========================================================
    print("\n📋 TABLE 0: DeBERTa ABSA Model Aspect Penetration & State Distribution Report")
 
    absa_report_df = pd.DataFrame({
        "Missing/API Error (Flag=0)": [
            f"{absa_flags['absa_skin_available'][0]*100:.2f}%",
            f"{absa_flags['absa_hair_available'][0]*100:.2f}%",
            f"{absa_flags['absa_product_available'][0]*100:.2f}%"
        ],
        "Verified Neutral (Flag=1)": [
            f"{absa_flags['absa_skin_available'][1]*100:.2f}%",
            f"{absa_flags['absa_hair_available'][1]*100:.2f}%",
            f"{absa_flags['absa_product_available'][1]*100:.2f}%"
        ],
        "Active Sentiment (Flag=2)": [
            f"{absa_flags['absa_skin_available'][2]*100:.2f}%",
            f"{absa_flags['absa_hair_available'][2]*100:.2f}%",
            f"{absa_flags['absa_product_available'][2]*100:.2f}%"
        ]
    },
    index=[
        "Skin Aspect",
        "Hair Aspect",
        "General Product Aspect"
    ])
 
    display(absa_report_df)
 
    absa_report_df.to_csv(
        os.path.join(tables_dir, "table_0_absa_report.csv"),
        index=True
    )
 
    # =========================================================
    # TABLE 1
    # =========================================================
    print("\n📋 TABLE 1: Dependent Variable (Target) Class Distribution Report")
 
    total_before = y_train_before.sum()
    total_after = y_train_after.sum()
 
    dist_data = {
        "Class Category": [
            "Negative/Neutral (1-2-3)",
            "Positive (4-5)",
            "Total"
        ],
        "Raw Count (Train)": [
            y_train_before[0],
            y_train_before[1],
            total_before
        ],
        "Raw Ratio": [
            f"{(y_train_before[0] / total_before) * 100:.1f}%",
            f"{(y_train_before[1] / total_before) * 100:.1f}%",
            "100%"
        ],
        "Balanced Count (Undersampling)": [
            y_train_after[0],
            y_train_after[1],
            total_after
        ],
        "Balanced Ratio": [
            f"{(y_train_after[0] / total_after) * 100:.1f}%",
            f"{(y_train_after[1] / total_after) * 100:.1f}%",
            "100%"
        ]
    }
 
    table1_df = pd.DataFrame(dist_data)
 
    display(table1_df)
 
    table1_df.to_csv(
        os.path.join(tables_dir, "table_1_class_distribution.csv"),
        index=False
    )
 
    # =========================================================
    # TABLE 2
    # =========================================================
    print("\n📋 TABLE 2: Robustness & Hyperparameter-Tuned Model Comparison Summary")
 
    summary_data = []
 
    for name, res in model_results.items():
 
        summary_data.append({
            "Model Architecture": name,
            "Balanced Accuracy": f"{res['b_acc']:.4f}",
            "Macro F1-Score": f"{res['f1']:.4f}",
            "ROC-AUC Score": f"{res['roc_auc']:.4f}",
            "Mean Train CV F1": f"{res['cv_mean']:.4f} (±{res['cv_std']:.3f})"
        })
 
    table2_df = pd.DataFrame(summary_data)
 
    display(table2_df)
 
    table2_df.to_csv(
        os.path.join(tables_dir, "table_2_model_comparison.csv"),
        index=False
    )
 
    # =========================================================
    # TABLE 3
    # =========================================================
    print("\n📋 TABLE 3: Comprehensive Class-Based Performance Report")
 
    detailed_metrics = []
 
    for name, res in model_results.items():
 
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test,
            res["preds"]
        )
 
        for class_idx, class_label in enumerate([
            "Negative/Neutral (0)",
            "Positive (1)"
        ]):
 
            detailed_metrics.append({
                "Model Architecture": name,
                "Target Class": class_label,
                "Precision": f"{precision[class_idx]:.2f}",
                "Recall": f"{recall[class_idx]:.2f}",
                "F1-Score": f"{f1[class_idx]:.2f}",
                "Test Support": support[class_idx]
            })
 
    df_table3 = pd.DataFrame(detailed_metrics).set_index([
        "Model Architecture",
        "Target Class"
    ])
 
    display(df_table3)
 
    df_table3.to_csv(
        os.path.join(tables_dir, "table_3_class_performance.csv")
    )
 
    # =========================================================
    # PART 2 - EDA VISUALS
    # =========================================================
    print("\n" + "="*60)
    print("PART 2: EXPLORATORY DATA ANALYSIS (EDA) PLOTS")
    print("="*60)
 
    df_eda = pd.read_csv(data_path)
 
    # =========================================================
    # TABLE 4 - RQ2 USER-PRODUCT ALIGNMENT ANALYSIS
    # =========================================================
    if "user_product_alignment" in df_eda.columns and "rating" in df_eda.columns:
        alignment_table = (
            df_eda
            .groupby("user_product_alignment")["rating"]
            .agg(["count", "mean"])
            .reset_index()
        )
 
        alignment_table.columns = [
            "User Product Alignment",
            "Review Count",
            "Average Rating"
        ]
 
        alignment_table["User Product Alignment"] = alignment_table["User Product Alignment"].replace({
            0: "Not Aligned",
            1: "Aligned"
        })
 
        alignment_table["Average Rating"] = alignment_table["Average Rating"].round(4)
 
        print("\n📋 TABLE 4: RQ2 User-Product Alignment Analysis")
        display(alignment_table)
 
        alignment_table.to_csv(
            os.path.join(tables_dir, "table_4_rq2_alignment_analysis.csv"),
            index=False
        )
    else:
        print("\n⚠️ TABLE 4 could not be created because 'user_product_alignment' or 'rating' column is missing.")
 
    absa_cols = [
        "absa_skin_aspect",
        "absa_hair_aspect",
        "absa_product_aspect"
    ]
 
    df_eda["general_sentiment_score"] = df_eda[
        absa_cols
    ].mean(axis=1, skipna=True)
 
    df_eda["sentiment_category"] = pd.cut(
        df_eda["general_sentiment_score"],
        bins=[-1.1, -0.1, 0.1, 1.1],
        labels=["Negative", "Neutral", "Positive"]
    ).fillna("Neutral")
 
    sns.set_theme(style="whitegrid")
 
    # =========================================================
    # G1 - CLASS BALANCE GRAPH
    # =========================================================
    plt.figure(figsize=(7, 4.5))
 
    df_before = pd.DataFrame({
        "Class": y_train_before.index,
        "Count": y_train_before.values,
        "Dataset Split": "Raw Training Set"
    })
 
    df_after = pd.DataFrame({
        "Class": y_train_after.index,
        "Count": y_train_after.values,
        "Dataset Split": "Balanced Set (Undersampled)"
    })
 
    df_plot = pd.concat([df_before, df_after]).replace({
        "Class": {
            0: "Negative/Neutral",
            1: "Positive"
        }
    })
 
    sns.barplot(
        data=df_plot,
        x="Class",
        y="Count",
        hue="Dataset Split",
        palette="Set2"
    )
 
    plt.title("Class Balance State Before/After Undersampling")
    plt.xlabel("Target Categories")
    plt.ylabel("Data Frequency")
 
    plt.tight_layout()
 
    plt.savefig(
        os.path.join(figures_dir, "01_balance.png"),
        dpi=300,
        bbox_inches="tight"
    )
 
    plt.show()
 
    # =========================================================
    # G2 - SENTIMENT DISTRIBUTION
    # =========================================================
    plt.figure(figsize=(7, 4.5))
 
    sns.countplot(
        x="target",
        hue="sentiment_category",
        data=df_eda,
        palette="RdYlGn"
    )
 
    plt.title("ABSA-Driven Sentiment Category Distribution across Targets")
    plt.xlabel("Actual Target Status (0: Neg/Neut, 1: Pos)")
    plt.ylabel("Count")
    plt.legend(title="ABSA Sentiment")
 
    plt.tight_layout()
 
    plt.savefig(
        os.path.join(figures_dir, "02_sentiment.png"),
        dpi=300,
        bbox_inches="tight"
    )
 
    plt.show()
 
    # =========================================================
    # G3 - CORRELATION MATRIX
    # =========================================================
    plt.figure(figsize=(8, 6))
 
    corr_cols = [
        "target",
        "absa_skin_aspect",
        "absa_hair_aspect",
        "absa_product_aspect",
        "review_length"
    ]
 
    initial_len = len(df_eda)
 
    corr_data = df_eda[corr_cols].dropna(
        subset=absa_cols,
        how='any'
    )
 
    retained_len = len(corr_data)
 
    rename_dict = {
        "target": "Target Status",
        "absa_skin_aspect": "Skin Sentiment",
        "absa_hair_aspect": "Hair Sentiment",
        "absa_product_aspect": "Product Sentiment",
        "review_length": "Review Length"
    }
 
    matrix_data = corr_data.rename(columns=rename_dict)
 
    sns.heatmap(
        matrix_data.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1
    )
 
    plt.title(
        f"Feature Correlation Matrix\n(Complete Cases: {retained_len}/{initial_len} Rows)",
        fontsize=12
    )
 
    plt.tight_layout()
 
    plt.savefig(
        os.path.join(figures_dir, "04_corr.png"),
        dpi=300,
        bbox_inches="tight"
    )
 
    plt.show()
 
    # =========================================================
    # PART 3 - MODEL PERFORMANCE PANEL
    # =========================================================
    print("\n" + "="*60)
    print("PART 3: TRIPLE MODEL PERFORMANCE COMPARISON PANEL")
    print("="*60)
 
    fig = plt.figure(figsize=(18, 11))
 
    colors = {
        "Logistic Regression": "#1f77b4",
        "Random Forest": "#ff7f0e",
        "SVM (Linear)": "#2ca02c"
    }
 
    active_ml_models = {
        k: v for k, v in model_results.items()
        if k != "Majority Baseline"
    }
 
    # =========================================================
    # ROC CURVES
    # =========================================================
    ax_roc = plt.subplot2grid((2, 6), (0, 0), colspan=3)
 
    for name, res in active_ml_models.items():
 
        fpr_m, tpr_m, _ = roc_curve(
            y_test,
            res["probs"]
        )
 
        ax_roc.plot(
            fpr_m,
            tpr_m,
            label=f'{name} (AUC = {res["roc_auc"]:.4f})',
            color=colors[name],
            lw=2
        )
 
    ax_roc.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Random baseline (0.50)"
    )
 
    ax_roc.set_title(
        "Receiver Operating Characteristic (ROC) Curves",
        fontsize=12
    )
 
    ax_roc.set_xlabel("False Positive Rate (FPR)")
    ax_roc.set_ylabel("True Positive Rate (TPR)")
    ax_roc.legend()
 
    # =========================================================
    # PR CURVES
    # =========================================================
    ax_pr = plt.subplot2grid((2, 6), (0, 3), colspan=3)
 
    for name, res in active_ml_models.items():
 
        precision_m, recall_m, _ = precision_recall_curve(
            y_test,
            res["probs"]
        )
 
        ax_pr.plot(
            recall_m,
            precision_m,
            label=f'{name} (F1 = {res["f1"]:.4f})',
            color=colors[name],
            lw=2
        )
 
    ax_pr.set_title(
        "Precision-Recall (PR) Curves",
        fontsize=12
    )
 
    ax_pr.set_xlabel("Recall (Sensitivity)")
    ax_pr.set_ylabel("Precision (PPV)")
    ax_pr.legend()
 
    # =========================================================
    # CONFUSION MATRICES
    # =========================================================
    col_idx = 0
 
    for name in active_ml_models.keys():
 
        res = active_ml_models[name]
 
        ax_cm = plt.subplot2grid(
            (2, 6),
            (1, col_idx * 2),
            colspan=2
        )
 
        cm_m = confusion_matrix(
            y_test,
            res["preds"]
        )
 
        sns.heatmap(
            cm_m,
            annot=True,
            fmt="d",
            cmap="Purples",
            ax=ax_cm,
            cbar=False
        )
 
        ax_cm.set_title(f"{name} Confusion Matrix")
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Label")
 
        col_idx += 1
 
    plt.tight_layout()
 
    plt.savefig(
        os.path.join(figures_dir, "03_model_comparison_panel.png"),
        dpi=300,
        bbox_inches="tight"
    )
 
    plt.show()
 
# =========================================================
# RUN EVALUATION PIPELINE
# =========================================================
run_ultimate_evaluation_with_all_visuals(
    y_train_before,
    y_train_after,
    model_results,
    y_test,
    absa_missing_series,
    absa_flag_distributions
)
