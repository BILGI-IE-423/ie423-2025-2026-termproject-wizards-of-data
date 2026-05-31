import os
import re
import numpy as np
import pandas as pd
import torch
import joblib
 
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
 
import warnings
warnings.filterwarnings("ignore")

# Transformer Environment Setup
try:
    from transformers import pipeline
    print("Hugging Face Transformers is ready.")
except ImportError:
    print("Installing Transformers package... Please wait.")
    os.system("pip install -q transformers")
    from transformers import pipeline

# [ VALIDATION FRAMEWORK ]
# Train-Test Split with Stratification to preserve original class distribution
df_train, df_test = train_test_split(
    df_hf_processed,
    test_size=0.2,
    random_state=42,
    stratify=df_hf_processed["target"]
)
 
y_train_before = df_train["target"].value_counts().sort_index()
min_binary_size = df_train["target"].value_counts().min()
 
# [ DATA BALANCING SUITE ]
# Addressing class imbalance via dynamic majority undersampling
df_train_balanced = (
    df_train
    .groupby("target", group_keys=False)
    .apply(lambda x: x.sample(min_binary_size, random_state=42))
    .reset_index(drop=True)
)
 
y_train_after = df_train_balanced["target"].value_counts().sort_index()
 
# [ LOCAL FEATURE ALIGNMENT ]
# Syncing feature engineering engine across train/test splits
df_train_feat = engineer_features(df_train_balanced)
df_test_feat = engineer_features(df_test)
 
y_train = df_train_feat["target"].values
y_test = df_test_feat["target"].values
 
# [ MULTI-MODAL FEATURE MAPPING ]
# Mapping explicit text matrices, product metadata, and DeBERTa ABSA layers
absa_score_cols = ["absa_skin_aspect", "absa_hair_aspect", "absa_product_aspect"]
absa_flag_cols  = ["absa_skin_available", "absa_hair_available", "absa_product_available"]
other_numeric   = [
    "review_length", "price_usd",
    "is_vegan", "is_cruelty_free", "is_clean_beauty",
    "has_hyaluronic", "has_retinol", "has_niacinamide",
    "has_acids", "has_fragrance", "user_product_alignment"
]
categorical_cols = ["skin_type", "skin_tone"]
text_col = "review_text"
 
# [ PIPELINE PREPROCESSING ARCHITECTURE ]
# Assembling TF-IDF vectorization and scaling rules into an automated ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(max_features=1500, stop_words="english", dtype=np.float32), text_col),
        ("cat", Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('ohe', OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32))
        ]), categorical_cols),
      
        ("absa_scores", Pipeline([
            ('imputer', SimpleImputer(strategy="constant", fill_value=0.0)), 
            ('scaler', StandardScaler())
        ]), absa_score_cols),
        ("absa_flags", SimpleImputer(strategy="constant", fill_value=0), absa_flag_cols), 
        ("num", Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('scaler', StandardScaler())
        ]), other_numeric)
    ],
    remainder="drop"
)
 
print("\n--- Model Training Process Started (GridSearchCV & Baseline Integrated) ---")
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
 
model_results = {}
 
# [ BENCHMARK BASELINE ]
# Instantiating a Dummy Classifier for empirical performance floor validation
print(">> Training Majority Baseline Model...")
dummy_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", DummyClassifier(strategy="most_frequent"))
])
dummy_cv_scores = cross_val_score(dummy_pipe, df_train_feat, y_train, cv=cv_strategy, scoring="f1_macro", n_jobs=-1)
 
dummy_pipe.fit(df_train_feat, y_train)
dummy_preds = dummy_pipe.predict(df_test_feat)
dummy_probs = dummy_pipe.predict_proba(df_test_feat)[:, 1]
 
try:
    dummy_auc = roc_auc_score(y_test, dummy_probs)
except ValueError:
    dummy_auc = 0.5
 
model_results["Majority Baseline"] = {
    "b_acc": balanced_accuracy_score(y_test, dummy_preds),
    "f1": f1_score(y_test, dummy_preds, average="macro", zero_division=0),
    "roc_auc": dummy_auc,
    "cv_mean": dummy_cv_scores.mean(),
    "cv_std": dummy_cv_scores.std(),
    "preds": dummy_preds,
    "probs": dummy_probs
}
 
# [ MODEL CONFIGURATION & PARAMETER MATRIX ]
# Targets RQ2 & RQ3: Testing multi-family architectures (Linear, Ensemble, Margin-based)
param_grids = {
    "Helvetica/Logistic Regression": {"model__C": [0.1, 1.0, 10.0]}, # (Örnek isimlendirme korundu)
    "Logistic Regression": {"model__C": [0.1, 1.0, 10.0]},
    "Random Forest": {"model__max_depth": [5, 10, 15], "model__n_estimators": [50, 100]},
    "SVM (Linear)": {"model__estimator__C": [0.1, 0.5, 1.0]}
}
 
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "SVM (Linear)": CalibratedClassifierCV(LinearSVC(dual=False, random_state=42), cv=3)
}
 
# [ HYPERPARAMETER OPTIMIZATION LOOP ]
# Targets RQ2 & RQ3: Executes Cross-Validated Hyperparameter Search
for name, model in base_models.items():
    print(f">> Running hyperparameter tuning for {name}...")
 
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])
 
    grid_search = GridSearchCV(
        pipe,
        param_grids[name],
        cv=cv_strategy,
        scoring="f1_macro",
        n_jobs=-1
    )
 
    grid_search.fit(df_train_feat, y_train)
    best_model = grid_search.best_estimator_
 
    cv_mean = grid_search.best_score_
    cv_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
 
    preds = best_model.predict(df_test_feat)
    probs = best_model.predict_proba(df_test_feat)[:, 1]
 
    model_results[name] = {
        "b_acc": balanced_accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average="macro"),
        "roc_auc": roc_auc_score(y_test, probs),
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "preds": preds,
        "probs": probs
    }
    print(f"   Best Params: {grid_search.best_params_} | Unbiased CV F1: {cv_mean:.4f}")
 
# [ MLOPS SERIALIZATION ]
# Encapsulates all experimental configurations and arrays for down-stream evaluation (Cell 4)
joblib.dump({
    "y_train_before": y_train_before,
    "y_train_after": y_train_after,
    "model_results": model_results,
    "y_test": y_test,
    "absa_missing_series": absa_missing_series,
    "absa_flag_distributions": absa_flag_distributions
}, "data/model_outputs.joblib")
 
print("\n✅ CELL SUCCESSFULY COMPLETED AND STORED VIA JOBLIB!")
