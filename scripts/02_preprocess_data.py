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
 
try:
    from transformers import pipeline
    print("Hugging Face Transformers is ready.")
except ImportError:
    print("Installing Transformers package... Please wait.")
    os.system("pip install -q transformers")
    from transformers import pipeline
 
def clean_build_and_subsample(df, sample_size=5000):
    print("\n--- Data Cleaning and Subsampling ---")
    df = df.dropna(subset=["review_text", "rating"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(np.int8)
    df = df[df["rating"].isin([1, 2, 3, 4, 5])].copy()
 
    df["target"] = np.where(df["rating"] <= 3, 0, 1).astype(np.int32)
 
    if len(df) > sample_size:
        print(f">> Subsampling a balanced set of {sample_size} rows from large dataset...")
        df_stratified, _ = train_test_split(
            df,
            train_size=sample_size,
            random_state=42,
            stratify=df["target"]
        )
        df_stratified = df_stratified.reset_index(drop=True)
    else:
        df_stratified = df.reset_index(drop=True)
 
    return df_stratified
 
def extract_absa_features_pure_hf(df):
    print(f"\n--- Pure Hugging Face ABSA Layer Initialized (Row Count: {len(df)}) ---")
  
    chosen_device = 0 if torch.cuda.is_available() else -1
    print(f">> Executing text classification model on: {'GPU (device=0)' if chosen_device == 0 else 'CPU (device=-1)'}")
 
    try:
        absa_pipeline = pipeline(
            "text-classification",
            model="Yangheng/deberta-v3-base-absa-v1.1",
            device=chosen_device
        )
    except Exception as e:
        print(f"Custom model loading failed, falling back to base model: {e}")
        absa_pipeline = pipeline("sentiment-analysis", device=chosen_device)
 
    absa_cols = ["absa_skin_aspect", "absa_hair_aspect", "absa_product_aspect"]
    available_cols = ["absa_skin_available", "absa_hair_available", "absa_product_available"]
 
    for col in absa_cols: df[col] = np.nan
    for col in available_cols: df[col] = 0
 
    aspect_mapping = {
        "skin": ("absa_skin_aspect", "absa_skin_available"),
        "hair": ("absa_hair_aspect", "absa_hair_available"),
        "product": ("absa_product_aspect", "absa_product_available")
    }
 
    total_rows = len(df)
 
    for i, idx in enumerate(df.index):
        if i % 250 == 0 and i > 0:
            print(f"   [{i}/{total_rows}] rows processed via DeBERTa ABSA...")
 
        text = str(df.at[idx, "review_text"])[:500]
        if not text.strip():
            continue
 
        for aspect, (score_col, available_col) in aspect_mapping.items():
            try:
                result = absa_pipeline([{"text": text, "text_pair": aspect}])[0]
                label = result["label"].lower()
                confidence = result["score"]
 
                if "positive" in label:
                    score = confidence
                    flag = 2 
                elif "negative" in label:
                    score = -confidence
                    flag = 2 
                elif "neutral" in label:
                    score = 0.0
                    flag = 1 
                else:
                    score = np.nan
                    flag = 0 
 
                df.at[idx, score_col] = np.float32(score)
                df.at[idx, available_col] = int(flag)
            except:
                df.at[idx, score_col] = np.nan
                df.at[idx, available_col] = 0
 
    return df
 
def engineer_features(df):
    df["review_length"] = df["review_text"].fillna("").astype(str).apply(lambda x: len(x.split())).astype(np.int32)
 
    ingredients_lower = df["ingredients"].fillna("").astype(str).str.lower()
    highlights_lower = df["highlights"].fillna("").astype(str).str.lower()
 
    df["is_vegan"] = highlights_lower.str.contains("vegan", na=False).astype(np.int8)
    df["is_cruelty_free"] = highlights_lower.str.contains("cruelty", na=False).astype(np.int8)
    df["is_clean_beauty"] = highlights_lower.str.contains("clean", na=False).astype(np.int8)
 
    df["has_hyaluronic"] = ingredients_lower.str.contains("hyaluronic", na=False).astype(np.int8)
    df["has_retinol"] = ingredients_lower.str.contains("retinol", na=False).astype(np.int8)
    df["has_niacinamide"] = ingredients_lower.str.contains("niacinamide", na=False).astype(np.int8)
    df["has_acids"] = ingredients_lower.str.contains("salicylic|glycolic|aha|bha", na=False).astype(np.int8)
    df["has_fragrance"] = ingredients_lower.str.contains("fragrance|perfume|parfum", na=False).astype(np.int8)
 
    if "price_usd" in df.columns:
        df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce").astype(np.float32)
    else:
        df["price_usd"] = np.nan
 
    st_lower = df["skin_type"].fillna("unknown").astype(str).str.lower()
    df["user_product_alignment"] = (
        ((st_lower == "dry") & (df["has_hyaluronic"] == 1)) |
        ((st_lower == "oily") & (df["has_acids"] == 1)) |
        ((st_lower == "sensitive") & (df["has_fragrance"] == 0))
    ).astype(np.int8)
 
    return df
 
# --- MAIN EXECUTION FLOW ---
if "raw_df" not in globals():
    raise NameError("Error: 'raw_df' not found in memory. Please execute Cell 1 first!")
 
df_subsampled = clean_build_and_subsample(raw_df, sample_size=5000)
df_hf_processed = extract_absa_features_pure_hf(df_subsampled)
 
absa_missing_series = df_hf_processed[["absa_skin_aspect", "absa_hair_aspect", "absa_product_aspect"]].isna().mean()
 
absa_flag_distributions = {}
for col in ["absa_skin_available", "absa_hair_available", "absa_product_available"]:
    dist = df_hf_processed[col].value_counts(normalize=True).reindex([0, 1, 2], fill_value=0.0)
    absa_flag_distributions[col] = dist.to_dict()
 
print(">> Preparing a clean, non-skewed dataset copy for EDA plots...")
df_eda_save = engineer_features(df_hf_processed.copy())
os.makedirs("data", exist_ok=True)
df_eda_save.to_csv("data/raw_original_data.csv", index=False)
