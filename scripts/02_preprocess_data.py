import os
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Clean review text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# Main preprocessing pipeline
def preprocess_data(data_dir, output_dir):
    print("--- Preprocessing Data ---")

    review_files = [
        os.path.join(data_dir, "reviews_0-250.csv"),
        os.path.join(data_dir, "reviews_250-500.csv"),
        os.path.join(data_dir, "reviews_500-750.csv"),
        os.path.join(data_dir, "reviews_750-1250.csv"),
        os.path.join(data_dir, "reviews_1250-end.csv")
    ]

    for f in review_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing file: {f}")

    df_reviews = pd.concat(
        [pd.read_csv(f, low_memory=False) for f in review_files],
        ignore_index=True
    )

    df_products = pd.read_csv(os.path.join(data_dir, "product_info.csv"))
    df = pd.merge(df_reviews, df_products, on="product_id")

    print("Initial dataset shape (after merging):", df.shape)

    df = df[[
        "review_text",
        "rating_x",
        "brand_name_x",
        "ingredients",
        "highlights",
        "skin_type",
        "skin_tone",
        "hair_color"
    ]]


    df.columns = df.columns.str.strip().str.lower()


    if "rating_x" in df.columns:
      df = df.rename(columns={"rating_x": "rating"})
    elif "rating" not in df.columns:
     raise ValueError("❌ 'rating' column not found!")


    if "brand_name_x" in df.columns:
      df = df.rename(columns={"brand_name_x": "brand_name"})
    elif "brand_name" not in df.columns:
      raise ValueError("❌ 'brand_name' column not found!")


    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["review_text", "rating"])

    # TEXT
    df["clean_review"] = df["review_text"].apply(clean_text)

    # SENTIMENT
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["clean_review"].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )

    # REVIEW LENGTH
    df["review_length"] = df["clean_review"].apply(lambda x: len(x.split()))

    # FEATURES
    df["is_vegan"] = df["highlights"].str.lower().str.contains("vegan", na=False).astype(int)
    df["is_clean"] = df["highlights"].str.lower().str.contains("clean", na=False).astype(int)
    df["is_oily"] = df["highlights"].str.lower().str.contains("oily", na=False).astype(int)

    df["has_hyaluronic"] = df["ingredients"].str.lower().str.contains("hyaluronic", na=False).astype(int)
    df["has_niacinamide"] = df["ingredients"].str.lower().str.contains("niacinamide", na=False).astype(int)

    # PERSONALIZATION
    df["skin_type"] = df["skin_type"].fillna("unknown").str.lower()

    # ADVANCED ALIGNMENT FEATURES 
    highlights = df["highlights"].fillna("").str.lower()

    dry_keywords = ["dry", "hydrating", "moisturizing", "nourishing"]
    oily_keywords = ["oily", "oil control", "mattifying", "shine"]
    combo_keywords = ["combination", "balanced"]
    sensitive_keywords = ["sensitive", "gentle", "soothing"]

    def contains_any(text_series, keywords):
        pattern = "|".join(keywords)
        return text_series.str.contains(pattern, na=False)
        return df
    df["is_dry_match"] = (
        contains_any(highlights, dry_keywords) &
        (df["skin_type"] == "dry")
    ).astype(int)

    df["is_oily_match"] = (
        contains_any(highlights, oily_keywords) &
        (df["skin_type"] == "oily")
    ).astype(int)

    df["is_combination_match"] = (
        contains_any(highlights, combo_keywords) &
        (df["skin_type"] == "combination")
    ).astype(int)

    df["is_sensitive_match"] = (
        contains_any(highlights, sensitive_keywords) &
        (df["skin_type"] == "sensitive")
    ).astype(int)

    # CATEGORICAL
    df["skin_tone"] = df["skin_tone"].fillna("unknown").astype(str)
    df["hair_color"] = df["hair_color"].fillna("unknown").astype(str)

    df = pd.get_dummies(
        df,
        columns=["brand_name", "skin_type", "skin_tone", "hair_color"],
        drop_first=True
    )

    print("Duplicates before:", df.duplicated().sum())
    df = df.drop_duplicates()
    print("Duplicates after:", df.duplicated().sum())
    print("Final dataset shape:", df.shape)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cleaned_data.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")
   


if __name__ == "__main__":
    data_dir = "data/raw"
    output_dir = "data/processed"

    df_clean = preprocess_data(data_dir, output_dir)

import pandas as pd

df= pd.read_csv("data/processed/cleaned_data.csv", nrows=5)
print(df.head())

