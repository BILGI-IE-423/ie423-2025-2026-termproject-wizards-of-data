import os
import pandas as pd
import re
from textblob import TextBlob

# Clean review text: lowercase + remove non-alphabetic chars
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# Main preprocessing pipeline
def preprocess_data(data_dir, output_dir):
    print("--- Preprocessing Data ---")

    # Load multiple review files and check existence
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

    # Load product info and merge with reviews
    df_products = pd.read_csv(os.path.join(data_dir, "product_info.csv"))
    df = pd.merge(df_reviews, df_products, on="product_id")
    print("Initial dataset shape (after merging):", df.shape)

    # Select and rename relevant columns
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

    df = df.rename(columns={
        "rating_x": "rating",
        "brand_name_x": "brand_name"
    })

    # Ensure numeric rating and drop missing critical info
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["review_text", "rating"])

    # text cleaning
    df["clean_review"] = df["review_text"].apply(clean_text)

    #  Sentiment analysis
    df["sentiment_score"] = df["clean_review"].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    # review length
    df["review_length"] = df["clean_review"].apply(lambda x: len(x.split()))

    # feature engineering
    df["is_vegan"] = df["highlights"].fillna("").apply(lambda x: int("vegan" in x.lower()))
    df["is_clean"] = df["highlights"].fillna("").apply(lambda x: int("clean" in x.lower()))
    df["is_oily"] = df["highlights"].fillna("").apply(lambda x: int("oily" in x.lower()))

    df["has_hyaluronic"] = df["ingredients"].fillna("").apply(lambda x: int("hyaluronic" in x.lower()))
    df["has_niacinamide"] = df["ingredients"].fillna("").apply(lambda x: int("niacinamide" in x.lower()))

    # Personalization feature based on skin type
    df["skin_type"] = df["skin_type"].fillna("unknown").str.lower()

    df["is_dry_match"] = df.apply(
        lambda x: int("dry" in str(x["highlights"]).lower() and x["skin_type"] == "dry"),
        axis=1
    )

    # Fill other categorical fields and encode them
    df["skin_tone"] = df["skin_tone"].fillna("unknown").astype(str)
    df["hair_color"] = df["hair_color"].fillna("unknown").astype(str)
    df = pd.get_dummies(
        df,
        columns=["brand_name", "skin_type", "skin_tone", "hair_color"],
        drop_first=True
    )

    # Remove duplicates and check final shape
    print("Duplicates before:", df.duplicated().sum())
    df = df.drop_duplicates()
    print("Duplicates after:", df.duplicated().sum())
    print("Final dataset shape (after preprocessing):", df.shape)

    # output
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "cleaned_data.csv")
    df.to_csv(output_path, index=False)

    print(f"Processed data saved to: {output_path}")

    return df


if __name__ == "__main__":
    data_dir = "data/raw"
    output_dir = "data/processed"
  
    # Run the preprocessing pipeline
    df_clean = preprocess_data(data_dir, output_dir)
