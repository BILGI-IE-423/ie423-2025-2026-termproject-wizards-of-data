import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Create output directory
figures_dir = "outputs/figures"
os.makedirs(figures_dir, exist_ok=True)

print("EDA started...")

# =========================================================
# RQ1 — Feature Group Impact on Rating
# =========================================================

brand_score = df.filter(like="brand_name_").sum(axis=1)
product_score = df[[
    "is_vegan",
    "is_clean",
    "is_oily",
    "has_hyaluronic",
    "has_niacinamide"
]].sum(axis=1)
user_score = df.filter(like="skin_").sum(axis=1)

rq1_df = pd.DataFrame({
    "Feature": [
        "Sentiment",
        "Review Length",
        "Brand Information",
        "Product and Ingredient Features",
        "User Characteristics"
    ],
    "Impact": [
        df["sentiment_score"].corr(df["rating"]),
        df["review_length"].corr(df["rating"]),
        brand_score.corr(df["rating"]),
        product_score.corr(df["rating"]),
        user_score.corr(df["rating"])
    ]
}).sort_values(by="Impact", ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(
    rq1_df["Feature"],
    rq1_df["Impact"],
    color=["#FF6B6B", "#4ECDC4", "#FFD93D", "#6A4C93", "#1A936F"]
)
plt.title("Feature Group Impact on Rating")
plt.ylabel("Correlation with Rating")
plt.xticks(rotation=20)

plt.savefig(os.path.join(figures_dir, "rq1_feature_group_impact.png"))
plt.close()

# =========================================================
# RQ1 — Individual Feature Analysis
# =========================================================

all_correlations = df.corr(numeric_only=True)["rating"].sort_values(ascending=False)
top_features = all_correlations.drop("rating").head(10)

plt.figure(figsize=(10, 6))
top_features.plot(kind="bar", color="#2EC4B6")
plt.title("Top Individual Features Affecting Rating")
plt.ylabel("Correlation")
plt.xticks(rotation=45)

plt.savefig(os.path.join(figures_dir, "rq1_top_individual_features.png"))
plt.close()

# =========================================================
# Sentiment Analysis (Detailed)
# =========================================================

plt.figure(figsize=(8, 5))
sns.boxplot(x="rating", y="sentiment_score", data=df)
plt.title("Sentiment Distribution Across Rating Levels")

plt.savefig(os.path.join(figures_dir, "sentiment_distribution.png"))
plt.close()

# =========================================================
# RQ2 — User–Product Alignment Impact
# =========================================================

alignment_df = pd.DataFrame({
    "Feature": [
        "Dry Skin Match",
        "Oily Skin Match",
        "Combination Skin Match",
        "Sensitive Skin Match"
    ],
    "Impact": [
        df["is_dry_match"].corr(df["rating"]),
        df["is_oily_match"].corr(df["rating"]),
        df["is_combination_match"].corr(df["rating"]),
        df["is_sensitive_match"].corr(df["rating"])
    ]
}).sort_values(by="Impact", ascending=False)

plt.figure(figsize=(8, 5))
plt.bar(
    alignment_df["Feature"],
    alignment_df["Impact"],
    color=["#FF9F1C", "#2EC4B6", "#E71D36", "#6A4C93"]
)
plt.title("User–Product Alignment Impact on Rating")
plt.ylabel("Correlation with Rating")

plt.savefig(os.path.join(figures_dir, "rq2_alignment_impact.png"))
plt.close()

# =========================================================
# Correlation Heatmap
# =========================================================

plt.figure(figsize=(6, 5))
sns.heatmap(
    df[["rating", "sentiment_score", "review_length"]].corr(),
    annot=True
)
plt.title("Correlation Heatmap")

plt.savefig(os.path.join(figures_dir, "correlation_heatmap.png"))
plt.close()

print("EDA completed. All figures saved in outputs/figures.")
