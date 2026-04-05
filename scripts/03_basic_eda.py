import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

df = pd.read_csv("data/processed/cleaned_data.csv")

figures_dir = "outputs/figures"
tables_dir = "outputs/tables"

os.makedirs(figures_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

print("EDA started...")

# =========================================================
# SENTIMENT CATEGORY
# =========================================================

def sentiment_category(score):
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

df["sentiment_category"] = df["sentiment_score"].apply(sentiment_category)

# =========================================================
# TABLE 1 — FEATURE STATISTICS
# =========================================================

feature_stats = df[["rating","review_length","sentiment_score"]].describe()
feature_stats.to_csv(os.path.join(tables_dir, "01_feature_statistics.csv"))

# =========================================================
# TABLE 2 — FEATURE RATIOS
# =========================================================

feature_ratio = pd.DataFrame({
    "Feature": ["Vegan","Clean","Oily","Niacinamide","Hyaluronic"],
    "Percentage (%)": [
        df["is_vegan"].mean()*100,
        df["is_clean"].mean()*100,
        df["is_oily"].mean()*100,
        df["has_niacinamide"].mean()*100,
        df["has_hyaluronic"].mean()*100
    ]
})

feature_ratio.to_csv(os.path.join(tables_dir, "02_feature_ratios.csv"), index=False)

# =========================================================
# FIGURE 01 — RATING DISTRIBUTION
# =========================================================

plt.figure(figsize=(8,5))
sns.countplot(x="rating", data=df, palette="viridis")
plt.title("Rating Distribution")
plt.savefig(os.path.join(figures_dir, "01_rating_distribution.png"))
plt.close()

# =========================================================
# FIGURE 02 — REVIEW LENGTH DISTRIBUTION
# =========================================================

plt.figure(figsize=(8,5))
sns.histplot(df["review_length"], bins=40, color="#3498DB")
plt.title("Review Length Distribution")
plt.savefig(os.path.join(figures_dir, "02_review_length_distribution.png"))
plt.close()

# =========================================================
# FIGURE 03 — SENTIMENT vs RATING (UPDATED)
# =========================================================

plt.figure(figsize=(6,4))
sns.countplot(
    x="rating",
    hue="sentiment_category",
    data=df,
    palette={"negative":"#E15759","neutral":"#BAB0AC","positive":"#59A14F"}
)
plt.title("Sentiment vs Rating")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "03_sentiment_vs_rating.png"))
plt.close()

# =========================================================
# FIGURE 04 — COMBINED PRODUCT FEATURES (UPDATED)
# =========================================================

features = ["is_vegan","is_clean","is_oily","has_niacinamide"]

feature_counts = []

for feature in features:
    count_1 = df[feature].sum()
    count_0 = len(df) - count_1

    feature_counts.append({
        "Feature": feature,
        "Present": count_1,
        "Absent": count_0
    })

feature_df = pd.DataFrame(feature_counts)
feature_df = feature_df.melt(id_vars="Feature", var_name="Status", value_name="Count")

plt.figure(figsize=(8,5))
sns.barplot(
    data=feature_df,
    x="Feature",
    y="Count",
    hue="Status",
    palette={"Present":"#59A14F","Absent":"#E15759"}
)
plt.title("Product Feature Distribution (Present vs Absent)")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "04_combined_features.png"))
plt.close()

# =========================================================
# FIGURE 05 — CORRELATION HEATMAP
# =========================================================

plt.figure(figsize=(6,5))
sns.heatmap(
    df[["rating","sentiment_score","review_length"]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(figures_dir, "05_correlation_heatmap.png"))
plt.close()

print("EDA completed successfully.")
