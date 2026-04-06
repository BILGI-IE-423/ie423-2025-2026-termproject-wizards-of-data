import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set(style="whitegrid")

# Load processed dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Define output directories
figures_dir = "outputs/figures"
tables_dir = "outputs/tables"

os.makedirs(figures_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

print("EDA started...")

# Create sentiment category (positive / negative / neutral)
def sentiment_category(score):
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

df["sentiment_category"] = df["sentiment_score"].apply(sentiment_category)

# Generate summary statistics table for key numerical features
feature_stats = df[["rating", "review_length", "sentiment_score"]].describe()
feature_stats.to_csv(os.path.join(tables_dir, "01_feature_statistics.csv"))

# Generate feature ratio table for product attributes
feature_ratio = pd.DataFrame({
    "Feature": ["Vegan", "Clean", "Oily", "Niacinamide", "Hyaluronic"],
    "Percentage (%)": [
        df["is_vegan"].mean()*100,
        df["is_clean"].mean()*100,
        df["is_oily"].mean()*100,
        df["has_niacinamide"].mean()*100,
        df["has_hyaluronic"].mean()*100
    ]
})

feature_ratio.to_csv(os.path.join(tables_dir, "02_feature_ratios.csv"), index=False)

# Plot rating distribution
plt.figure(figsize=(8,5))
sns.countplot(x="rating", data=df, palette="viridis")
plt.title("Rating Distribution")
plt.savefig(os.path.join(figures_dir, "01_rating_distribution.png"))
plt.close()

# Plot review length distribution
plt.figure(figsize=(8,5))
sns.histplot(df["review_length"], bins=40, color="#3498DB")
plt.title("Review Length Distribution")
plt.savefig(os.path.join(figures_dir, "02_review_length_distribution.png"))
plt.close()

# Plot sentiment vs rating distribution
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

# Plot combined product feature distribution (present vs absent)
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

# Plot correlation heatmap for key variables
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
