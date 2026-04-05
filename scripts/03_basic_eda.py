import os
import pandas as pd
import matplotlib.pyplot as plt

# Main EDA function
def perform_eda(input_path, figures_dir, tables_dir=None):
    print("--- Performing EDA ---")

    # Load processed dataset
    df = pd.read_csv(input_path)
    print("Dataset shape:", df.shape)
    print("\nColumns:\n", df.columns.tolist())

    # Create output directories for figures and tables
    os.makedirs(figures_dir, exist_ok=True)
    if tables_dir:
        os.makedirs(tables_dir, exist_ok=True)

    # ----------------------------
    # Rating Distribution
    # ----------------------------
    print("\nRating Distribution:\n", df["rating"].value_counts())

    plt.figure()
    df["rating"].hist()
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")

    fig_path = os.path.join(figures_dir, "rating_distribution.png")
    plt.savefig(fig_path)
    plt.close()

    print(f"Saved: {fig_path}")

    # ----------------------------
    # Review Length Distribution
    # ----------------------------
    plt.figure()
    df["review_length"].hist()
    plt.title("Review Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")

    fig_path = os.path.join(figures_dir, "review_length_distribution.png")
    plt.savefig(fig_path)
    plt.close()

    print(f"Saved: {fig_path}")

    # ----------------------------
    # Sentiment vs Rating
    # ----------------------------
    plt.figure()
    plt.scatter(df["sentiment_score"], df["rating"])
    plt.title("Sentiment Score vs Rating")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Rating")

    fig_path = os.path.join(figures_dir, "sentiment_vs_rating.png")
    plt.savefig(fig_path)
    plt.close()

    print(f"Saved: {fig_path}")

    # ----------------------------
    # Feature Averages by Rating
    # ----------------------------
    feature_means = df.groupby("rating")[
        ["review_length", "sentiment_score"]
    ].mean()

    print("\nFeature Means by Rating:\n", feature_means)

    if tables_dir:
        table_path = os.path.join(tables_dir, "feature_means_by_rating.csv")
        feature_means.to_csv(table_path)
        print(f"Saved: {table_path}")

    # ----------------------------
    # Distribution of binary features (vegan / clean / oily)
    # ----------------------------
    binary_features = ["is_vegan", "is_clean", "is_oily"]

    for col in binary_features:
        plt.figure()
        df[col].value_counts().plot(kind="bar")
        plt.title(f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Count")

        fig_path = os.path.join(figures_dir, f"{col}_distribution.png")
        plt.savefig(fig_path)
        plt.close()

        print(f"Saved: {fig_path}")

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    input_path = "data/processed/cleaned_data.csv"
    figures_dir = "outputs/figures"
    tables_dir = "outputs/tables"

     # Run the EDA pipeline
    perform_eda(input_path, figures_dir, tables_dir)
