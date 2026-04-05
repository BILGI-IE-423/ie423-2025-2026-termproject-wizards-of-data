import os
import sys
import pandas as pd


def load_data(data_dir: str):
    print("\n" + "="*50)
    print("              LOADING DATA")
    print("="*50)

    review_files = [
        os.path.join(data_dir, "reviews_0-250.csv"),
        os.path.join(data_dir, "reviews_250-500.csv"),
        os.path.join(data_dir, "reviews_500-750.csv"),
        os.path.join(data_dir, "reviews_750-1250.csv"),
        os.path.join(data_dir, "reviews_1250-end.csv")
    ]

    for f in review_files:
        if not os.path.exists(f):
            print(f"Missing file: {f}")
            sys.exit(1)

    df_reviews = pd.concat(
        [pd.read_csv(f, low_memory=False) for f in review_files],
        ignore_index=True
    )

    product_path = os.path.join(data_dir, "product_info.csv")
    if not os.path.exists(product_path):
        print(" Missing product_info.csv")
        sys.exit(1)

    df_products = pd.read_csv(product_path)

    print("Data successfully loaded")
    return df_reviews, df_products


def print_basic_info(df, name):
    print("\n" + "="*50)
    print(f"              {name} DATASET")
    print("="*50)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\n Data Types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\n Missing Values:")
    print(df.isnull().sum())

    print("\nBasic Statistics:")
    print(df.describe())

    print("\n" + "-"*50)


if __name__ == "__main__":
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        repo_root = os.getcwd()

    os.chdir(repo_root)

    data_dir = os.path.join("data", "raw")
    tables_dir = os.path.join("outputs", "tables")

    df_reviews, df_products = load_data(data_dir)

    print_basic_info(df_reviews, "REVIEWS")
    print_basic_info(df_products, "PRODUCTS")

    #  UNIQUE VALUE ANALYSIS
    columns_to_check = ["skin_type", "skin_tone", "hair_color"]

    print("\n" + "="*50)
    print("        UNIQUE VALUE ANALYSIS (REVIEWS/skin_type,skin_tone,hair_color)")
    print("="*50)

    for col in columns_to_check:
        print("\n" + "-"*40)
        print(f"Column: {col}")
        print("-"*40)

        if col in df_reviews.columns:
            unique_vals = sorted(df_reviews[col].dropna().unique())

            print(f"Total unique values: {len(unique_vals)}")
            print("Values:")

            for val in unique_vals:
                print(f"  - {val}")
        else:
            print(" Column not found")

    print("\n" + "="*50)