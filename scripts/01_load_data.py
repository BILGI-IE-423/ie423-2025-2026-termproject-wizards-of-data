import os
import sys
import pandas as pd

# Load review and product CSV files into DataFrames
def load_data(data_dir: str):
    print("--- Loading Data ---") 
    review_files = [
        os.path.join(data_dir, "reviews_0-250.csv"),
        os.path.join(data_dir, "reviews_250-500.csv"),
        os.path.join(data_dir, "reviews_500-750.csv"),
        os.path.join(data_dir, "reviews_750-1250.csv"),
        os.path.join(data_dir, "reviews_1250-end.csv")
    ]
    # Ensure all review files exist
    for f in review_files:
        if not os.path.exists(f):
            print(f"Missing file: {f}")
            sys.exit(1)
          
    # Combine all review CSVs
    df_reviews = pd.concat(
    [pd.read_csv(f, low_memory=False) for f in review_files],
    ignore_index=True
)

    product_path = os.path.join(data_dir, "product_info.csv")
    if not os.path.exists(product_path):
        print("Missing product_info.csv")
        sys.exit(1)

    df_products = pd.read_csv(product_path)


    return df_reviews, df_products

# Print basic info about a DataFrame
def print_basic_info(df, name):
    print(f"\n--- {name} INFO ---")

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nBasic Statistics:")
    print(df.describe())


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
