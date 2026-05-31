# =====================================================================
# 1.LOAD DATA & MERGE
# =====================================================================
import os
import zipfile
import gc
import pandas as pd
import numpy as np
 
def extract_zip_files(zip_dir="data/raw", extract_to="data/raw"):
    print("--- Opening Zip Files ---")
    if not os.path.exists(zip_dir):
        os.makedirs(extract_to, exist_ok=True)
        zip_dir = "."
 
    for file in os.listdir(zip_dir):
        if file.endswith(".zip"):
            file_path = os.path.join(zip_dir, file)
            print(f"Opening: {file_path}")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
 
def find_rating_column(df):
    possible_cols = ["rating", "rating_x", "rating_y"]
    for col in possible_cols:
        if col in df.columns: return col
    rating_like_cols = [col for col in df.columns if "rating" in col.lower()]
    if len(rating_like_cols) > 0: return rating_like_cols[0]
    raise KeyError("The Rating column was not found.")
 
def load_and_merge_data(data_dir="data/raw"):
    print("\n--- Data Upload and Merging Has Begun ---")
 
    review_files_raw = [os.path.join(data_dir, f"reviews_{i}.csv") for i in ["0-250", "250-500", "500-750", "750-1250", "1250-end"]]
    review_files_root = [f"reviews_{i}.csv" for i in ["0-250", "250-500", "500-750", "750-1250", "1250-end"]]
 
    existing_review_files = [f for f in review_files_raw if os.path.exists(f)]
    if not existing_review_files:
        existing_review_files = [f for f in review_files_root if os.path.exists(f)]
    # -----------------------------------------------------------------
 
    if not existing_review_files:
        raise FileNotFoundError(
            f" ERROR: The required ‘reviews_*.csv’ files were not found in the ‘{data_dir}’ folder or the root directory!\n"
            f"Please make sure the ZIP files have been extracted correctly or that you know the path to your CSV files."
        )
    # -----------------------------------------------------------------
 
    review_dfs = []
    for file in existing_review_files:
        print(f"Loading: {file}")
 
      
        temp = pd.read_csv(
            file,
            quoting=3,          
            engine='python',    
            on_bad_lines='skip' 
        )
 
        temp.columns = temp.columns.str.strip('"').str.strip("'")
        needed_cols = [col for col in temp.columns if col in ["review_text", "skin_type", "skin_tone", "product_id"] or "rating" in col.lower()]
        review_dfs.append(temp[needed_cols])
 
    df_reviews = pd.concat(review_dfs, ignore_index=True)
 
    if "review_text" in df_reviews.columns:
        df_reviews["review_text"] = df_reviews["review_text"].astype(str).str.strip('"').str.strip("'")
 
    print(f"Total Number of Review Lines: {len(df_reviews)}")
 
    product_file = os.path.join(data_dir, "product_info.csv")
    if not os.path.exists(product_file) and os.path.exists("product_info.csv"):
        product_file = "product_info.csv"
 
    if os.path.exists(product_file):
        print(f"Loading: {product_file}")
        df_products = pd.read_csv(product_file, engine='python', on_bad_lines='skip')
        df_products.columns = df_products.columns.str.strip('"').str.strip("'")
 
        # Sadece gerekli sütunları filtrele
        valid_prod_cols = [c for c in ["product_id", "ingredients", "highlights"] if c in df_products.columns]
        df_products = df_products[valid_prod_cols]
 
        df = pd.merge(df_reviews, df_products, on="product_id", how="left")
    else:
        print(" Warning: ‘product_info.csv’ not found; proceeding with review data only.")
        df = df_reviews
 
    rating_col = find_rating_column(df)
    if rating_col != "rating":
        df = df.rename(columns={rating_col: "rating"})
      
    return df
 
# Akışı başlatıyoruz
extract_zip_files()
raw_df = load_and_merge_data()
print("\nStep 1 Successfully Completed! ‘raw_df’ has been loaded into memory.")
