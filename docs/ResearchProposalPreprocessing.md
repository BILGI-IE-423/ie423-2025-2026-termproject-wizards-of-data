# IE 423 Term Project Proposal — Customer Satisfaction in Sephora Products: A Data-Driven Machine Learning Approach

## Team Information

- İlke Bakangil
- Sıla Doğan
- Avsu Evren
- Yağmur Battal
  
## Dataset Description

The dataset used in this project is the *Sephora Products and Skincare Reviews* dataset, obtained from Kaggle (https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews).

This dataset contains detailed information on skincare products available on Sephora, along with user-generated reviews reflecting customer experiences. It includes structured variables such as product attributes (e.g., brand, category, price, ingredients, product highlights, and ratings), as well as user-related characteristics (e.g., skin type, skin tone, and hair color). In addition, it contains unstructured textual review data.

The dataset is particularly valuable as it combines multiple dimensions of consumer behavior, including product features, user characteristics, and subjective feedback. This enables a comprehensive analysis of customer satisfaction and supports the exploration of relationships between product attributes and user experiences.

Due to its rich and diverse structure, the dataset is well-suited for both exploratory analysis and predictive modeling, providing a meaningful basis for understanding consumer preferences and evaluating data-driven approaches in the skincare domain.

After initial inspection, the dataset consists of over 8,000 products and approximately 1 million user reviews across multiple related files.

## Dataset Access

The dataset used in this project is not stored directly in the repository due to its size. It can be downloaded from the following Kaggle link:

https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews

After downloading the dataset, the files should be placed in the following directory:

`data/raw/`

Detailed instructions on how to obtain and organize the dataset are provided in:

`data/README.md`

The project assumes that all dataset files are located in this folder with their original file names. Users are required to manually download the dataset before running the scripts.

## Research Questions

### Research Question 1  
How accurately can product ratings be predicted using customer review sentiment analysis, brand information, content data, and product feature data?

**Explanation:**  
This question aims to evaluate the extent to which product ratings can be explained and predicted using a combination of structured and unstructured data. Understanding the predictive power of different feature types—such as sentiment extracted from textual reviews, product characteristics, and brand-related information—can provide insights into the key drivers of customer satisfaction. The dataset supports this analysis by combining textual review data with structured product and user-related features, enabling the integration of sentiment analysis with feature-based predictive modeling approaches.

---
### Research Question 2  
Do users give higher ratings to products when their personal characteristics (skin type, skin tone, and hair color) are aligned with product highlights?

**Explanation:**  
This question explores whether compatibility between user characteristics and product attributes influences customer satisfaction. By examining the relationship between user profiles and product highlights, the analysis aims to uncover whether personalized product-user alignment leads to higher ratings. The dataset includes both user-related features and product descriptors, making it possible to investigate patterns of preference and evaluate the role of personalization in shaping user experiences.

---
### Research Question 3  
How does the performance of different machine learning algorithms compare in predicting product ratings, and which model provides the highest predictive accuracy?

**Explanation:**  
This question focuses on comparing the effectiveness of different machine learning algorithms in predicting product ratings. By applying and evaluating multiple models, the goal is to identify which approaches perform best in capturing the underlying patterns in the data. The dataset’s combination of structured and unstructured features makes it suitable for testing a variety of models, allowing for a comprehensive comparison in terms of predictive accuracy and overall performance.

---

## Project Proposal

This project aims to investigate how product ratings reflect customer satisfaction and how accurately they can be predicted using different feature groups such as customer review sentiment, product attributes, brand information, ingredient data, and user characteristics, based on the Sephora Products and Skincare Reviews dataset.

First, we will clean and preprocess the dataset by handling missing values, removing duplicates, selecting relevant variables, and preparing both structured and textual data for analysis. In particular, customer reviews will be processed and transformed into sentiment scores using natural language processing techniques.

Then, we will conduct exploratory data analysis to understand the distribution of product ratings, examine relationships between sentiment and ratings, and identify patterns across different feature groups and user characteristics.

Based on our research questions, we will apply predictive modeling methods to evaluate how different feature groups contribute to rating prediction, analyze whether alignment between user characteristics and product attributes leads to higher ratings, and compare the performance of different machine learning models.

Our goal is not only to build a predictive model, but also to interpret product ratings as indicators of customer satisfaction, identify the most influential factors affecting these ratings, and understand how personalization influences customer satisfaction.

Possible challenges include handling missing and noisy data, processing large-scale textual data, constructing meaningful alignment features, managing high-dimensional data after encoding, and preventing overfitting in predictive models.

---

## Preprocessing Steps

### Step 1 — Loading the Data
The dataset was loaded using `scripts/01_load_data.py`. The Sephora Products and Skincare Reviews dataset consists of multiple files belonging to a single dataset. Review files were loaded separately and concatenated into a single dataframe using pandas. Product-level data was loaded from `product_info.csv`, and both datasets were merged on the `product_id` key using an inner join. This integration step ensured that each customer review is directly associated with its corresponding product attributes.

---

### Step 2 — Initial Inspection
We checked:
- dataset shape and dimensionality  
- column names and data types using `.info()`  
- missing values using `.isnull().sum()`  
- basic statistical summaries to understand value distributions  

This step provided a clear overview of the dataset and highlighted potential issues to address.

---

### Step 3 — Cleaning
Using `scripts/02_preprocess_data.py`, we:
- selected relevant variables (review text, rating, brand, ingredients, highlights, user features)  
- renamed columns such as `rating_x` → `rating` and `brand_name_x` → `brand_name`  
- converted the `rating` variable to numeric using `pd.to_numeric()`  
- removed rows with missing values in `review_text` and `rating`  
- removed duplicate observations using `.drop_duplicates()`  
- cleaned review text using regular expressions (lowercasing and removing non-alphabetic characters)  
- generated sentiment scores using the TextBlob library  
- created additional features such as review length (word count)  
- extracted product-related indicators (e.g., vegan, clean, oily) from text fields  
- extracted ingredient-based features (e.g., hyaluronic acid, niacinamide)  
- created a user–product alignment feature (e.g., `is_dry_match`)  
- encoded categorical variables (brand name, skin type, skin tone, hair color) using `pd.get_dummies()` with `drop_first=True`  

These steps ensured that the dataset was cleaned, enriched, and transformed into a format suitable for machine learning.

---

### Step 4 — Saving Processed Data
The cleaned and fully processed dataset was saved using `scripts/02_preprocess_data.py` to:

`data/processed/cleaned_data.csv`

Before saving, non-essential columns such as raw text fields were removed, and all variables were structured in a model-ready format. This step ensures reproducibility and allows the dataset to be reused in later stages without repeating preprocessing.
