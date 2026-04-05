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
The dataset was loaded using `scripts/01_load_data.py`. Multiple review files were read and concatenated into a single dataframe using pandas to create a unified reviews dataset. Product-level data was loaded separately from `product_info.csv`. These datasets were prepared for merging by ensuring file availability and structural consistency.

**Purpose:**  
Combining fragmented review files into a single dataset and linking them with product-level information allows each observation to carry both user-generated content and product attributes, which is essential for downstream analysis.

**Result:**  
Two structured datasets were obtained:
- a consolidated reviews dataset  
- a product-level dataset ready for integration  

---

### Step 2 — Initial Inspection
An initial inspection was performed using `scripts/01_load_data.py` to understand the structure and quality of the data. This included:
- checking dataset shape and dimensionality  
- examining column names and data types  
- identifying missing values  
- reviewing basic statistical summaries  
- analyzing unique values in key categorical variables (e.g., skin type, skin tone, hair color)

**Purpose:**  
This step provides a clear understanding of the dataset’s structure and reveals potential data quality issues that may affect analysis, such as missing values or inconsistent categorical entries.

**Result:**  
A comprehensive overview of the dataset was obtained, including its structure, distributions, and potential preprocessing requirements.

---

### Step 3 — Cleaning and Feature Engineering
Data cleaning and transformation were performed using `scripts/02_preprocess_data.py` to prepare the dataset for analysis and machine learning.

**What was done:**
- selected relevant columns (review text, rating, product attributes, and user features)  
- renamed columns for consistency (`rating_x` → `rating`, `brand_name_x` → `brand_name`)  
- converted the `rating` column to numeric format  
- removed rows with missing `review_text` or `rating` values  
- removed duplicate observations  

**Text processing and feature creation:**
- cleaned review text (lowercasing and removing non-alphabetic characters)  
- generated sentiment scores using the VADER sentiment analysis tool  
- created review length (word count)  

**Feature engineering:**
- extracted product-related indicators (e.g., vegan, clean, oily) from highlights  
- extracted ingredient-based features (e.g., hyaluronic acid, niacinamide)  
- created user–product alignment features (e.g., `is_dry_match`, `is_oily_match`)  
- handled missing categorical values (filled with "unknown")  
- encoded categorical variables using one-hot encoding  

**Purpose:**  
These transformations standardize the data, eliminate noise, and convert both textual and categorical information into structured numerical features that can be effectively used in analytical and predictive models.

**Result:**  
A cleaned and enriched dataset was produced, where raw inputs were transformed into structured variables suitable for quantitative analysis.

---

### Step 4 — Saving Processed Data
The final processed dataset was saved using `scripts/02_preprocess_data.py` to:

`data/processed/cleaned_data.csv`

**Purpose:**  
Storing the processed dataset ensures that all preprocessing steps are preserved and do not need to be repeated, enabling consistent and reproducible analysis in subsequent stages.

Before saving, the dataset was organized into a structured, model-ready format with all relevant features retained. This step further ensures reproducibility and allows the dataset to be reused in later stages without repeating preprocessing.

**Result:**  
A finalized, model-ready dataset was generated and stored for use in EDA and machine learning tasks.
