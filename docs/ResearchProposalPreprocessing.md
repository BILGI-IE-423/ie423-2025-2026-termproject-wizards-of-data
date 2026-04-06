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


### Step 4 — Saving Processed Data
The final processed dataset was saved using `scripts/02_preprocess_data.py` to:

`data/processed/cleaned_data.csv`

**Purpose:**  
Storing the processed dataset ensures that all preprocessing steps are preserved and do not need to be repeated, enabling consistent and reproducible analysis in subsequent stages.

Before saving, the dataset was organized into a structured, model-ready format with all relevant features retained. This step further ensures reproducibility and allows the dataset to be reused in later stages without repeating preprocessing.

**Result:**  
A finalized, model-ready dataset was generated and stored for use in EDA and machine learning tasks.

---
## Initial Outputs

### Dataset Shape
After loading the dataset, the reviews data consists of 1,094,411 rows and 19 columns, while the products data contains 8,494 rows and 27 columns.

### Missing Value Summary
The following variables had missing values:

**Reviews:**  
- is_recommended 
- helpfulness  
- review_text
- review_text
- review_title                
- skin_tone                   
- eye_color                   
- skin_type                   
- hair_color

**Products:**  
- rating                 
- reviews                
- size                  
- variation_type        
- variation_value       
- variation_desc        
- ingredients 
- value_price_usd       
- sale_price_usd        
- highlights 
- secondary_category       
- tertiary_category  
- child_max_price       
- child_min_price 
## Visualizations

All outputs below are generated by `scripts/03_basic_eda.py`.
## Figure 1 - Rating Distribution

The distribution shows that most products receive high ratings, with a strong concentration at 5 stars. This indicates a positive skew in the dataset and generally high customer satisfaction.

<p align="center">
  <img src="https://github.com/user-attachments/assets/991265f5-838c-4a97-971c-2954997d4866" width="800">
</p>


## Figure 2 - Review Length Distribution

The histogram reveals that most reviews are relatively short, with only a small number of longer reviews. This suggests that users tend to provide concise feedback.

<p align="center">
  <img src="https://github.com/user-attachments/assets/73cbd09f-6592-4cb8-8a90-4c6fbbbbe34d" width="800">
</p>


## Figure 3 - Sentiment vs Rating

The chart illustrates that positive sentiment increases with higher ratings, while negative sentiment is more common in lower ratings. This highlights a clear relationship between sentiment and user ratings.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d30187f0-3c75-4755-9177-22bae867c304" width="700">
</p>
                                    
## Figure 4 - Correlation Heatmap

The heatmap indicates a moderate positive correlation between sentiment score and rating, whereas review length shows almost no relationship with rating. This suggests that sentiment is more informative than review length.

<p align="center">
  <img src="https://github.com/user-attachments/assets/294e3804-834d-4fce-851e-78541f2a2d47" width="700">
</p>

---

## Reproducibility Instructions

This section explains how to reproduce all results in this project.

### 1. Clone the Repository

```bash
git clone https://github.com/BILGI-IE-423/ie423-2025-2026-termproject-wizards-of-data.git
cd ie423-2025-2026-termproject-wizards-of-data
```

### 2. Install Required Packages

Make sure Python is installed, then run:

```bash
pip install -r requirements.txt
```

### 3. Place the Dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews

After downloading, place all dataset files inside the following directory:

```
data/raw/
```
> **Important:** The project assumes that all raw dataset files are located in the `data/raw/` directory with their original filenames.  
> If the files are missing or placed incorrectly, the scripts will not run.  
> For detailed setup instructions, see `data/README.md`.
The folder structure should look like:

```
data/
 └── raw/
     ├── reviews_0-250.csv
     ├── reviews_250-500.csv
     ├── reviews_500-750.csv
     ├── reviews_750-1250.csv
     ├── reviews_1250-end.csv
     └── product_info.csv
```

### 4. Run the Scripts

Run the scripts in the following order:

```bash
python scripts/01_load_data.py
python scripts/02_preprocess_data.py
python scripts/03_basic_eda.py
```

### 5. Outputs

After running the scripts:

- Cleaned dataset will be saved in:
  `data/processed/cleaned_data.csv`

- Figures will be saved in:
  `outputs/figures/`

- Tables will be saved in:
  
  `outputs/tables/`

---

## Transparency and Traceability

All outputs presented in this document are generated directly from the Python scripts located in the `scripts/` folder.

- Data loading and initial inspection outputs are generated by:
  `scripts/01_load_data.py`

- Data preprocessing and feature engineering are performed by:
  `scripts/02_preprocess_data.py`

- All visualizations and analysis results are generated by:
  `scripts/03_basic_eda.py`

### Data Pipeline

The project follows a structured pipeline:

1. Raw data is stored in:
   `data/raw/`

2. Processed data is generated and saved in:
   `data/processed/cleaned_data.csv`

3. Visual outputs (plots) are saved in:
   `outputs/figures/`

4. Tables (if any) are stored in:
   `outputs/tables/`

### Reproducibility Guarantee

- All results are reproducible by running the provided scripts.
- No manual modifications are applied to the outputs.
- Every figure and dataset can be traced back to its corresponding script.
