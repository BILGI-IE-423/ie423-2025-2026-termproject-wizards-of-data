# IE 423 Term Project Proposal — Customer Satisfaction in Sephora Products: A Data-Driven Machine Learning Approach

## Team Members
- İlke Bakangil
- Sıla Doğan
- Avsu Evren
- Yağmur Battal
## Website
https://bilgi-ie-423.github.io/ie423-2025-2026-termproject-wizards-of-data/

## Dataset
Dataset: Sephora Products and Skincare Reviews 

Source: https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews/data

## Project Objective

This project develops an end-to-end machine learning pipeline for analyzing customer sentiment in Sephora skincare reviews. By combining review text, Aspect-Based Sentiment Analysis (ABSA), product attributes, ingredient information, and user characteristics, the project aims to identify the key factors influencing customer satisfaction and evaluate how effectively customer sentiment can be predicted.

In addition to predictive modeling, the study investigates the impact of user–product compatibility on customer sentiment and explores common patterns among misclassified reviews to better understand the strengths and limitations of sentiment prediction models.

### Research Questions

**RQ1 — Sentiment Prediction**
To what extent can customer sentiment be predicted using review text, product attributes, ingredient information, and aspect-based sentiment features?

**RQ2 — User–Product Compatibility**
Does alignment between user characteristics (such as skin type, skin tone, and hair-related attributes) and product features lead to more positive customer sentiment?

**RQ3 — Prediction Errors and Model Limitations**
What common characteristics are observed among misclassified reviews, and what do these errors reveal about the limitations of sentiment prediction models?


## Repository Structure
```text
ie423-2025-2026-termproject-wizards-of-data/
│
├── README.md                               • Overview of the project, objectives, and instructions to run the code
├── index.html                              • GitHub Pages website
├── requirements.txt                        • List of required Python libraries  

├── data/                                   • Dataset directory  
│   ├── raw/                                • Original dataset files (see data/README.md for download instructions) 
│   ├── raw_original_data.csv               • Cleaned dataset generated after preprocessing  
│   └── README.md                           • Instructions for downloading and placing the dataset  

├── scripts/                                • Python scripts for each project stage  
│   ├── 01_load_data.py                     • Extracts and merges raw data files. 
│   ├── 02_preprocess_data.py               • Cleans text and extracts DeBERTa ABSA features.  
│   ├── 03_model_training.py                • Handles class balancing and trains optimized ML models (Logistic Regression, Random Forest, SVM).
│   └── 04_model_evaluation                 • Generates all EDA plots, evaluation metrics, and performance tables.

├── visuals/                                • Generated analysis results and assets.  
│   ├── figures/                            • Saved plots and evaluation visualizations (ROC, PR curves, etc.). 
│   └── tables/                             • Generated summary performance and distribution tables (CSVs).
└──
```

## Installation
```text
pip install -r requirements.txt
```
## Running the Scripts
```text
python scripts/01_load_data.py
python scripts/02_preprocess_data.py
python scripts/03_model_training.py
python scripts/04_model_evaluation
```
> **Note:** Make sure to place the raw dataset files in the `data/raw/` directory before running the scripts.  
> For detailed instructions, refer to `data/README.md`.

