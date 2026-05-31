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
How accurately can product ratings be predicted using customer review sentiment analysis, product-related information, review content, and engineered product features?

**RQ2 — User–Product Compatibility**
Do users give higher ratings to products when their personal characteristics are aligned with product highlights?

**RQ3 — Prediction Errors and Model Limitations**
How does the performance of different machine learning algorithms compare in predicting product ratings, and which model provides the highest predictive accuracy?


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
## Project Summary

This project develops an end-to-end machine learning pipeline for analyzing customer sentiment in Sephora skincare reviews by combining high-dimensional text data, product attributes, ingredient information, and user characteristics.

* **The Dataset:** Built upon a massive raw corpus comprising over **478,000 review lines**.
* **Core Innovation:** Integration of deep-learning semantic insights extracted via a **Hugging Face DeBERTa-v3 Aspect-Based Sentiment Analysis (ABSA)** layer, specifically isolating customer feedback across three distinct target dimensions: *Skin, Hair, and General Product* aspects.
* **Imbalance Resolution:** Mitigated a severe **79.2% raw negative/neutral class bias** by implementing a targeted **undersampling strategy**, establishing a perfectly balanced subset of **1,664 reviews (832 negative/neutral and 832 positive rows)** for robust exploratory data analysis and model training.
* **Domain Feature Engineering:** Embedded a custom **User–Product Compatibility Alignment Layer** that flags direct matches between a consumer’s biological traits (*skin type, skin tone, hair attributes*) and a product’s specific formulation.
* **Model Optimization:** Systematically trained and optimized multiple predictive architectures (**Majority Baseline, Logistic Regression, Random Forest, and Linear SVM**) using **GridSearchCV** combined with a robust **Stratified 3-Fold Cross-Validation** strategy to reduce overfitting and improve generalization performance.
* **Performance Comparison:** Rather than a single model overwhelmingly dominating all aspects, the execution metrics revealed shared strengths across candidate architectures. **Logistic Regression** achieved the highest Macro F1-Score (**0.7684**), **Random Forest** achieved the highest Balanced Accuracy (**0.8658**), while **Linear SVM** delivered the highest overall discriminatory power with an ROC-AUC Score of **0.9085**. The relatively narrow performance margins among these three tuned algorithms indicate that our engineered domain feature space contributed substantially to stable predictive success.

---

* **Features Used:** TF-IDF Vectorization + DeBERTa ABSA Sentiments + User-Product Alignment Rules
* **Final Performance Breakdown:**
  * **Highest Macro F1-Score:** 0.7684 *(Logistic Regression)*
  * **Highest Balanced Accuracy:** 86.58% *(Random Forest)*
  * **Highest ROC-AUC Score:** 0.9085 *(Linear SVM)*
    
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

