# IE 423 Term Project Proposal — Customer Satisfaction in Sephora Products: A Data-Driven Machine Learning Approach

## Team Members
- İlke Bakangil
- Sıla Doğan
- Avsu Evren
- Yağmur Battal
## Dataset
Dataset: Sephora Products and Skincare Reviews 

Source: https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews/data

## Project Objective
The objective of this project is to analyze and predict customer satisfaction as reflected by product ratings in the Sephora Products and Skincare Reviews dataset. The project aims to integrate structured data, such as product attributes, brand information, and ingredient details, with unstructured textual data from customer reviews, which will be transformed into sentiment scores using natural language processing techniques. By combining these diverse feature groups, the study seeks to identify the key factors that drive product ratings and to develop predictive models that can accurately estimate customer satisfaction. 

In addition, the project investigates the alignment between user characteristics—such as skin type, skin tone, and hair color—and product highlights, examining whether personalized matches lead to higher ratings. Different machine learning algorithms will be applied and compared to determine which methods perform best in predicting product ratings, while careful feature engineering and data preprocessing will ensure meaningful and interpretable results. Ultimately, this project aims not only to build predictive models, but also to provide insights into the relative importance of various factors, uncover patterns in customer rating behavior, and understand the role of personalization in shaping user satisfaction.

## Repository Structure
```text
ie423-2025-2026-termproject-wizards-of-data/
│
├── README.md                               • Overview of the project, objectives, and instructions to run the code  
├── requirements.txt                        • List of required Python libraries  

├── data/                                   • Dataset directory  
│   ├── raw/                                • Original dataset files (see data/README.md for download instructions) 
│   ├── processed/                          • Cleaned dataset generated after preprocessing  
│   └── README.md                           • Instructions for downloading and placing the dataset  

├── scripts/                                • Python scripts for each project stage  
│   ├── 01_load_data.py                     • Loads and combines raw data files  
│   ├── 02_preprocess_data.py               • Cleans data, handles missing values, and creates new features  
│   └── 03_basic_eda.py                     • Performs exploratory data analysis and generates outputs  

├── outputs/                                • Generated analysis results  
│   ├── figures/                            • Saved plots and visualizations  
│   ├── tables/                             • Generated summary tables  

└── docs/                                   • Project documentation  
    └── ResearchProposalPreprocessing.md    • Main report including proposal and preprocessing steps
```

## Installation
```text
pip install -r requirements.txt
```
## Running the Scripts
```text
python scripts/01_load_data.py
python scripts/02_preprocess_data.py
python scripts/03_basic_eda.py
```
## Proposal Document
See: docs/ResearchProposalPreprocessing.md
