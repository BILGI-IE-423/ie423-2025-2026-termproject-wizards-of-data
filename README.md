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
The objective of this project is to analyze and predict customer satisfaction as reflected by product ratings in the Sephora Products and Skincare Reviews dataset. The project aims to integrate structured data, such as product attributes, brand information, and ingredient details, with unstructured textual data from customer reviews, which will be transformed into sentiment scores using natural language processing techniques. By combining these diverse feature groups, the study seeks to identify the key factors that drive product ratings and to develop predictive models that can accurately estimate customer satisfaction. In addition, the project investigates the alignment between user characteristics—such as skin type, skin tone, and hair color—and product highlights, examining whether personalized matches lead to higher ratings. Different machine learning algorithms will be applied and compared to determine which methods perform best in predicting product ratings, while careful feature engineering and data preprocessing will ensure meaningful and interpretable results. Ultimately, this project aims not only to build predictive models, but also to provide insights into the relative importance of various factors, uncover patterns in customer rating behavior, and understand the role of personalization in shaping user satisfaction.

## Repository Structure

```text
├── README.md                               • Project overview and instructions
├── requirements.txt                        • Python libraries required to run scripts
│
├── data/                                   • Dataset storage
│   ├── raw/                                • Original/unprocessed files or download instructions
│   ├── processed/                          • Cleaned/processed dataset
│   └── README.md                           • How to obtain dataset and dataset description
│
├── scripts/                                • Python scripts for project workflow
│   ├── 01_load_data.py                     • Load and combine datasets
│   ├── 02_preprocess_data.py               • Clean, handle missing values, feature engineering
│   └── 03_basic_eda.py                     • Exploratory data analysis
│
├── outputs/                                • Analysis results
│   ├── figures/                            • Plots and visualizations
│   ├── tables/                             • Summary tables
│
└── docs/                                   • Documentation
    └── ResearchProposalPreprocessing.md    • Preprocessing notes & project proposal
