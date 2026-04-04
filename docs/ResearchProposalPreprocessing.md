# IE 423 Term Project Proposal — A Data-Driven Analysis of Customer Satisfaction in Sephora Products Using Machine Learning and Consumer Review Data

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
