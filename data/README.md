
# Dataset Description

## Source
This project uses the "Sephora Products and Skincare Reviews" dataset from Kaggle.

## Dataset Link
https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews

## Files
The dataset includes the following files:
- product_info.csv
- reviews_0-250.csv
- reviews_250-500.csv
- reviews_500-750.csv
- reviews_750-1250.csv
- reviews_1250-end.csv

## How to Obtain the Dataset
1. Go to the dataset link above
2. Download the dataset manually from Kaggle
3. Extract the downloaded zip file

## Where to Place the Files
After downloading and extracting, place the files inside the following directory:

data/raw/

The expected folder structure is:

data/
  raw/
     product_info.csv
     reviews_0-250.csv
     reviews_250-500.csv
     reviews_500-750.csv
     reviews_750-1250.csv
     reviews_1250-end.csv

## Notes
- The dataset is not included in this repository due to its large size.
- Manual download is required to run the code.
- Make sure the file paths in the code match the structure above.
  
## Processed Data

The processed dataset (`cleaned_data.csv`) is not included in the repository due to its size.

It can be downloaded from the following link:

[Cleaned Dataset](https://drive.google.com/file/d/1hmojSxUpNTYt6nFqC2A71e5dG4HCVNh5/view?usp=drive_link)

Alternatively, it can be reproduced by running:

```bash
python scripts/02_preprocess_data.py

