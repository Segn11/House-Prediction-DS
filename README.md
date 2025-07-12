# 🏘️ Buenos Aires Apartment Price Prediction

This project uses machine learning to **predict apartment prices** in Capital Federal, Buenos Aires, based on features like **size**, **location**, and **neighborhood**.

## 📊 Problem Statement

Given apartment data such as:
- Covered surface area (in square meters)
- Latitude & Longitude
- Neighborhood

The goal is to build a regression model that accurately predicts the **approximate price in USD**.

---

## 📂 Dataset

The project uses cleaned and combined `.csv` files from the Buenos Aires real estate dataset:

- **Source files:** `buenos-aires-real-estate-*.csv`
- **Test file:** `buenos-aires-test-features.csv`

The dataset is preprocessed to:
- Focus only on apartments in **Capital Federal**
- Filter outliers in surface area
- Extract and split location info (latitude & longitude)
- Create a clean **neighborhood** feature

---

🧹 Data Wrangling

The `wrangle()` function handles:
- Filtering apartments under $400,000
- Splitting the `"lat-lon"` column into separate features
- Creating `"neighborhood"` from hierarchical location data
- Dropping irrelevant or noisy columns

---

 🧠 Model

A regression pipeline is built using:
- `category_encoders.OneHotEncoder` for categorical encoding
- `SimpleImputer` for missing value handling
- `Ridge` regression for better generalization over `LinearRegression`

 Baseline:
- Predicts the **mean apartment price** for all samples
- Used to compare against model performance

