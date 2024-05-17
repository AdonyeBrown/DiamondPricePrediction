# Diamond Price Prediction

![th](https://github.com/AdonyeBrown/fullDataScienceSetup/assets/134440796/07095def-04c2-4426-8e05-5b698e39937b)

## Overview
This project aims to predict the price of diamonds based on various features such as carat, cut, color, and clarity using machine learning techniques. The goal is to provide an accurate estimation that can help in the valuation of diamonds.

## Dataset
The dataset used in this project includes the following attributes:
- **Carat**: Weight of the diamond.
- **Cut**: Quality of the cut (Fair, Good, Very Good, Premium, Ideal).
- **Color**: Diamond color, from J (worst) to D (best).
- **Clarity**: How clear the diamond is (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF).
- **X**: Length in mm.
- **Y**: Width in mm.
- **Z**: Depth in mm.
- **Depth**: Total depth percentage = $$\frac{z}{\text{mean}(x, y)} \times 100$$
- **Table**: The width of the diamond's top facet in percentage.
- **Price**: The price of the diamond (target variable).

## Features
- Exploratory Data Analysis (EDA) to understand the dataset.
- Feature Engineering to create new features that can aid in prediction.
- Model Selection to choose the best performing machine learning model.
- Model Evaluation to assess the accuracy of the predictions.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Installation
To set up the project environment, run the following commands:
```bash
git clone https://github.com/AdonyeBrown/fullDataScienceSetup.git
cd diamond-price-prediction
pip install -r requirements.txt
