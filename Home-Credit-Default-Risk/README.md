# Home-Credit-Default-Risk
[Kaggle Home Credit Default Risk Competition](https://www.kaggle.com/c/home-credit-default-risk)

Can you predict how capable each applicant is of repaying a loan?

***
## Introduction
Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.

Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.


***
## Data
- application_{train|test}.csv
- bureau.csv
- bureau_balance.csv
- POS_CASH_balance.csv
- credit_card_balance.csv
- previous_application.csv
- installments_payments.csv
- HomeCredit_columns_description.csv
- [Download](https://www.kaggle.com/c/home-credit-default-risk/data)

## Major Challenge
- How to combine data from separate tables?
- Feature Engineering
- Imbalanced classification


***
## Data Cleaning
- [1. Data Cleaning and Processing.ipynb](https://github.com/JifuZhao/Home-Credit-Default-Risk/blob/master/1.%20Data%20Cleaning%20and%20Processing.ipynb)
    - Process bureau.csv and bureau_balance.csv.
    - Process previsous_application.csv, POS_CASH_balance.csv, credit_card_balance.csv, and installments_payments.csv.
    - Merge processed information with application_train.csv and application_test.csv.
    - Finally get training data with $307,511$ records and $320$ features, test data with $48,744$ records and $319$ features.


***
## Data Visualization
- [2. Data Visualization.ipynb](https://github.com/JifuZhao/Home-Credit-Default-Risk/blob/master/2.%20Data%20Visualization.ipynb)
    - Delete obviously useless features.
    - Visualize numerical features.
    - Visualize categorical features.


***
## Feature Engineering
- [3. Feature Engineering.ipynb](https://github.com/JifuZhao/Home-Credit-Default-Risk/blob/master/3.%20Feature%20Engineering.ipynb)
    - Missing values
    - Categorical features
    - Numerical features


***
## Modeling
- Implemented Models
    - Logistic Regression (h2o)
    - Random Forest (LightGBM)
    - Boosting (LightGBM)
    - Boosting (CatBoost)
    - Naive data vs. Over-sampling vs. SMOTE (imbalanced-learn)
- Corresponding Notebooks
    - [4. Naive Models.ipynb](https://github.com/JifuZhao/Home-Credit-Default-Risk/blob/master/4.%20Naive%20Models.ipynb)
    - [5. Balanced Models.ipynb]()
    - [6. Over-sampling Models.ipynb]()
    - [7. SMOTE Models.ipynb]()


***
## Result (Kaggle Public Leaderboard)
- Full Features
    - H2O Logistic Regression: AUC = 0.760
    - H2O Random Forest: AUC = 0.728
    - LightGBM Random Forest: AUC =
    - LightGBM Boosting: AUC =

- Over-sampling Models
    - Random Forest: AUC =
    - Boosting: AUC =
- SMOTE Models
    - Random Forest: AUC =
    - Boosting: AUC =


Copyright @ Jifu Zhao (2018)
