### Loan Approval Prediction
## Project Overview
This project focuses on predicting whether a loan will be approved or not, based on customer financial data and loan application details. By analyzing key features such as income, loan amount, credit score, and interest rate, the model aims to assist lenders in making data-driven decisions. The project was part of a Kaggle competition, "Loan Approval Prediction," and is designed to be applied to real-world loan risk assessment scenarios.

## Dataset
## Source: Kaggle Loan Approval Prediction Dataset
## Features:
person_income: Applicant’s income
loan_amnt: Loan amount requested
loan_int_rate: Loan interest rate
person_emp_length: Employment length
And others related to credit history and demographic details.
Goal
The goal is to classify loan applications as either approved (1) or denied (0) using a variety of machine learning algorithms.

## Key Findings
Model Selection: After testing several algorithms (Logistic Regression, Random Forest, and XGBoost), XGBoost performed best.
Performance:
XGBoost achieved an accuracy of 95.05% and an AUC score of 0.9541 on the training data.
Key predictors like person_income, loan_amnt, and loan_int_rate played a major role in driving the model’s predictions.
Insights: Income and loan-related features were the most significant factors affecting loan approval, highlighting the importance of financial stability in lending decisions.
Tools and Techniques
Programming Language: Python
Libraries:
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib
Seaborn

## Key Techniques:
Data Preprocessing: Handled missing values, feature encoding (one-hot encoding), and scaling.
Model Evaluation: Accuracy, AUC-ROC curve, confusion matrix, and cross-validation were used to assess model performance.
Hyperparameter Tuning: Performed grid search to optimize model parameters for Random Forest and XGBoost.
Feature Importance Analysis: Identified the most impactful features using XGBoost's built-in importance metrics.
Feature Importance Analysis
The top contributing features in the final model:

person_income
loan_amnt
loan_int_rate
person_emp_length
This analysis highlights that the financial situation of an applicant, including income and loan-specific factors, are critical to predicting loan approvals.

## How to Use
1. Clone the Repository
bash
Copy code
git clone https://github.com/your-username/loan-approval-prediction.git
2. Install Dependencies
Ensure that all necessary libraries are installed by running:

bash
Copy code
pip install -r requirements.txt
3. Run the Project
Open the Jupyter Notebook to explore the analysis:

bash
Copy code
jupyter notebook Loan_Approval_Prediction.ipynb
4. Predict on New Data
To use the trained model to predict loan statuses on a new dataset:

python
Copy code
# Load new data (ensure it's preprocessed similarly)
new_data = pd.read_csv('new_test_data.csv')
# Predict
predictions = xgb_model.predict(new_data)

## Results
Final Model: XGBoost
Accuracy: 95.05%
AUC: 0.9541 
Challenges and Future Work
Imbalanced Data: The dataset had some class imbalance. Future work could focus on using techniques like SMOTE or adjusting class weights to further improve performance on underrepresented classes.
Feature Engineering: More domain-specific feature engineering could be applied to improve the model’s interpretability and accuracy, especially focusing on credit history and employment duration.
Acknowledgments

This project is based on the competition hosted by Kaggle:

Walter Reade and Ashley Chow. Loan Approval Prediction. Kaggle, 2024.
License
