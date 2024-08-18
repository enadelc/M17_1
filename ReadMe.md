Bank Marketing Campaign Classifier
Overview
This project aims to predict whether a client will subscribe to a term deposit based on the results of multiple marketing campaigns conducted by a Portuguese banking institution. The goal is to compare the performance of four machine learning classifiers—K-Nearest Neighbors (KNN), Logistic Regression, Decision Trees, and Support Vector Machines (SVM)—and evaluate their effectiveness in predicting the outcome of the campaign.

Dataset
The dataset used in this project is from the UCI Machine Learning Repository. It contains information about various marketing campaigns, including the client's demographic information, campaign-related features, and economic context.

Input Variables
Age: Client's age (numeric)
Job: Type of job (categorical)
Marital: Marital status (categorical)
Education: Education level (categorical)
Default: Has credit in default? (categorical)
Housing: Has housing loan? (categorical)
Loan: Has personal loan? (categorical)
Contact: Contact communication type (categorical)
Month: Last contact month of year (categorical)
Day of Week: Last contact day of the week (categorical)
Campaign: Number of contacts performed during this campaign and for this client (numeric)
Pdays: Number of days that passed by after the client was last contacted from a previous campaign (numeric)
Previous: Number of contacts performed before this campaign for this client (numeric)
Poutcome: Outcome of the previous marketing campaign (categorical)
Emp.var.rate: Employment variation rate - quarterly indicator (numeric)
Cons.price.idx: Consumer price index - monthly indicator (numeric)
Cons.conf.idx: Consumer confidence index - monthly indicator (numeric)
Euribor3m: Euribor 3 month rate - daily indicator (numeric)
Nr.employed: Number of employees - quarterly indicator (numeric)
Target Variable
y: Has the client subscribed to a term deposit? (binary: "yes" or "no")
Project Structure
lua
Copy code
|-- data/
|   |-- bank-additional-full.csv
|
|-- Bank_Marketing_Classifier.ipynb
|
|-- README.md
1. data/bank-additional-full.csv
This directory contains the dataset used for the project.

2. Bank_Marketing_Classifier.ipynb
This Jupyter Notebook file includes the following sections:

Data Loading: Load the dataset into a Pandas DataFrame.
Data Preprocessing: Encode categorical variables and scale numeric features.
Modeling: Train and evaluate four different classifiers.
Results Visualization: Visualize the performance metrics of the classifiers.
3. README.md
This file provides an overview of the project, dataset, project structure, setup instructions, and usage.

Setup Instructions
Prerequisites
Before running the notebook, ensure you have the following installed:

Python 3.6 or later
Jupyter Notebook or JupyterLab
Python libraries:
pandas
numpy
scikit-learn
seaborn
matplotlib
Installation
Clone the repository:

Launch the Jupyter Notebook:

Copy code
jupyter notebook Bank_Marketing_Classifier.ipynb
Usage
Run the Notebook: Open the Bank_Marketing_Classifier.ipynb file in Jupyter Notebook and run the cells sequentially to execute the analysis.

Model Comparison: Review the performance of the classifiers in the notebook's output to determine which model is best suited for predicting term deposit subscriptions.

Next Steps: Explore further improvements by tuning hyperparameters, experimenting with other models, or conducting additional feature engineering.

Results
The notebook will output a comparison of the classifiers based on several performance metrics:

Accuracy
Precision
Recall
F1-Score
ROC-AUC
A bar chart will also be displayed to visualize the performance of each classifier.


Conclusion

This project demonstrates the application of various machine learning models to predict the success of marketing campaigns. By comparing the performance of KNN, Logistic Regression, Decision Trees, and SVM, we can identify the most effective model for predicting whether a client will subscribe to a term deposit.


License

This project is licensed under the MIT License - see the LICENSE file for details.


Acknowledgments

The dataset was provided by the UCI Machine Learning Repository.
This project was developed as a practical application of machine learning concepts.
