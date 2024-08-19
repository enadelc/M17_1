# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data/bank-additional-full.csv', sep=';')

# Display the first few rows of the dataset
data.head()

data.dropna(inplace=True)

# Drop the 'duration' column as it should not be used for a realistic predictive model
data = data.drop(columns=['duration'])

# Preprocess the data
# Encode the categorical variables
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 
                       'month', 'day_of_week', 'poutcome']

label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Encode the target variable
data['y'] = data['y'].map({'yes': 1, 'no': 0})

# Split the data into features and target variable
X = data.drop(columns=['y'])
y = data['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the numeric features
numeric_columns = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                   'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

scaler = StandardScaler()
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

# Initialize classifiers
knn = KNeighborsClassifier()
log_reg = LogisticRegression(random_state=42, max_iter=1000)
decision_tree = DecisionTreeClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)

# Train and evaluate classifiers
classifiers = {
    "K-Nearest Neighbors": knn,
    "Logistic Regression": log_reg,
    "Decision Tree": decision_tree,
    "Support Vector Machine": svm
}

# Store the results
results = {}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
    
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

# Display the results
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by="ROC-AUC", ascending=False)
print("Initial Results:")
print(results_df)

# Visualize the performance metrics
results_df.plot(kind='bar', figsize=(15, 8))
plt.title("Comparison of Classifier Performance")
plt.xlabel("Classifier")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc='lower right', prop={'size': 8})  # Set the legend font size to 8
plt.show()

# More feature engineering and exploration
# Determine if we should keep the 'gender' feature
# Check the distribution of 'gender' feature
#gender_counts = data['gender'].value_counts()
#print(gender_counts)

# If the 'gender' feature is imbalanced or does not provide useful information, we can drop it
#if gender_counts['male'] < 0.1 * len(data) or gender_counts['female'] < 0.1 * len(data):
#    data = data.drop(columns=['gender'])

# Hyperparameter tuning and grid search
# For each classifier, perform hyperparameter tuning using grid search

# K-Nearest Neighbors
knn_param_grid = {
    'n_neighbors': [3,  9],
    'weights': ['uniform', 'distance']
}
knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=3,  n_jobs=-1)
knn_grid_search.fit(X_train, y_train)
best_knn = knn_grid_search.best_estimator_
print("KNN Tuning Complete")
# Logistic Regression
log_reg_param_grid = {
    'C': [ 1, 5],
    'solver': ['liblinear', 'lbfgs']
}
log_reg_grid_search = GridSearchCV(log_reg, log_reg_param_grid, cv=3,  n_jobs=-1)
log_reg_grid_search.fit(X_train, y_train)
best_log_reg = log_reg_grid_search.best_estimator_
print("Log Reg Tuning Complete")
# Decision Tree
decision_tree_param_grid = {
    'max_depth': [3,  9],
    'min_samples_split': [2,  4]
}
decision_tree_grid_search = GridSearchCV(decision_tree, decision_tree_param_grid, cv=3 ,  n_jobs=-1)
decision_tree_grid_search.fit(X_train, y_train)
best_decision_tree = decision_tree_grid_search.best_estimator_
print("Decision Tree Tuning Complete")
# Support Vector Machine
svm_param_grid = {
    'C': [ 1, 5],
    'kernel': ['linear', 'rbf']
}
svm_grid_search = HalvingGridSearchCV(svm, svm_param_grid, cv=3, factor=2, n_jobs=-1)
svm_grid_search.fit(X_train, y_train)
best_svm = svm_grid_search.best_estimator_
print("SVD Tuning Complete")
# Print best hyperparameters for each classifier
print("Best Hyperparameters:")
print("K-Nearest Neighbors:", best_knn.get_params())
print("Logistic Regression:", best_log_reg.get_params())
print("Decision Tree:", best_decision_tree.get_params())
print("Support Vector Machine:", best_svm.get_params())
# Adjust performance metric
# Instead of using ROC-AUC, let's use F1-Score as the performance metric
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }

# Display the results
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by="F1-Score", ascending=False)
print("Adjusted Results:")
print(results_df)

# Visualize the performance metrics
results_df.plot(kind='bar', figsize=(15, 8))
plt.title("Comparison of Classifier Performance")
plt.xlabel("Classifier")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc='lower right', prop={'size': 8})  # Set the legend font size to 8
plt.show()