import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier

# Uses KNN to calculate the missing values for numerical columns 
# and mode to calcuate missing values for categorial columns.
def clean_data(data):
    # Replace '?' with NaN
    data.replace('?', np.nan, inplace=True)

    # Copy the original DataFrame to avoid modifying the original data
    df = data.copy()

    # Define numerical columns
    numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    # Impute missing values in numerical columns using KNN
    imputer = KNNImputer(n_neighbors=2)
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    # Define categorical columns
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

    # Impute missing values in categorical columns
    for col in categorical_cols:
        # Fill missing values with the most frequent value (mode)
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode only the categorical columns that have missing values
    df_encoded = pd.get_dummies(df, columns=[col for col in categorical_cols if df[col].isnull().any()], drop_first=True)

    return df_encoded

# cleans the data by using mode to fill in missing values
def clean_data_simple(data):
    # Replace '?' with NaN
    data.replace('?', np.nan, inplace=True)

    # For numerical columns
    numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

    # For categorical columns
    categorical_cols = ['workclass', 'occupation', 'native-country']
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])

    # Return cleaned data
    return data

# Used to perform one-hot encoding on categorical columns in both 
# the training and test datasets. It ensures that both the training 
# and test datasets have the same categorical feature encoding.
def encode_data(train, test):
    # Concatenate train and test data to ensure uniform feature encoding
    combined = pd.concat([train, test])

    # Use one-hot encoding on categorical columns to avoid introducing ordinality
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    combined = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)

    # Split back into train and test datasets
    train_encoded = combined.iloc[:len(train)]
    test_encoded = combined.iloc[len(train):]
    return train_encoded, test_encoded

# calculates the Accuracy, F1-score, Precision, Confusion Matrix, TP, 
# FP, FN, TN, Recall, Specificity (True Negative Rate), False Positive 
# Rate (FPR), and False Negative Rate (FNR)
def scores(y_test, y_pred):
    #Initialize Label Encoder
    label_encoder = LabelEncoder()

    # Transform the test data labels to binary labels
    y_test_encoded = label_encoder.fit_transform(y_test)

    # Transform the predicted labels to binary labels
    y_pred_encoded = label_encoder.transform(y_pred)

    accuracy = accuracy_score(y_test, y_pred) * 100 #Calculate accuracy``
    f1 = f1_score(y_test_encoded, y_pred_encoded) #Calculate F1-score
    precision = precision_score(y_test_encoded, y_pred_encoded) #Calculate precision
    conf_matrix = confusion_matrix(y_test_encoded, y_pred_encoded) #Calculate confusion matrix --> Positive(1) = '>50K', Negative(0) = '<=50K'
    TN, FP, FN, TP = conf_matrix.ravel() #Extract (TN), (FP), (FN), and (TP) from the confusion matrix
    recall = TP / (TP + FN) #Calculate Recall (True Positive Rate)
    specificity = TN / (TN + FP) #Calculate Specificity (True Negative Rate)
    FPR = FP/(FP + TN) #Calculate False Positive Rate (FPR)
    FNR = FN / (FN + TP) #Calculate False Negative Rate (FNR)

    # Print evaluation metrics
    print("Accuracy:", "{:.4f}%".format(accuracy))
    print("F1-score:", "{:.4f}".format(f1))
    print("Precision:", "{:.4f}".format(precision))
    print("Confusion Matrix:")
    print(conf_matrix)
    print("TP:", TP)
    print("FP:", FP)
    print("FN:", FN)
    print("TN:", TN)
    print("Recall:", "{:.4f}".format(recall))
    print("Specificity (True Negative Rate):", "{:.4f}".format(specificity))
    print("False Positive Rate (FPR):", "{:.4f}".format(FPR))
    print("False Negative Rate (FNR):", "{:.4f} \n".format(FNR))
    
# Selects the top 10 features using SelectKBest
def feature_selection(estimator, X_train, y_train):
    # Perform feature selection using SelectKBest with ANOVA F-value
    selector = SelectKBest(score_func=f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    # Apply selected features to both training and test datasets
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected

# Trains each model. If feature is set to 1, then we perform feature 
# selection before training. If feature is set to 0, we train the 
# model on the data as is.
def models(model, X_train, y_train, name, feature):
    if feature == 1:
        # Select features Select K Best
        X_train_select, X_test_select = feature_selection(model, X_train, y_train)
        # Initialize and train classifier
        model.fit(X_train_select, y_train)

        # Predict and evaluate the model on the original, non-resampled test data
        predictions = model.predict(X_test_select)

        print(f'Evaluation Metrics for {name}: ')
        scores(y_test, predictions)
    else:
        # Initialize and train classifier
        model.fit(X_train, y_train)

        # Predict and evaluate the model on the original, non-resampled test data
        predictions = model.predict(X_test)

        print(f'Evaluation Metrics for {name}: ')
        scores(y_test, predictions)
    
# Below we are setting up the data. We first clean the data then balance it using SMOTE    
# Define column names based on your data description
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

# Load your data -- update to your relative path
data = pd.read_csv('census-income.data.csv', names=column_names)
test_data = pd.read_csv('census-income.test.csv', names=column_names)

# Clean both the training and testing datasets
data = clean_data(data)
test_data = clean_data(test_data)

# Encode training and testing datasets
data, test_data = encode_data(data, test_data)

# Split the data into features and target variable
X_train = data.drop('income', axis=1)  
y_train = data['income']

# Remove trailing period from income column in test data
test_data['income'] = test_data['income'].str.strip('.')

# Split the test data into features and target variable
X_test = test_data.drop('income', axis=1)
y_test = test_data['income']

# Initialize SMOTE and resample the training data
smote = SMOTE(random_state=42)
X_train_resample, y_train_resample = smote.fit_resample(X_train, y_train)


# Finds the best k for KNN using imbalanced data and no feature selection
k_val = [3, 10, 15, 20, 25, 30]
best_accuracy_knn = 0
best_k_knn = None

for k in k_val:
    knn_classifier = KNeighborsClassifier(k)
    knn_classifier.fit(X_train, y_train)
    predictions_knn = knn_classifier.predict(X_test)
    accuracy_knn = accuracy_score(y_test, predictions_knn)
    print("Accuracy of KNN with k = ", k, ": ", accuracy_knn)

    # Update best accuracy and best k
    if accuracy_knn > best_accuracy_knn:
        best_accuracy_knn = accuracy_knn
        best_k_knn = k

print("\nBest accuracy of KNN:", "{:.4f}%".format(best_accuracy_knn * 100), "achieved with k =", best_k_knn)

# Finds the best k for Bagging with KNN using imbalanced data and no feature selection
k_val = [3, 10, 20, 25, 30]
best_accuracy_bagging = 0
best_k_bagging = None
for k in k_val:
    knn_classifier = KNeighborsClassifier(k)
    bagging_knn = BaggingClassifier(knn_classifier, n_estimators=10, random_state=42)
    # Train bagging ensembles
    bagging_knn.fit(X_train, y_train)
    # Make predictions
    predictions_bknn = bagging_knn.predict(X_test)
    # Evaluate accuracy of each ensemble
    accuracy_knn = accuracy_score(y_test, predictions_bknn)
    print("Accuracy of Bagging with KNN, k = ", k, ": ", accuracy_knn)

    # Update best accuracy and best k
    if accuracy_knn > best_accuracy_bagging:
        best_accuracy_bagging = accuracy_knn
        best_k_bagging = k

print("\nBest accuracy of Bagging with KNN:", "{:.4f}%".format(best_accuracy_bagging * 100), "achieved with k =", best_k_bagging)


# Runs each model based on the data and feature selection parameters
# Data can be balanced or imbalanced
# feature_selection can either be 0 for no feature selection or
# 1 for feature selection
def run_models(X_train, y_train, X_test, y_test, feature_selection):
    # Random Forest Implementation
    models(RandomForestClassifier(), X_train, y_train,'Random Forest', feature_selection)

    # Random Forest with Bagging
    rf_classifier = RandomForestClassifier()
    bagging_rf = BaggingClassifier(estimator=rf_classifier, n_estimators=10, random_state=42)
    models(bagging_rf, X_train, y_train, 'Bagging with Random Forest', feature_selection)

    # KNN
    models(KNeighborsClassifier(20), X_train, y_train, 'KNN with k=20', feature_selection)

    # KNN with Bagging
    knn_classifier = KNeighborsClassifier(25)
    bagging_knn = BaggingClassifier(knn_classifier, n_estimators=10, random_state=42)
    models(bagging_knn, X_train, y_train, 'Bagging with KNN, k=25', feature_selection)

    # Logistic regression model
    models(LogisticRegression(), X_train, y_train, 'Logistic Regression', feature_selection)

    # Naive Bayes
    models(GaussianNB(), X_train, y_train, 'Naive Bayes', feature_selection)

    # Naive Bayes with Laplace Smoothing
    models(MultinomialNB(alpha=1.0), X_train, y_train, 'Naive Bayes with Laplace Smoothing', feature_selection)

    # Decision Tree Implementation
    models(DecisionTreeClassifier(), X_train, y_train, 'Decision Tree', feature_selection)

    # Ensemble Learning
    # Initialize individual classifiers
    rf_classifier = RandomForestClassifier(random_state=42)
    knn_classifier = KNeighborsClassifier(25)
    logistic_reg = LogisticRegression()
    nb_classifier = MultinomialNB(alpha=1.0)
    dt_classifier = DecisionTreeClassifier()
    b_classifier = BaggingClassifier(estimator=rf_classifier, n_estimators=10, random_state=42)

    if feature_selection == 1:
      # Feature selection fails with logistic regression
      # Define the ensemble classifier with heterogeneous base models
      ensemble_classifier = VotingClassifier(
        estimators=[
            ('rf', rf_classifier),
            ('knn', knn_classifier),
            ('nb', nb_classifier),
            ('dt', dt_classifier),
            ('b', b_classifier)
        ],
        voting='soft', weights=[4, 1, 1, 1, 1])
    else:
      # Define the ensemble classifier with heterogeneous base models
      ensemble_classifier = VotingClassifier(
          estimators=[
              ('rf', rf_classifier),
              ('knn', knn_classifier),
              ('lr', logistic_reg),
              ('nb', nb_classifier),
              ('dt', dt_classifier),
              ('b', b_classifier)
          ],
          voting='soft', weights=[4, 1, 1, 1, 1, 1])

    models(ensemble_classifier, X_train, y_train, 'Ensemble Classifier', feature_selection)
    
# Run all the models with each parameter 
# For each model we compute the following: Accuracy, F1-score, Precision, 
# Confusion Matrix, Recall, Specificity (True Negative Rate), False Positive 
# Rate (FPR), False Negative Rate (FNR)

# Runs all the models with balanced data without feature selection
print('Running all the models with balanced data without feature selection')
run_models(X_train_resample, y_train_resample, X_test, y_test, 0)

# Runs all the models with balanced data with feature selection
print('Running all the models with balanced data with feature selection')
run_models(X_train_resample, y_train_resample, X_test, y_test, 1)

# Runs all the models with imbalanced data without feature selection
print('Running all the models with imbalanced data without feature selection')
run_models(X_train, y_train, X_test, y_test, 0)

# Runs all the models with imbalanced data with feature selection
print('Running all the models with imbalanced data with feature selection')
run_models(X_train, y_train, X_test, y_test, 1)


# Below are our data visualizers 
# These help with understanding the data and correltations between the features 
# and the target variable. Classifiers, such as Decision Tree and Random Forest, 
# are used to see how the data is interpreted by the model

decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(X_train_resample, y_train_resample)
# Plot the decision tree
plt.figure(figsize=(20,10))  # Adjust the figure size as needed
plot_tree(decision_tree_classifier, filled=True, feature_names=X_train.columns, class_names=decision_tree_classifier.classes_)
plt.savefig('decision_tree_plot.png')  # Save the plot as a PNG file
plt.show()

# Visualizing the distribution of target variable 'income' in the training data
sns.countplot(x='income', data=data)
plt.title('Income Distribution in Training Data')
plt.show()
plt.savefig('income_distribution_training.png')  # Save the plot as a PNG file

# Get value counts of income in training data
income_distribution_train = data['income'].value_counts()
print("Income Distribution in Training Data:")
print(income_distribution_train)

# Convert the predicted incomes to a DataFrame for visualization
# Initialize and train classifier
classifier = RandomForestClassifier(random_state=52)
classifier.fit(X_train, y_train)
predictions_rf = classifier.predict(X_test)
predicted_df = pd.DataFrame({"income": predictions_rf})

# Visualizing the distribution of target variable 'income' in the testing data using RF
sns.countplot(x='income', data=predicted_df)
plt.title('Income Distribution in Testing Data Predicted with Random Forest')
plt.show()
plt.savefig('predicted_income_distribution_testing.png')  # Save the plot as a PNG file


# Get value counts of predicted income
income_distribution_test = predicted_df['income'].value_counts()
print("\nPredicted Income Distribution in Testing Data:")
print(income_distribution_test)


# Random Forest with balanced data
# Convert the predicted incomes to a DataFrame for visualization
# Initialize and train classifier
classifier = RandomForestClassifier(random_state=52)
classifier.fit(X_train_resample, y_train_resample)
predictions_rf = classifier.predict(X_test)
predicted_df = pd.DataFrame({"income": predictions_rf})

# Visualizing the distribution of target variable 'income' in the testing data using RF
sns.countplot(x='income', data=predicted_df)
plt.title('Income Distribution in Testing Data Predicted with Random Forest Using Balanced Training Data')
plt.show()
plt.savefig('predicted_income_distribution_testing.png')  # Save the plot as a PNG file


# Get value counts of predicted income
income_distribution_test = predicted_df['income'].value_counts()
print("\nPredicted Income Distribution in Testing Data with Balanced Training Data:")
print(income_distribution_test)

# Visualizing the correlation between numerical features

# Encode income
data_numeric = data
data_numeric['income'] = data_numeric['income'].apply(lambda x: 1 if x.strip() == '<=50K' else 0)
# Visualizing the correlation between numerical features

numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# Calculate correlation matrix for numerical columns
numerical_data = data[numerical_cols]
numerical_correlation_matrix = numerical_data.corr()

# Create a single subplot for the heatmap
fig, ax = plt.subplots(figsize=(12, 8))  # Adjusted figsize to your preference

# Plot correlation heatmap for numerical columns
sns.heatmap(numerical_correlation_matrix, ax=ax, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
ax.set_title('Correlation Heatmap of Numerical Variables')

plt.tight_layout()
plt.show()

# Create the decision tree classifier
# Limit the depth of the tree to prevent overfitting
clf = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# Predict the test set
predictions = clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Visualize the tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X_train.columns, class_names=['Low', 'High'], filled=True)
plt.show()

# Export the decision tree
tree_rules = export_text(clf, feature_names=list(X_train.columns))
print(tree_rules)