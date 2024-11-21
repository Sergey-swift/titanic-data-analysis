# Author: Sergey Swift
# Project: Titanic Data Analysis
# Date: November 2024
# Dataset Source: https://www.kaggle.com/datasets/yasserh/titanic-dataset

# %% Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Train-test split import
from sklearn.model_selection import train_test_split

# Basic ML algorithm imports (Linear, Logistic, KNN)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Scaler for KNN
from sklearn.preprocessing import StandardScaler

# Metrics for model evaluation
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

# sns style selection
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')

# Functions definitions
def load_data(filepath):
    '''Load data from CSV file and return a DataFrame'''
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Perform initial preprocessing on the DataFrame"""
    # Fill missing Age data with median values
    df.fillna({'Age': df['Age'].median()}, inplace=True)
    # Keep a copy of the original 'Embarked' column for plotting
    df['Embarked_original'] = df['Embarked']
    # Fill missing Embarked data with mode
    df.fillna({'Embarked_original': df['Embarked'].mode()[0]}, inplace=True)
    # Optionally, drop rows with missing Fare or other critical values
    # Handle remaining missing values
    df.dropna(inplace=True)
    # Create is_Alone column
    df['is_Alone'] = ((df['SibSp'] + df['Parch']) == 0).astype(int)
    # Encode categorical variables
    categorical_columns = ['Sex', 'Embarked']  # Replace with your categorical columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
    return df

def basic_statistic(df):
    """Display basic statistics and info about the DataFrame"""
    print('DataFrame Information:')
    # Basic info about df
    df.info()
    print('\nDescriptive Statistics:')
    print(df.describe())

def visualisations_of_df(df):
    # Heatmap of Correlation Between Features
    plt.figure(figsize=(10, 8))
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Features')
    plt.show()

    # Stacked Bar Plot of Survival Count by Passenger Class
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Pclass', hue='Survived', data=df, palette='viridis')
    plt.title('Survival Count by Passenger Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Number of Passengers')
    plt.legend(title='Survived', labels=['Did Not Survive', 'Survived'])
    plt.show()

    # Plot Age Distribution by Passenger Class and Survival Status
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Pclass', y='Age', hue='Survived', data=df, palette='coolwarm')
    plt.title('Age Distribution by Passenger Class and Survival Status')
    plt.xlabel('Passenger Class')
    plt.ylabel('Age')
    plt.legend(title='Survived')  # Automatically handles distinct colors
    plt.show()

    # Plot Fare Distribution by Survival Status and Passenger Class
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Survived', y='Fare', hue='Pclass', data=df, palette='muted')
    plt.title('Fare Distribution by Survival Status and Passenger Class')
    plt.xlabel('Survived')
    plt.ylabel('Fare')
    plt.xticks([0, 1], ['Did Not Survive', 'Survived'])
    plt.legend(title='Passenger Class')
    plt.show()

    # Plot Count of Passengers by Embarkation Point and Passenger Class
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Embarked_original', hue='Pclass', data=df, palette='Set2')
    plt.title('Count of Passengers by Embarkation Point and Passenger Class')
    plt.xlabel('Embarked')
    plt.ylabel('Number of Passengers')
    plt.legend(title='Passenger Class')
    plt.show()

    # Plot Sex Distribution within Each Passenger Class
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Pclass', hue='Sex_male', data=df, palette='Set1')
    plt.title('Gender Distribution by Passenger Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Number of Passengers')
    plt.legend(title='Gender', labels=['Female', 'Male'])
    plt.show()

    # KDE plot of Age by Survival Status
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=df[df['Survived'] == 0]['Age'], label='Did Not Survive',color='darkred', fill=True,)
    sns.kdeplot(data=df[df['Survived'] == 1]['Age'], label='Survived', fill=True)
    plt.title('Age Distribution by Survival Status')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Pairplot with 'Survived' hue
    pairplot_columns = ['Survived', 'Pclass', 'Fare', 'Age', 'SibSp', 'Parch', 'is_Alone']
    if all(col in df.columns for col in pairplot_columns):
        sns.pairplot(df[pairplot_columns], hue='Survived', palette='pastel')
        plt.suptitle('Pairplot of Key Numerical Features Colored by Survival Status', y=1.02)
        plt.show()
    else:
        print("One or more columns required for pairplot are missing.")

    # Linear regression plot for Age vs. Fare
    sns.lmplot(
        x='Age', y='Fare', data=df,
        scatter_kws={'color': 'skyblue'},
        line_kws={'color': 'navy'}
    )
    plt.xlabel('Age')
    plt.ylabel('Fare Price')
    plt.title('Linear Regression: Age vs. Fare Price')
    plt.show()

# Linear model
def train_and_predict_Lm(X, y, test_size=0.3, random_state=101):
    # Declare model
    lm = LinearRegression()
    # Train-test split for linear regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    # Fit the model
    lm.fit(X_train, y_train)
    # Make predictions
    predictions_lm = lm.predict(X_test)
    # Retrieve coefficients
    coefficients = lm.coef_
    # Print out model's metrics
    print('Coefficients: \n', coefficients)
    print('MAE:', metrics.mean_absolute_error(y_test, predictions_lm))
    print('MSE:', metrics.mean_squared_error(y_test, predictions_lm))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions_lm)))
    # Return predictions and coefficients
    return predictions_lm, coefficients, y_test

def visualize_lm_results(y_test, predictions_lm):
    # Plotting the results of linear regression
    plt.scatter(y_test, predictions_lm)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
             color='red', linestyle='--')  # Perfect prediction line
    plt.title('Linear Regression Predictions')
    plt.xlabel('Actual Fare')
    plt.ylabel('Predicted Fare')
    plt.show()

# Logistic model
def train_and_predict_logm(X, y, test_size=0.3, random_state=42):
    # Declare model
    logmodel = LogisticRegression()
    # Train-test split for logistic regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    logmodel.fit(X_train, y_train)
    # Predict
    predictions_log = logmodel.predict(X_test)
    # Metrics
    print("Classification Report:\n", classification_report(y_test, predictions_log))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions_log))
    return predictions_log, y_test, logmodel, X_train, X_test, y_train, y_test

def visualize_logm_confusion_matrix(y_test, predictions_log):
    # Visualize confusion matrix with hardcoded class labels
    cm = confusion_matrix(y_test, predictions_log)
    classes = ['Did Not Survive', 'Survived']  # Hardcoded class labels
    # Validate class labels
    unique_labels = np.unique(y_test)
    if len(classes) != len(unique_labels):
        print("Warning: Hardcoded classes do not match dataset labels.")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Data scaler for KNN algorithm
def data_scaler(data, target_column):
    """
    Scales the features of a dataset for the KNN algorithm.

    Args:
        data: A pandas DataFrame containing the dataset.
        target_column: The name of the target column to exclude from scaling.

    Returns:
        scaled_df: A DataFrame with scaled features.
    """
    scaler = StandardScaler()
    features = data.drop(target_column, axis=1)
    # Select only numeric columns
    numeric_features = features.select_dtypes(include=[np.number])
    scaled_features = scaler.fit_transform(numeric_features)
    scaled_df = pd.DataFrame(scaled_features, columns=numeric_features.columns)
    return scaled_df

# K-Nearest Neighbor Algorithm
def train_and_predict_KNN(scaled_df, target, test_size=0.3, random_state=42):
    """
    Train and predict using the KNN algorithm.

    Args:
        scaled_df: Scaled feature DataFrame.
        target: Target column values (Series or array).
        test_size: Proportion of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        predict_knn: Predicted target values for the test set.
        y_test: Actual target values for the test set.
        knn: Trained KNN model.
        X_train, X_test, y_train, y_test: Train-test split data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_df, target, test_size=test_size, random_state=random_state)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    # Predict
    predict_knn = knn.predict(X_test)
    # Metrics
    print("Confusion Matrix:\n", confusion_matrix(y_test, predict_knn))
    print("\nClassification Report:\n", classification_report(y_test, predict_knn))
    return predict_knn, y_test, knn, X_train, X_test, y_train, y_test

def visualize_knn_confusion_matrix(y_test, predict_knn):
    # Visualize confusion matrix with hardcoded class labels
    cm = confusion_matrix(y_test, predict_knn)
    classes = ['With Family', 'Alone']  # Hardcoded class labels
    # Validate class labels
    unique_labels = np.unique(y_test)
    if len(classes) != len(unique_labels):
        print("Warning: Hardcoded classes do not match dataset labels.")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Optimize K values
def optimise_k_values(scaled_df, target, test_size=0.3, random_state=42):
    """
    Optimize K value for KNN using error rate.

    Args:
        scaled_df: Scaled feature DataFrame.
        target: Target column values.
        test_size: Proportion of data for testing.
        random_state: Random seed for reproducibility.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_df, target, test_size=test_size, random_state=random_state)

    error_rate = []

    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error = np.mean(pred_i != y_test)
        error_rate.append(error)

    # Plotting error rate vs. K values
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, 40),
        error_rate,
        color='blue',
        linestyle='dashed',
        marker='o',
        markerfacecolor='red',
        markersize=10,
    )
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()

# Retrain KNN with optimized K value
def knn_retrain(X_train, X_test, y_train, y_test, n):
    """
    Retrain the KNN model with the best K value.

    Args:
        X_train, X_test, y_train, y_test: Train-test split data.
        n: Optimal K value.

    Returns:
        None
    """
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    pred_knn = knn.predict(X_test)

    print(f"WITH K={n}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred_knn))
    print("\nClassification Report:\n", classification_report(y_test, pred_knn))

def main():
    # Define file path
    filepath = input("Enter the path to the Titanic-Dataset CSV file: ").strip()
    # Load data
    df = load_data(filepath)
    if df is None:
        return None #Return None if data loading failed 

    # Preprocess data
    df = preprocess_data(df)

    # Display basic statistics
    basic_statistic(df)

    # Visualizations
    visualisations_of_df(df)

    # Linear Regression
    linear_target = 'Fare'  # Target variable for linear regression
    X = df[['Pclass', 'Sex_male', 'Age']]
    y = df[linear_target]
    predictions_lm, coefficients, y_test_lm = train_and_predict_Lm(X, y)
    visualize_lm_results(y_test_lm, predictions_lm)

    # Logistic Regression
    log_target = 'Survived'  # Target variable for logistic regression
    X_logistic = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Sex_male']]
    y_log = df[log_target]
    predictions_log, y_test_log, logmodel, X_train_log, X_test_log, y_train_log, y_test_log = train_and_predict_logm(
        X_logistic, y_log)
    visualize_logm_confusion_matrix(y_test_log, predictions_log)

    # kNN
    knn_target = 'is_Alone'
    scaled_features = data_scaler(df, knn_target)
    predict_knn, y_test_knn, knn_model, X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_and_predict_KNN(
        scaled_features, df[knn_target])
    visualize_knn_confusion_matrix(y_test_knn, predict_knn)
    # Optimize K value
    optimise_k_values(scaled_features, df[knn_target], test_size=0.3, random_state=42)

    # Optionally retrain KNN with an optimal K value (e.g., n=5)
    # knn_retrain(X_train_knn, X_test_knn, y_train_knn, y_test_knn, n=5)
    return df #Return df from main()

# Call main
if __name__ == "__main__":
    df = main()  # Assign the returned DataFrame to df

# %%
