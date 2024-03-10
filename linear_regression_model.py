import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_linear_regression_model(csv_file_path):
    # Load the student performance dataset from the specified CSV file with ';' as the delimiter
    data = pd.read_csv(csv_file_path, delimiter=';')

    # Select the columns you want to use as features for prediction
    selected_features = ['age', 'studytime', 'failures', 'freetime', 'goout', 'health', 'absences']

    # Define the target variable (academic performance) column
    target_variable = 'G3'

    # Extract the selected features and target variable
    X = data[selected_features]
    y = data[target_variable]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R-squared (R2) Score:", r2)

    # Predict academic performance for new data
    # For example, you can create a new data point (X_new) and use the trained model to predict the academic performance
    X_new = pd.DataFrame({'age': [16], 'studytime': [3], 'failures': [0], 'freetime': [3], 'goout': [2], 'health': [4], 'absences': [5]})
    predicted_performance = model.predict(X_new)
    print("Predicted Academic Performance:", predicted_performance[0])

    # Plotting results
    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Linear Regression Model Performance")
    plt.show()

if __name__ == "__main__":
    csv_file_path = 'student-mat.csv'
    train_linear_regression_model(csv_file_path)
