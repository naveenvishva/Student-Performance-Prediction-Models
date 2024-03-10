### Student Performance Prediction Models

This project consists of two Python scripts for training and evaluating machine learning models to predict student academic performance based on various features. The models included are Linear Regression and Decision Tree Regression. Each model is trained using student performance data from the provided CSV file.

### Files Included:

1. `linear_regression_model.py`: Python script for training and evaluating a Linear Regression model to predict student academic performance.
2. `decision_tree_model.py`: Python script for training and evaluating a Decision Tree Regression model to predict student academic performance.
3. `student-mat.csv`: CSV file containing student performance data used for training the models.

### Linear Regression Model:

- The Linear Regression model is trained using features such as age, study time, failures, free time, going out, health, and absences.
- The model's performance is evaluated using Mean Squared Error (MSE) and R-squared (R2) score.
- A sample prediction for new data is demonstrated.
- A scatter plot illustrating the relationship between true values and predictions is displayed.

### Decision Tree Regression Model:

- The Decision Tree Regression model is trained using the same features as the Linear Regression model.
- The model's performance is evaluated using Mean Squared Error (MSE) and R-squared (R2) score.
- A scatter plot illustrating the relationship between true values and predictions is displayed.

### Usage:

1. **Running the Models**:
   - Execute `linear_regression_model.py` to train and evaluate the Linear Regression model.
   - Execute `decision_tree_model.py` to train and evaluate the Decision Tree Regression model.

### Note:
- Ensure that the provided `student-mat.csv` file is in the same directory as the Python scripts.
- The CSV file contains student performance data necessary for training the models.

### References:
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
