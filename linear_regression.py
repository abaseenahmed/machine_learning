import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = {
    "age": [22, 25, 28, 30, 35, 40, 42, 45, 50, 55,
            60, 23, 27, 32, 38, 41, 48, 52, 57, 62],

    "experience": [1, 2, 4, 5, 8, 12, 15, 18, 22, 25,
                   30, 1, 3, 6, 10, 14, 19, 23, 27, 32],

    "education_years": [16, 16, 16, 18, 18, 18, 16, 18, 16, 14,
                        14, 16, 16, 18, 18, 16, 18, 16, 14, 14],

    "salary": [35000, 42000, 50000, 60000, 75000, 95000,
               110000, 130000, 155000, 170000,
               190000, 38000, 47000, 65000, 85000, 105000,
               125000, 145000, 165000, 200000]
}

df = pd.DataFrame(data)

X = df[[
    "age",
    "experience",
    "education_years"
]]

y = df["salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae}')

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

model_df = pd.DataFrame()
model_df['Features'] = X.columns
model_df['Coefficients'] = model.coef_
print(model_df)

# The Experience feature has the largest coefficient
# The age feature has the smallest coefficient of -831
# The positive coefficient mean the model is strongly dependent on that feature
# The negative coefficient mean the model is weekly dependent on that feature
# Yes the feature with the largest coefficient is automatically the most important feature. Because the model strongly depends on that feature

new_person = pd.DataFrame({
    "age": [29],
    "experience": [5],
    "education_years": [16]
})
print(new_person)
new_prediction = model.predict(new_person)
print("Predicted Salary:", new_prediction[0])

# model = LinearRegression(): This line creates a model based on linear regression algorithms
# model.fit(X_train, y_train): This line gives the model to train itself using the features and it's ouputs
# y_pred = model.predict(X_test): This line predicst the oupts for test data. to check how good it predicts
# mae = mean_absolute_error(y_test, y_pred): This line calculates the mean absolute erro, calculating the average, by substracting actual oupt from predicted output in the test data

# y^​ = b0​ + b1​x
# The training data determines the coefficients.When We want to find a line that gets as close as possible to these points. The line is controlled by the coefficients.

# Why can't we simply calculate the prediction and stop? Why do we need a loss/error measure such as MSE?
# we cann't simply predict and stop. whe have to compare our results and find the best method of prediction. It tries to find the values of:b, b1, b2, etc. that produce the best fit according to the objective.

# What is a residual?
# Residual is the difference between the actual value and predicted value. residuals tell us how far each prediction is from reality and in which direction.

# Why does Linear Regression try to minimize squared errors?
# Linear Regression traditionally finds coefficients by minimizing the sum of squared residuals: This method is called least squares.

# Data → coefficients → predictions → residuals → MSE → optimization → better coefficients
# For any ML data pipeline we follow a pattern in which we take the raw data, find it's coefficients, predict the outcome, fint and ealuate the residuals, and use the other techniques such as MSE, MAE etc.
