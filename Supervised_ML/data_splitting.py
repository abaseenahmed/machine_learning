import pandas as pd
from sklearn.model_selection import train_test_split

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

x = df[['age', 'experience', 'education_years']] 
y = df['salary']

x_train, x_temp, y_train, y_temp = train_test_split(
    x,
    y,
    test_size=0.30,
    random_state=42
)
print(x_train.shape)
print(y_train.shape)
print(x_temp.shape)
print(y_temp.shape)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)

print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)

# The random_state=42 keeps the test_size remains same everytime. if we don't use it the test size will be changing randomely
# using the current salary for predicting the future salary is not necessary to be data leakage. Data leakage occures when we share the necessary information in the testin phase

# The test_size=0.2 keeps the 20 percent of the data for testing purpose.
# The random_state=42 keeps the selection consisitent. we don't know which rows in 80% of the dataset are use to train so if we use random_state=42 those rows will be consitent and not chaning randomely
# The X_train has 800 records for trainin and X_test has 200 records for testing
# we need to repeatedly evaluate our model during the test and hanging model and hyperparameters to find the best model and accurate solution.
# Is using 2026_salary necessarily data leakage? No
