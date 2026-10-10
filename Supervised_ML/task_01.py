import pandas as pd

data = {
    "hours_studied": [2, 4, 5, 7, 8, 10],
    "attendance": [60, 65, 70, 80, 85, 95],
    "previous_score": [50, 55, 60, 70, 75, 85],
    "final_score": [45, 52, 58, 72, 78, 90]
}

df = pd.DataFrame(data)

# What are the features?
# The Features in the above dataset are the 'hours_studied', 'sttendance', 'previous_score'.

# What is the target?
# The target is to predict the final_score of a candidate based on his hours_studied, attendance, and previous_score features.

# Is this a supervised or unsupervised problem?
# This is a supervised machine learning problem. Because for every input there is a labelled out as a final_score. so it is a regressional problem too.

# Is this regression or classification?
# It is a regression problem because it involves predicting continous values rather than categorical values.

x = df[['hours_studied', 'attendance', 'previous_score']]
y = df['final_score']

# Why is final_score the target rather than a feature?
# The final_score is the target because we have to predic the final_score values for each row. 

# What should the model produce?
# the final_score value as around 63 or something


