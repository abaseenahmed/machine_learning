import numpy as np
import pandas as pd
import random

# ---------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------
np.random.seed(42)
random.seed(42)

N_SAMPLES = 1000

# ---------------------------------------------------------
# 2. Numerical features
# ---------------------------------------------------------
age           = np.random.randint(18, 70, N_SAMPLES)
income        = np.random.normal(55000, 20000, N_SAMPLES).clip(15000, 200000).round(2)
experience    = np.random.randint(0, 40, N_SAMPLES)
education_yrs = np.random.randint(10, 22, N_SAMPLES)
hours_worked  = np.random.normal(40, 8, N_SAMPLES).clip(10, 80).round(1)
satisfaction  = np.random.uniform(1, 10, N_SAMPLES).round(1)

# ---------------------------------------------------------
# 3. Categorical features
# ---------------------------------------------------------
gender        = np.random.choice(['Male', 'Female', 'Other'], N_SAMPLES, p=[0.48, 0.48, 0.04])
department    = np.random.choice(['Engineering', 'Sales', 'HR', 'Marketing', 'Finance'], N_SAMPLES)
city          = np.random.choice(['Karachi', 'Lahore', 'Islamabad', 'Peshawar', 'Quetta'], N_SAMPLES)
education_lvl = np.random.choice(['Bachelor', 'Master', 'PhD', 'Diploma'],
                                 N_SAMPLES, p=[0.5, 0.3, 0.1, 0.1])
remote_work   = np.random.choice(['Yes', 'No'], N_SAMPLES, p=[0.4, 0.6])

# ---------------------------------------------------------
# 4. Targets (with realistic relationships + noise)
# ---------------------------------------------------------
base_salary = (
    income * 0.9
    + experience * 800
    + education_yrs * 500
    + satisfaction * 300
    + np.where(gender == 'Female', -1500, 0)
    + np.where(remote_work == 'Yes', 1200, 0)
    + np.random.normal(0, 5000, N_SAMPLES)
)

salary      = base_salary.round(2)
high_earner = np.where(base_salary > np.median(base_salary), 'Yes', 'No')

# ---------------------------------------------------------
# 5. Introduce missing values
# ---------------------------------------------------------
income_series       = pd.Series(income)
satisfaction_series = pd.Series(satisfaction)
city_series         = pd.Series(city)

for s in [income_series, satisfaction_series, city_series]:
    idx = np.random.choice(s.index, size=int(0.03 * N_SAMPLES), replace=False)
    s.loc[idx] = np.nan

# ---------------------------------------------------------
# 6. Fake names without faker (simple generator)
# ---------------------------------------------------------
first_names = ['Ali', 'Sara', 'Ahmed', 'Ayesha', 'Bilal', 'Hina', 'Usman',
               'Fatima', 'Zain', 'Mariam', 'Omar', 'Noor', 'Hamza', 'Sana',
               'John', 'Emma', 'Liam', 'Olivia', 'Noah', 'Ava']
last_names  = ['Khan', 'Ahmed', 'Malik', 'Sheikh', 'Iqbal', 'Smith',
               'Johnson', 'Brown', 'Raza', 'Hussain']

names = [f"{random.choice(first_names)} {random.choice(last_names)}"
         for _ in range(N_SAMPLES)]

# ---------------------------------------------------------
# 7. Build DataFrame
# ---------------------------------------------------------
df = pd.DataFrame({
    'employee_id'      : [f"EMP{1000+i}" for i in range(N_SAMPLES)],
    'name'             : names,
    'age'              : age,
    'gender'           : gender,
    'city'             : city_series,
    'department'       : department,
    'education_level'  : education_lvl,
    'education_years'  : education_yrs,
    'experience_years' : experience,
    'hours_worked'     : hours_worked,
    'remote_work'      : remote_work,
    'income'           : income_series,
    'satisfaction'     : satisfaction_series,
    'salary'           : salary,
    'high_earner'      : high_earner
})

# ---------------------------------------------------------
# 8. Save
# ---------------------------------------------------------
output_file = 'synthetic_ml_dataset.csv'
df.to_csv(output_file, index=False)

print(f"✅ Dataset created: {output_file}")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nTarget distribution (high_earner):")
print(df['high_earner'].value_counts())