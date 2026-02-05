import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import os

train_hh_features = pd.read_csv('/train_hh_features.csv')
train_hh_gt =  pd.read_csv('/train_hh_gt.csv')

train_hh_features.head()

"""Because the household consumption feature is in different tables we need to perform a merge the tables on that feature so that we could train a tabular model. We merge on survey_id"""

X_train_full = pd.merge(train_hh_features, train_hh_gt, on=['survey_id', 'hhid'],
    how='inner' )

missing_counts_features = train_hh_features.isnull().sum()
missing_counts_gt = train_hh_gt.isnull().sum()
print(missing_counts_features)

print(missing_counts_gt)

"""We can see that only 5 columns have missing values. So we perform a simple impuatation of median for the numeric columns and most frequent strategy for categorical columns. First we do a train test split"""

from sklearn.model_selection import train_test_split

# Preparing the data
X_train_full

y = X_train_full.cons_ppp17

X = X_train_full.drop(['cons_ppp17','hhid'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

print(y_train.isnull().sum())

from sklearn.impute import SimpleImputer

# Numeric data
numeric_imputer = SimpleImputer(strategy="median")

num_cols = X_train.select_dtypes(include=['number']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

X_train_imputed = X_train.copy()
X_valid_imputed = X_valid.copy()

X_train_imputed[num_cols] = numeric_imputer.fit_transform(X_train[num_cols])
X_valid_imputed[num_cols] = numeric_imputer.transform(X_valid[num_cols])

# Categorical data
categorical_imputer = SimpleImputer(strategy="most_frequent")

X_train_imputed[categorical_cols] = categorical_imputer.fit_transform(X_train[categorical_cols])
X_valid_imputed[categorical_cols] = categorical_imputer.transform(X_valid[categorical_cols])

"""Next we encode the text data so the model can interpret them."""

from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data
label_X_train = X_train_imputed.copy()
label_X_valid = X_valid_imputed.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
label_X_train[categorical_cols] = ordinal_encoder.fit_transform(X_train_imputed[categorical_cols])
label_X_valid[categorical_cols] = ordinal_encoder.transform(X_valid_imputed[categorical_cols])

"""Now we train an XGBoost model."""

my_model = XGBRegressor(n_estimators=500, learning_rate=0.05)
my_model.fit(label_X_train, y_train)

"""We evaluate using MAE as a loss function"""

from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(label_X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

"""A good start, we will check if there are outliers that might be causing the MAE to give a high-ish score of 3.27"""

# Creating a comparison table

results = pd.DataFrame({"Actual": y_valid , "Predicted": predictions}, index=y_valid.index)
results['Error'] = abs(results['Actual'] - results['Predicted'])

import matplotlib.pyplot as plt

# Look at the top 10 biggest misses
print("Biggest Errors:")
print(results.sort_values(by='Error', ascending=False).head(10))

# Visualizing the 'clump'
plt.figure(figsize=(8, 6))
plt.scatter(results['Actual'], results['Predicted'], alpha=0.3, color='blue')
plt.plot([0, results['Actual'].max()], [0, results['Actual'].max()], color='red', linestyle='--')
plt.xlabel('Actual Consumption ($)')
plt.ylabel('Predicted Consumption ($)')
plt.title('Actual vs. Predicted Consumption')
plt.show()

"""It seems that big variance between consumption values is causing the model some trouble. We will use the log trick to minimise the variance between the data points. So the model treats rich households as slightly bigger data points than poor house."""

#log scale y_train
y_train_log = np.log1p(y_train)

my_model.fit(label_X_train, y_train_log)
prediction_log = my_model.predict(label_X_valid)

actual_prediction = np.expm1(prediction_log)
print("Mean Absolute Error: " + str(mean_absolute_error(actual_prediction, y_valid)))

"""Now that the base model is designed, we shall begin with the test data given from the comepition."""

X_test = pd.read_csv('/test_hh_features.csv')

X_test.head()

X_test_imputed = X_test.copy()

# Data imputation
X_test_imputed[num_cols] = numeric_imputer.transform(X_test[num_cols])
X_test_imputed[categorical_cols] = categorical_imputer.transform(X_test[categorical_cols])

# Data encoding
label_X_test = X_test_imputed.copy()
label_X_test[categorical_cols] = ordinal_encoder.transform(X_test_imputed[categorical_cols])

# model fitting
label_X_test_final = label_X_test.drop(columns=['hhid']) # Forget to drop it. It was not in the training data.

test_log_pred = my_model.predict(label_X_test_final)

test_actual_pred = np.expm1(test_log_pred) # Reverse the log scale

"""Now we prepare to submit the model."""

test_results = pd.DataFrame({"survey_id" : X_test["survey_id"], "hhid" : X_test["hhid"], "cons_ppp17" : test_actual_pred })

print(test_results.head())

"""Building the submission for distribution of the consumption per household value"""

threshold_list = [3.17, 3.94, 4.60, 5.26, 5.88, 6.47, 7.06, 7.70, 8.40, 9.13,
            9.87, 10.70, 11.62, 12.69, 14.03, 15.64, 17.76, 20.99, 27.37]

submission_rows = []

# Get all consumption values for each survey (without going through all appereances of the survey)
for survey in test_results['survey_id'].unique():
  survey_predictions = test_results[test_results['survey_id'] == survey]['cons_ppp17']
  total_predictions = len(survey_predictions)

  row = {'survey_id': survey}

  # Threshold comparison
  for i, threshold in enumerate(threshold_list):
    count = 0

    for prediction in survey_predictions:
      if prediction < threshold:
        count += 1

    # calculate percenatge
    percentage = count / total_predictions

    column_title = f'pct_hh_below_{threshold}'
    row[column_title] = percentage

  submission_rows.append(row)

final_submission = pd.DataFrame(submission_rows)

final_submission.head()

"""Great now we will save the dataframes into csv and the submission should be ready."""

final_submission.to_csv("submission_distribution.csv", index=False)

#test_results.to_csv("household_predictions.csv",index=False)