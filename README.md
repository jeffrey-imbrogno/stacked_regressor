# stacked_regressor
Create a dataframe with the output of multiple python regression algorithms

# motivation for the project 

There are a number of voting and stacking models for classification projects in Python, but the existence of stacked regression pipelines is limited to non-existent

# code example
import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR, LinearSVR

from sklearn import linear_model

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

data = [[1,1,0,0,0,25], [0,0,0,1,0,55], [0,0,1,0,0,37], [1,0,0,1,0,57]]

df = pd.DataFrame(data, col=['Workday', 'Spring', 'Summer', 'Fall', 'Winter', 'Count'])

output_df = stackedregressor(df, 'Count', 2)

output_df.head()

