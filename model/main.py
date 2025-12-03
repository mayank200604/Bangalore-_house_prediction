# importing libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import json

df= pd.read_csv('Bengaluru_House_Data.csv')

# Fixing size.
def convert_BHK(x):
    try:
        return int(str(x).split(' ')[0])
    except:
        return None
    
df['BHK']= df['size'].apply(convert_BHK)
df1=df.drop(['size'],axis=1)

df2= df1.dropna(subset=['location','BHK','bath'])

# fixing total_sqft.
def convert_total_sqft(x):
    try:
        if '-' in str(x): #checking if the sting contains -(100-200)
            tokens= str(x).split('-')
            if len(tokens)==2:
                return float(tokens[0].strip()) + float(tokens[1].strip())/2
            
        return float(x)
    
    except:
        return None
    
df2= df2.copy()
df2['total_squareft']= df2['total_sqft'].apply(convert_total_sqft)
df3= df2.drop(['total_sqft'],axis=1)

median_sqft= df3['total_squareft'].median()
df3['total_squareft']= df3['total_squareft'].fillna(median_sqft)
print("The average median of sqft is:",median_sqft)


# Convert columns to numeric (in case they aren't already)
df3['total_squareft'] = pd.to_numeric(df3['total_squareft'], errors='coerce')
df3['BHK'] = pd.to_numeric(df3['BHK'], errors='coerce')
df3['bath'] = pd.to_numeric(df3['bath'], errors='coerce')

# Apply both outlier removal rules
df3 = df3[
    (df3['total_squareft'] / df3['BHK'] >= 300) &
    (df3['bath'] <= df3['BHK'] + 2)]
df4=df3.copy()
df4=df4.drop(['society'],axis=1)

location_counts = df4['location'].value_counts()
rare_locations = location_counts[location_counts <= 10].index

df4['location'] = df4['location'].apply(lambda x: 'Other' if x in rare_locations else x)
df5=df4.dropna(subset=['balcony'])

df5= pd.get_dummies(df5,columns=['location'],drop_first=True)

df6= df5.drop(['area_type','availability'],axis=1)

# checking for the right column(228).
location_counts = df4['location'].value_counts()
#locations with more than 10 data points
locations_gt_10 = location_counts[location_counts > 10]
df6 = df6[df6['price'] >= 10].copy()
df6['price'] = np.log1p(df6['price'])

X = df6.drop(['price'], axis=1)
y = df6['price']

# Use pipeline without ColumnTransformer
pipe = Pipeline([
    ('model', LGBMRegressor())
])

param_grid = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.1],
    'model__max_depth': [5, 10],
    'model__num_leaves': [31, 63],
    'model__min_child_samples': [5, 10],
    'model__subsample': [0.8],
    'model__colsample_bytree': [0.8]
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

# Predict
from sklearn.metrics import r2_score
print("R² Score:", r2_score(y_test, grid.predict(X_test)))


new_data = np.zeros(X.shape[1])  
columns = X.columns.tolist()

# Manually set feature values
new_data[columns.index('total_squareft')] = 1500
new_data[columns.index('BHK')] = 2
new_data[columns.index('bath')] = 2
new_data[columns.index('location_Whitefield')] = 1 # Set location

# Convert to 2D array and predict
predicted_price = grid.predict([new_data])[0]
actual_price = np.expm1(predicted_price)
print("Predicted price:", round(actual_price, 2), "lakhs")

y_pred = grid.predict(X_test)
y_pred_actual = np.expm1(y_pred)       # reverse log1p
y_test_actual = np.expm1(y_test)       # reverse log1p

# Scatter plot
plt.scatter(y_test_actual, y_pred_actual, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.xlabel('Actual Price (Lakhs)')
plt.ylabel('Predicted Price (Lakhs)')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.grid(True)
plt.show()



with open('bangalore_home_prediction_model.pickle','wb') as f:
    pickle.dump(grid.best_estimator_,f)

original_features=X.columns.tolist()
final_features=[col.lower() for col in original_features]

with open('model_features.json', 'w') as f:
    json.dump(final_features, f)
