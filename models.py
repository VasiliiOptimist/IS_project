# %%
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import GradientBoostingRegressor as gbr

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import pickle


def get_rmse(y_test, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_test,y_pred))

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = get_rmse(true, predicted)
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('______')
    return [mae, rmse]


def print_evaluate_rmse_mae(X_test, y_test, gbr_y_pred):
    r_squared = gbr.score(X_test, y_test)
    mse = metrics.mean_squared_error(y_test, gbr_y_pred)
    mae = metrics.mean_absolute_error(y_test, gbr_y_pred)
    rmse = get_rmse(y_test, gbr_y_pred)
    print(f"R-squared value of GradientBoostingRegressor: {r_squared}")
    print(f"The mean squared error of GradientBoostingRegressor: {mse}")
    print(f"The mean absoluate error of GradientBoostingRegressor: {mae}")
    return [mae, rmse]


def show_df(
    title: str,
    columns: List[str],
    rows: List[list],
    label_x: str = "Metrics",
    label_y: str = "Value",
):
    width = 0.3
    x = np.arange(len(columns))
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ["NFT"]

    rects1 = ax.bar(x, rows[0], width, label=labels[0])
    # rects1 = ax.bar(x - width/2, rows[0], width, label=labels[0])
    # rects1 = ax.bar(x + width/2, rows[1], width, label=labels[1])
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(columns)
    ax.legend(loc='upper left')

# %% [markdown]
# ## Prepare DataFrame

# %%
df = pd.read_csv("Dataset_v1.csv")
df.head()

# %%
length = len(df.sort_values(by=['price_usd']))

print(length * 0.1)
first_10_percents = df.sort_values(by=['price_usd']).index[:int(length*0.1)]
last_10_percents = df.sort_values(by=['price_usd']).index[-int(length*0.1):]
first_10_percents
df.drop(first_10_percents, axis=0, inplace=True)
df.drop(last_10_percents, axis=0, inplace=True)
df

# %%
df['eth (usd)'] = df['price_usd'] / df['price_eth']
df.drop(['price_eth', 'transaction_date'], axis = 1, inplace=True)

# %%
df.dropna(inplace=True)

# %%
df.sort_values(by=['price_usd'])

# %%
df
# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,1:], df.iloc[:, 0],test_size=0.3,random_state=42)

# %%
X_train
# %% [markdown]
# ## Check by LinearRegression

# %%
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)


LinearRegression_test = print_evaluate(y_test, y_pred)

# %%
plt.figure(figsize = (20, 8))
plt.plot(y_test.to_numpy(dtype=object)[:100], color = "purple")
plt.plot(y_pred[:100], color = "green")
plt.title("NFT", fontsize = 14, fontweight = "bold") #updated here
plt.ylabel("Y",fontsize = 20)
plt.xlabel("X",fontsize = 20)

plt.show()

# %% [markdown]
# ## Check by GradientBoostingRegressor

# %%
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

gbr_y_pred = gbr.predict(X_test)

GradientBoostingRegressor_test = print_evaluate_rmse_mae(X_test, y_test, gbr_y_pred)


# %%
plt.figure(figsize=(25,10))
plt.plot(y_test.to_numpy(dtype=object)[:100],c='red')
plt.plot(gbr_y_pred[:100],c='black')  #predicts
plt.legend(['real','predict'],fontsize="large")
plt.title('The result of GradientBoostingRegressor')

# %% [markdown]
# ## Adding Scaler

# %%
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=500)
rfr.fit(X_train,y_train)

rfr_y_pred = rfr.predict(X_test)


test = print_evaluate_rmse_mae(X_test, y_test, rfr_y_pred)

# %%
plt.figure(figsize=(25,10))
plt.plot(y_test.to_numpy(dtype=object)[:200],c='blue')
plt.plot(rfr_y_pred[:200],c='red')
plt.legend(['real','predict'],fontsize="large")
plt.title('The result of RandomForestRegressor')

# %% [markdown]
# #### Train set

# %%
X_train
# %%
from sklearn.ensemble import RandomForestRegressor

rfr_reg = RandomForestRegressor(n_estimators=200)  # maybe parameter tuning
rfr_reg.fit(X_train, y_train)

with open('models/rfr.p', 'wb') as fp:
    pickle.dump(rfr, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
train_pred = rfr_reg.predict(X_train)


print('Testing set evaluation:\n______')
RandomForestRegressor_test = print_evaluate(y_test, rfr_y_pred)
print('==*******==')
print('Training set evaluation:\n_____')
RandomForestRegressor_train = print_evaluate(y_train, train_pred)




# %%
plt.figure(figsize=(25,10))
plt.plot(y_test.to_numpy(dtype=object)[:100],c='pink')
plt.plot(rfr_y_pred[:100],c='black')  #predicts

plt.legend(['real','predict'],fontsize="large")
plt.title('The result of RandomForestRegressor')

# %% [markdown]
# ## Check by XGBoost regressor

# %%
import xgboost as xgb

xgb_reg = xgb.XGBRegressor(
    objective ='reg:linear',
    colsample_bytree = 0.3,
    learning_rate = 0.1,
    max_depth = 5,
    alpha = 10,
    n_estimators = 10
)
xgb_reg.fit(X_train,y_train)

xgb_y_pred = xgb_reg.predict(X_test)

XGBoost_regressor = print_evaluate(y_test, xgb_y_pred)

# %%
plt.figure(figsize=(25,10))
plt.plot(y_test.to_numpy(dtype=object)[:100],c='red')
plt.plot(gbr_y_pred[:100],c='black')  #predicts
plt.legend(['real','predict'],fontsize="large")
plt.title('The result of GradientBoostingRegressor')

# %% [markdown]
# ## Comparison of results before and after optimizations

# %% [markdown]
# * best values of metrics
# * MAE -> 0.0
# * MSE -> 0.0
# * RMSE -> should be less 180
# * R2 Square -> 1.0

# %% [markdown]
# #### LinearRegression

# %%
show_df(
    "LinearRegression",
    ["MAE", "RMSE"],
    [
        LinearRegression_test
    ]
)

# %% [markdown]
# #### GradientBoostingRegressor

# %%
show_df(
    "GradientBoostingRegressor",
    ["MAE", "RMSE"],
    [
        GradientBoostingRegressor_test
    ]
)

# %% [markdown]
# #### RandomForestRegressor - Test set

# %%
show_df(
    "RandomForestRegressor - Test set",
    ["MAE", "RMSE"],
    [
        RandomForestRegressor_test
    ]
)

# %% [markdown]
# #### RandomForestRegressor - Train set

# %%
show_df(
    "RandomForestRegressor - Train set",
    ["MAE", "RMSE"],
    [
        RandomForestRegressor_train
    ]
)

# %% [markdown]
# ## XGBoost regressor

# %%
show_df(
    "XGBoost regressor",
    ["MAE", "RMSE"],
    [
        XGBoost_regressor
    ]
)

# %%
rmse = [LinearRegression_test[1],
    GradientBoostingRegressor_test[1],
    RandomForestRegressor_test[1],
    XGBoost_regressor[1]]

colors = ['#789644','#968E44', '#B0E35B', '#E34F8A']

df_rmse = pd.DataFrame([rmse],
    columns=['Linear', 'GradientBoosting',
            'RandomForest', 'XGBoost']
)

width = 0.3
x = np.arange(len(df_rmse.columns))
fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x, df_rmse.iloc[0,:], width, color=colors)
ax.set_title('Comprasion of NFT price predictions', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(df_rmse.columns)
ax.set_xlabel('Regression', fontweight='bold', fontsize=14)
ax.set_ylabel('RMSE', fontweight='bold', fontsize=14)
ax.axhline(y = min(rmse), color = 'k', linestyle = '--')

ax.annotate('Our winner!', xy=(2, 29), 
            xytext=(1.25, 20),
            arrowprops=dict(arrowstyle='fancy', color='grey'),
            fontsize = 16,
            fontname='Comic Sans MS',
            color='grey'
            )

plt.show()

# %%



