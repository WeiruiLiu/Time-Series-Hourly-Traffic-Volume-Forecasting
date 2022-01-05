# ============================== Import Libraries ==============================================================================================================================================================================
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
import statsmodels.api as sm
from scipy.stats import chi2
from numpy import linalg as LA
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
import statsmodels.tsa.holtwinters as ets
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL, seasonal_decompose
import warnings
from datetime import datetime
import sys
sys.path.append('../toolbox')
from toolbox import autocorr, autocorr_values,ADF_Cal,Cal_GPAC, Cal_GPAC_DataFrame, ACF_PACF_Plot,Rolling_Mean_Var_Plot, difference, Q_value, LMA, A_Cal

# Ignore warnings
warnings.filterwarnings('ignore')

# Calculate the running time
start_time = datetime.now()        # It will take about 12 minus to run the whole file (0:11:30.590180)


# ============================== Load Dataset ===============================================================================================================================================================================
data = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')


# ============================== Dataset Preprocessing ======================================================================================================================================================================
# Drop the duplicates rows
data = data.drop_duplicates(subset=['date_time'], ignore_index=True)  # date_time from from 2012-10-02 09:00:00 to 2018-09-30 23:00:00

# Handle the missing data
series = pd.Series(data['traffic_volume'].values, index=pd.to_datetime(data['date_time']))
full_time = pd.date_range('2012-10-02 09:00:00', '2018-09-30 23:00:00', freq='H')
series.index = pd.DatetimeIndex(series.index)
full_series = series.reindex(full_time, fill_value=None)
df_traffic = full_series.to_frame().reset_index()
df_traffic.rename(columns={'index':'date_time', 0:'traffic_volume'}, inplace=True)
df_traffic['Hour'] = df_traffic['date_time'].dt.strftime('%H')

# Calculate the mean traffic volume by hour
df_traffic_volume_hourly_mean = df_traffic.groupby(by='Hour')['traffic_volume'].mean().reset_index(name='Mean of Traffic Volume')

# Replace the missing data with the mean traffic volume by hour
for i in range(len(df_traffic_volume_hourly_mean['Hour'])):
    df_traffic.loc[(df_traffic['traffic_volume'].isnull() == True) & (df_traffic['Hour'] == df_traffic_volume_hourly_mean['Hour'][i]),'traffic_volume'] = int(df_traffic_volume_hourly_mean['Mean of Traffic Volume'][i])

# Dependent Variable and Time column
traffic = df_traffic['traffic_volume']
time = df_traffic['date_time']
traffic_series = pd.Series(np.array(df_traffic['traffic_volume']), index=pd.date_range('2012-10-02 09:00:00',freq='H',periods=len(df_traffic['traffic_volume'])),name='Traffic Volume')

# Place none holiday and holiday in the 'holiday' column to 0 and 1.
data['holiday'] = data['holiday'].apply(lambda x: 0 if x == 'None' else 1)

# Convert 'weather_main' column into indicator variables
data = pd.get_dummies(data, columns=['weather_main'])


# ============================== Description of the dataset ================================================================================================================================================================
print('# ======================= Description of Dataset ================================================================================================= #')
print('Number of row:', data.shape[0])
print('Number of column:', data.shape[1])
print('Independent Variables:\n', data.columns[:7].tolist()+data.columns[8:].tolist())
print('Dependent Variable:', data.columns[7])
print()

# a. Plot the dependent variable versus time (first 200 samples) -- Since this is a large hourly change of the dataset, showing all the instances will obscure some information
plt.figure(figsize=(14,8))
plt.plot(time[:200], traffic[:200], label='Traffic Volume')     # show only first 200 instances
plt.legend(fontsize=12)
plt.xlabel('Time (Hour)', fontsize=16)
plt.ylabel('Traffic Volume', fontsize=16)
plt.title('The Traffic Volume versus Time (First 200 samples)', fontsize=20)
plt.show()

# Plot of the dependent variable versus time (whole dataset)
# plt.figure(figsize=(14,8))
# plt.plot(traffic_series, label='Traffic Volume')
# plt.legend(fontsize=12)
# plt.xlabel('Time (Hour)', fontsize=16)
# plt.ylabel('Traffic Volume', fontsize=16)
# plt.title('The Traffic Volume versus Time (Whole dataset)', fontsize=20)
# plt.show()

# b. ACF/PACF of the dependent variable.
lags = 100
ACF_PACF_Plot(traffic,lags,'Traffic Volume')

# c. Correlation Matrix with seaborn heatmap and Pearson's correlation coefficient
corr = data.corr()
plt.figure(figsize=(18,16))
ax = sns.heatmap(corr,vmin=-1, vmax=1, center=0, cmap='YlOrRd_r',square=True,annot=True)
bottom,top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, horizontalalignment='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=360)
ax.set_title('Correlation Matrix of Traffic Volume Dataset', fontsize=20)
plt.show()

# d. Check if the dataset have missing data or NAN.
data.info()
print()

# e. Split the dependent variable into train set (80%) and test set (20%)
train, test = train_test_split(traffic_series, test_size=0.2, shuffle=False)
print('The number of train set', train.shape[0])
print('The number of test set', test.shape[0])
print()


# ============================== Stationarity ===========================================================================================================================================================================
print('# ======================= Stationarity ================================================================================================== #')
# ADF Test of raw dataset
ADF_Cal(traffic,'Traffic Volume')
print()

# Plot rolling mean and rolling variance of raw dataset
Rolling_Mean_Var_Plot(traffic,'Traffic Volume')

# Apply the seasonal differencing
traffic_diff = difference(traffic, interval=168)       # seasonal differencing (24 hours x 7 days)
# Plot ACF/PACF of seasonal differenced dataset
ACF_PACF_Plot(traffic_diff,lags*2,'Seasonal Differenced of Traffic Volume')
# Apply the non-seasonal differencing
traffic_diff = difference(traffic_diff, interval=1)   # 1st order non-seasonal differencing
# Plot ACF/PACF of non-seasonal differenced dataset
ACF_PACF_Plot(traffic_diff,lags*2,'Seasonal and Non-Seasonal Differenced of Traffic Volume')

# ADF Test of differenced dataset
ADF_Cal(traffic_diff,'Differenced Traffic Volume')
print()

# Plot rolling mean and rolling variance of differenced dataset
# Rolling_Mean_Var_Plot(traffic_diff,'Differenced Traffic Volume')


# ============================== Time Series Decomposition ===========================================================================================================================================================================
print('# ======================= Time Series Decomposition ====================================================================================== #')
# Apply the STL decomposition method
STL = STL(traffic_series, period=168)
res = STL.fit()
T = res.trend
S = res.seasonal
R = res.resid

# Plot the trend, seasonality, and reminder
plt.figure(figsize=(20,10))
plt.plot(T[:500], label='Trend')
plt.plot(S[:500], label='Seasonal')
plt.plot(R[:500], label='Residuals')
plt.legend(fontsize=14)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Traffic Volume', fontsize=16)
plt.xticks(rotation=25)
plt.title('The Trend, Seasonality, and Reminder of Traffic Volume (First 500 samples)', fontsize=20)
plt.show()

# De-trended dataset
detrended_traffic = traffic_series - T

# Seasonally adjusted dataset
adj_seasonal_traffic = traffic_series - S

# Plot the raw dataset versus the de-trended dataset and seasonal adjusted dataset
plt.figure(figsize=(20,10))
plt.plot(traffic_series[:500], label='Raw dataset')
plt.plot(detrended_traffic[:500], label='De-trended dataset')
plt.plot(adj_seasonal_traffic[:500], label='Seasonal Adjusted dataset')
plt.legend(fontsize=14)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Traffic Volume', fontsize=16)
plt.xticks(rotation=25)
plt.title('The Raw dataset versus the De-trended dataset and Seasonal Adjusted dataset (First 500 samples)', fontsize=20)
plt.show()

# Calculate the strength of trend
Ft = np.max([0,1-np.var(R)/np.var(T*R)])     # T is trend, R is residuals
print('The strength of trend for the dataset is:',Ft)

# Calculate the strength of seasonality
Fs = np.max([0,1-np.var(R)/np.var(S*R)])     # R is residuals, S is seasonal
print('The strength of seasonality for the dataset is:',Fs)
print()


# ============================== Holt-Winters Method ===========================================================================================================================================================================
print('# ======================= Holt-Winters Method =============================================================================================== #')
# Holt-winter Method
holtt = ets.ExponentialSmoothing(train, trend='additive', seasonal_periods=168, damped_trend=True, seasonal='additive').fit()
holt_pred = holtt.forecast(steps=len(test))
holt_pred = pd.DataFrame(holt_pred).set_index(test.index)

# Calculate the forecast error, MSE, Q value, Mean, Variance
forecast_error_holt_winter = test.values-np.ndarray.flatten(holt_pred.values)
MSE_holt_winter = np.round(np.square(forecast_error_holt_winter).mean(),3)
Q_holt_winter = np.round(Q_value(forecast_error_holt_winter,lags),3)
Mean_holt_winter = np.round(np.mean(forecast_error_holt_winter),3)
Var_holt_winter = np.round(np.var(forecast_error_holt_winter),3)
print('The mean of the forecast error of Holt-Winter Method is', Mean_holt_winter)
print('The variance of the forecast error of Holt-Winter Method is', Var_holt_winter)
print('The Q value of the forecast error of Holt-Winter Method is', Q_holt_winter)
print('MSE for Holt-Winter Method Forecast is', MSE_holt_winter)
print()

# Plot the ACF of forecast error
autocorr(forecast_error_holt_winter,50,'forecast error of Holt-Winter Method')

# Plot the train set, test set and forecast with Holt-winter Method
plt.figure(figsize=(20,10))
plt.plot(train[-250:], label='Train')
plt.plot(test[:250], label='Test')
plt.plot(holt_pred[:250], label='Holt-Winter Method Forecast')
plt.legend(fontsize=14)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Traffic Volume', fontsize=16)
plt.title('The Holt-Winter Season Method Forecast with MSE={}\n(Last 250 samples of train set and first 250 samples of test set)'.format(np.round(MSE_holt_winter,3)), fontsize=20)
plt.show()


# ============================== Feature Selection ===========================================================================================================================================================================
print('# ======================= Feature Selection ========================================================================================================================================================================== #')
# Set dependent variable (all numeric columns expect target) and independent variables (target)
X = data[['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all',
          'weather_main_Clear', 'weather_main_Clouds', 'weather_main_Drizzle',
          'weather_main_Fog', 'weather_main_Haze', 'weather_main_Mist',
          'weather_main_Rain', 'weather_main_Smoke', 'weather_main_Snow',
          'weather_main_Squall', 'weather_main_Thunderstorm']]
Y = data[['traffic_volume']]

# Add a column of ones to an array ('const' column)
X = sm.add_constant(X)

# Split the whole dataset into train set (80%) and test set (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.2)

# Co-linearity Detection of the raw dataset
# Singular Values Analysis
X_v = X.values
H = np.matmul(X_v.T,X_v)
s,d,v = np.linalg.svd(H)
print('The Singular Values of the raw dataset is\n',d)   # One of singular value close to zero means that one or more feature are correlated (co-llinearity exist).

# Condition Number
print('The condition number of the raw dataset is', LA.cond(X))    # The condition number is higher than 1000, which means severe degree of co-linearity exist.
print()

# Estimate the regression model unknown coefficients by using OLS function
model = sm.OLS(Y_train, X_train).fit()
print(model.summary())
print()

# Feature Selection (Using a backward stepwise regression reduce the feature space dimension)
# Step 1: Remove 'snow_1h' feature
# 'snow_1h' feature has highest p-value in t-test, which means we fail to reject the null hypothesis. This coefficient is not significance.
print("# ===== Remove 'snow_1h' feature =====")
X_train.drop(['snow_1h'], axis=1, inplace=True)
model_1 = sm.OLS(Y_train, X_train).fit()
print(model_1.summary())
print()
# After 'snow_1h' features has been removed, the AIC, BIC, and Adj. R-squared do not change.

# Step 2: Remove 'weather_main_Drizzle' feature
# 'weather_main_Drizzle' feature has highest p-value in t-test, which means we fail to reject the null hypothesis. This coefficient is not significance.
print("# ===== Remove 'weather_main_Drizzle' feature =====")
X_train.drop(['weather_main_Drizzle'], axis=1, inplace=True)
model_2 = sm.OLS(Y_train, X_train).fit()
print(model_2.summary())
print()
# After 'weather_main_Drizzle' features has been removed, the AIC, BIC, and Adj. R-squared do not change. A huge decrease on the the condition number

# Step 3: Remove 'rain_1h' feature
# 'rain_1h' feature has highest p-value in t-test, which means we fail to reject the null hypothesis. This coefficient is not significance.
print("# ===== Remove 'rain_1h' feature =====")
X_train.drop(['rain_1h'], axis=1, inplace=True)
model_3 = sm.OLS(Y_train, X_train).fit()
print(model_3.summary())
print()
# After 'rain_1h' features has been removed, the AIC, BIC, and Adj. R-squared do not change.

# Step 4: Remove 'weather_main_Mist ' feature
# 'weather_main_Mist' feature has highest p-value in t-test, which means we fail to reject the null hypothesis. This coefficient is not significance.
print("# ===== Remove 'weather_main_Mist' feature =====")
X_train.drop(['weather_main_Mist'], axis=1, inplace=True)
model_4 = sm.OLS(Y_train,X_train).fit()
print(model_4.summary())
print()
# After 'weather_main_Mist' features has been removed, the AIC, BIC, and Adj. R-squared do not change. The condition number decrease 100.

# Step 5: Remove 'weather_main_Fog' feature
# 'weather_main_Fog' feature has highest p-value in t-test, which means we fail to reject the null hypothesis. This coefficient is not significance.
print("# ===== Remove 'weather_main_Fog' feature =====")
X_train.drop(['weather_main_Fog'], axis=1, inplace=True)
model_5 = sm.OLS(Y_train,X_train).fit()
print(model_5.summary())
print()
# After 'weather_main_Fog' features has been removed, the AIC, BIC, and Adj. R-squared do not change.

# Step 6: Remove 'weather_main_Squall' feature
# 'weather_main_Squall' feature has highest p-value in t-test, which means we fail to reject the null hypothesis. This coefficient is not significance.
print("# ===== Remove 'weather_main_Squall' feature =====")
X_train.drop(['weather_main_Squall'], axis=1, inplace=True)
model_6 = sm.OLS(Y_train,X_train).fit()
print(model_6.summary())
print()
# After 'weather_main_Squall' features has been removed, the AIC, BIC, and Adj. R-squared do not change. The condition number drop from 5.154+04 to 1.43e+04

# Step 7: Remove 'weather_main_Smoke' feature
# 'weather_main_Smoke' feature has highest p-value in t-test, which means we fail to reject the null hypothesis. This coefficient is not significance.
print("# ===== Remove 'weather_main_Smoke' feature =====")
X_train.drop(['weather_main_Smoke'], axis=1, inplace=True)
model_7 = sm.OLS(Y_train,X_train).fit()
print(model_7.summary())
print()
# After 'weather_main_Smoke' features has been removed, the AIC, BIC, and Adj. R-squared do not change. The condition number drop drom 1.43e+04 to 7.85e+03

# Step 8: Remove 'const' feature
# 'const' is not a feature
print("# ===== Remove 'const' feature =====")
X_train.drop(['const'], axis=1, inplace=True)
model_8 = sm.OLS(Y_train,X_train).fit()
print(model_8.summary())
print()
# After 'const' features has been removed, the AIC and BIC slightly increase. Adjusted R-squared go up sharply and increase from 0.040 to 0.741

# Step 9: Remove 'weather_main_Thunderstorm' feature
# 'weather_main_Thunderstorm' has the highest standard error
print("# ===== Remove 'weather_main_Thunderstorm' feature =====")
X_train.drop(['weather_main_Thunderstorm'], axis=1, inplace=True)
model_9 = sm.OLS(Y_train, X_train).fit()
print(model_9.summary())
print()
# After 'weather_main_Thunderstorm' features has been removed, the AIC, BIC and Adj. R-squared do not change.

# Step 10: Remove 'weather_main_Snow' feature
# 'weather_main_Snow' feature has highest p-value in t-test, which means we fail to reject the null hypothesis. This coefficient is not significance. Also, 'weather_main_Snow' festure has the highest standard error.
print("# ===== Remove 'weather_main_Snow' feature =====")
X_train.drop(['weather_main_Snow'], axis=1, inplace=True)
model_10 = sm.OLS(Y_train, X_train).fit()
print(model_10.summary())
print()
# After 'weather_main_Snow' features has been removed, the BIC, AIC, and Adj. R-squared do not change.

# Step 11: Remove 'clouds_all' feature
# 'clouds_all' feature has highest p-value in t-test, which means we fail to reject the null hypothesis. This coefficient is not significance.
print("# ===== Remove 'clouds_all' feature =====")
X_train.drop(['clouds_all'], axis=1, inplace=True)
model_11 = sm.OLS(Y_train,X_train).fit()
print(model_11.summary())
print()
# After 'clouds_all' features has been removed, the BIC, AIC, and Adj. R-squared do not change. The condition number decrease from 7.85e+03 to 7.75e+03

# Step 12: Remove 'weather_main_Clear' feature
# 'weather_main_Clear' feature has highest p-value
print("# ===== Remove 'weather_main_Clear' feature =====")
X_train.drop(['weather_main_Clear'], axis=1, inplace=True)
model_12 = sm.OLS(Y_train, X_train).fit()
print(model_12.summary())
print()
# After 'weather_main_Clear' features has been removed, the BIC, AIC, and Adj. R-squared do not change.

# Step 13: Remove 'holiday' feature
# 'holiday' feature has highest standard error
print("# ===== Remove 'holiday' feature =====")
X_train.drop(['holiday'], axis=1, inplace=True)
model_13 = sm.OLS(Y_train, X_train).fit()
print(model_13.summary())
print()
# After 'holiday' features has been removed, the Adj. R-squared decrease 0.001. The condition number decrease rapidly from 7.75e+03 to 2.11e+03.

# Step 14: Remove 'weather_main_Haze' feature
# 'weather_main_Haze' feature has highest standard error
print("# ===== Remove 'weather_main_Haze' feature =====")
X_train.drop(['weather_main_Haze'], axis=1, inplace=True)
model_14 = sm.OLS(Y_train, X_train).fit()
print(model_14.summary())
print()
# After 'weather_main_Haze' features has been removed, the condition number decrease rapidly from 2.11+03 to 950

# Final model after feature reduction
model_MLR = model_14
print("# ===== Final Model after Feature Selection =====")
print(model_MLR.summary())
print()

# Co-linearity Detection of the feature reduction dataset
# Singular Values Analysis
X_train_v = X_train.values                        # Show only values
H_train = np.matmul(X_train_v.T,X_train_v)
s,d_train,v = np.linalg.svd(H_train)
print('The Singular Values of the feature reduction dataset is\n',d_train)   # One of singular value close to zero means that one or more feature are correlated (co-llinearity exist).

# Condition Number
print('The condition number of the feature reduction dataset is', LA.cond(X_train))    # The condition number is higher than 1000, which means severe degree of co-linearity exist.
print()


# ============================== Multiple Linear Regression ===========================================================================================================================================================================
print('# ======================= Multiple Linear Regression ========================================================================================= #')

# a) Perform one-step ahead forecast and compare the performance versus the test set
# Perform one-step ahead forecast on X_test
X_test = X_test[['temp', 'weather_main_Clouds', 'weather_main_Rain']]
Y_test_pred = np.zeros(len(X_test))
for t in range(len(X_test)):
    if t == 0:
        Y_test_pred[t] = 10.9367*X_train['temp'].values[-1] + 486.3230*X_train['weather_main_Clouds'].values[-1] + 203.2478*X_train['weather_main_Rain'].values[-1]
    else:
        Y_test_pred[t] = 10.9367*X_test['temp'].values[t-1] + 486.3230*X_test['weather_main_Clouds'].values[t-1] + 203.2478*X_train['weather_main_Rain'].values[t-1]
Y_test_pred = pd.DataFrame(Y_test_pred).set_index(Y_test.index)

# Calculate the forecast error, MSE, and Q value
forecast_error_MLR = Y_test.values.flatten()-Y_test_pred.values.flatten()
MSE_MLR = np.round(np.square(forecast_error_MLR).mean(),3)
Q_forecast_error_MLR = np.round(Q_value(forecast_error_MLR,lags),3)
Mean_forecast_error_MLR = np.round(np.mean(forecast_error_MLR),3)
Var_forecast_error_MLR = np.round(np.var(forecast_error_MLR),3)
print('The mean of the forecast error of MLR model is', Mean_forecast_error_MLR)
print('The variance of the forecast error of MLR model is', Var_forecast_error_MLR)
print('The Q value of the forecast error of MLR model is', Q_forecast_error_MLR)
print('MSE for Multiple Linear Regression Forecast is', MSE_MLR)
print()

# Plot the ACF of forecast error
autocorr(forecast_error_MLR,50,'forecast error of Multiple Linear Regression')

# Plot train and test set versus the one-step ahead forecast
plt.figure(figsize=(20,10))
plt.plot(Y_train[-250:], label='Train')
plt.plot(Y_test[:250], label='Test')
plt.plot(Y_test_pred[:250], label='Multiple Linear Regression one-step ahead Forecast')
plt.legend(fontsize=14)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Traffic Volume', fontsize=16)
plt.title('The Multiple Linear Regression one-step ahead Forecast with MSE = {}\n(Last 250 samples of training set and first 250 samples of testing set)'.format(MSE_MLR), fontsize=20)
plt.show()

# b) Hypothesis tests analysis: F-test,t-test
print("# ===== Final Model after Feature Selection =====")
print(model_MLR.summary())
print()

# c) AIC, BIC, RMSE, R-squared, and Adjusted R-squared
# Perform one-step ahead prediction on X_train
Y_train_pred = np.zeros(len(X_train))
for t in range(len(X_train)):
    if t == 0:
        Y_train_pred[t] = None
    else:
        Y_train_pred[t] = 10.9367*X_train['temp'].values[t-1] + 486.3230*X_train['weather_main_Clouds'].values[t-1] + 203.2478*X_train['weather_main_Rain'].values[t-1]
Y_train_pred = pd.DataFrame(Y_train_pred).set_index(Y_train.index)

# AIC and BIC
print('The AIC of MLR model is', np.round(model_MLR.aic,3))
print('The BIC of MLR model is', np.round(model_MLR.bic,3))

# Calculate the residuals and RMSE (Root Mean Square Error)
residuals_MLR = Y_train[1:].values-Y_train_pred[1:].values
RMSE_residuals_MLR = np.std(residuals_MLR)
print('The RMSE of MLR model is', np.round(RMSE_residuals_MLR,3))

# R-squared and Adjusted R-squared
print('The R-squared of MLR model is', 0.740)
print('The Adjusted R-squared of MLR model is', 0.740)

# d) ACF of residuals
autocorr(residuals_MLR.flatten(),50,'the Residuals of Multiple Linear Regression')

# e) Q-value
Q_residuals_MLR = Q_value(residuals_MLR.flatten(),lags)
print('The Q-value of the residuals of MLR model is', np.round(Q_residuals_MLR,3))

# f) Variance and Mean of the residuals
Mean_residual_MLR = np.round(np.mean(residuals_MLR.flatten()),3)
Var_residual_MLR = np.round(np.var(residuals_MLR.flatten()),3)
print('The mean of the residuals of MLR model is', Mean_residual_MLR)
print('The variance of the residuals of MLR model is', Var_residual_MLR)
print()


# ============================== ARMA and ARIMA and SARIMA ===========================================================================================================================================================================
print('# ======================= ARMA and ARIMA and SARIMA ========================================================================================== #')
# Split differenced dataset (stationary) into 80% training set and 20% testing set
train_diff, test_diff = train_test_split(traffic_diff, test_size=0.2, shuffle=False)

# a) Implement differenced train (stationary) to GPAC table to estimate the order of SARIMA model (non-seasonal differencing + seasonal differencing)  (Not useful for S=168)
ry_SARIMA_train = autocorr_values(train_diff,lags)
Cal_GPAC(ry_SARIMA_train,10,10,'Differenced Traffic Volume (Training Set)')

# b) Plot ACF of differenced dataset (stationary)
ACF_PACF_Plot(train_diff,lags*2,'Differenced Traffic Volume (Training Set)')                       # lags = 200
ACF_PACF_Plot(train_diff,lags*10,'Differenced of Traffic Volume (Training Set)')    # lags = 1000
# Estimated Model from ACF/PACF is ARIMA(0,1,3)xARIMA(0,1,1)_s168, which is equalivalent to ARMA(0,171)


# ============================== Estimate Parameters of ARMA Model ===========================================================================================================================================================================
print('# ======================= Estimate Parameters of ARMA Model ========================================================================================= #')
# Use package to estimate the parameters   (Since ARMA(0,171) is too large, the computer could not work, try other order number of na and nb)
na = 2
nb = 2
model_ARMA = sm.tsa.ARMA(train_diff,(na,nb)).fit(trend='nc',disp=0)
for i in range(na):
    print('The AR coefficient a{}'.format(i), 'is:', model_ARMA.params[i])
for i in range(nb):
    print('The MA coefficient b{}'.format(i), 'is:', model_ARMA.params[i+na])
print(model_ARMA.summary())
print()

# Use SARIMA package to estimate the parameter  (Since 168 is too large, the computer could not work)
# model = sm.tsa.statespace.SARIMAX(traffic_series, order=(0,1,3), seasonal_order=(0,1,1,168), simple_differencing=True)
# model_SARIMA = model.fit(disp=0)
# print(model_SARIMA.summary())


# ============================== Diagnostic Analysis ===========================================================================================================================================================================
print('# ======================= Diagnostic Analysis ========================================================================================================================================================================== #')

# a) Diagnostic Test (confidence interval, zero/pole cancellation, chi-square test)
# zero/pole cancellation
num = [1,-0.7514,-0.2159]            # ma
den = [1,0.1934,0.3454]              # ar
print('The roots of numerator is\n', np.roots(num))
print('The roots of denominator is\n', np.roots(den))
print()

# Chi-Square Test
# Calculate residuals
model_train_hat = model_ARMA.predict(start=0,end=len(train_diff)-1)
residual_ARMA_train = train_diff-model_train_hat

# Plot the ACF of residuals
autocorr(residual_ARMA_train,50,'the residuals of $ARMA(2,2)$ model')

# Calculate the Q value of residuals
ry_ARMA_train = autocorr_values(residual_ARMA_train,lags)
Q_ARMA_train = len(train_diff)*np.sum(np.square(ry_ARMA_train[lags+1:]))
DOF = lags-na-nb
alfa = 0.01
chi_critical = chi2.ppf(1-alfa, DOF)   # Qc
if Q_ARMA_train < chi_critical:
    print('The residual is white')
else:
    print('The residual is NOT white')
lbvalue, pvalue = sm.stats.acorr_ljungbox(residual_ARMA_train,lags=[lags])
print('The Q value is', lbvalue)
print('The chi critical is', chi_critical)
print('The p-value of chi square test is', pvalue)   # p-value of Q test

# b) Display the variance of error and the estimated covariance of estimated parameters
# Calculate the variance of error
system_rev = (den, num, 1)
_, e_theta = signal.dlsim(system_rev, train_diff)   # generate e(theta1,theta2,...,thetan)
SSE_e_theta = np.dot(e_theta.T,e_theta)
var_e = SSE_e_theta/(len(train_diff)-(na+nb))
print('The estimated variance of error is', var_e.flatten()[0])

# Calculate the estimated covariance of estimated parameters
theta = np.array([0.1934,0.3454,-0.7514,-0.2159])
delta = 10e-6
A = A_Cal(theta,na,nb,delta,train_diff)
covar_matrix = var_e.flatten()[0]*np.linalg.inv(A)
print('The estimated covariance of the a1 is', covar_matrix.item(0))
print('The estimated covariance of the a2 is', covar_matrix.item(5))
print('The estimated covariance of the a3 is', covar_matrix.item(10))
print('The estimated covariance of the b1 is', covar_matrix.item(15))
print()

# c) Is the derived model biased or this is an unbiased estimator?
Mean_residual_ARMA = np.mean(residual_ARMA_train)
print('The mean of the residual of ARMA(2,2) is', np.round(Mean_residual_ARMA,3))

# d) Check the variance of residual errors versus the variance of forecast errors
model_test_hat = model_ARMA.predict(start=0,end=len(test_diff)-1)
forecast_error_ARMA_test = test_diff-model_test_hat

# Calculate the rate
rate = np.var(residual_ARMA_train)/np.var(forecast_error_ARMA_test)
print('The variance of residual errors versus the variance of forecast errors is', rate)
print()

# Plot the train set, test set and forecast with ARMA(2,2) model
t = np.arange(len(traffic_diff))
plt.figure(figsize=(20,10))
plt.plot(t[:len(train_diff)][-100:], train_diff[-100:], label='Train')
plt.plot(t[len(train_diff):][:100],test_diff[:100], label='Test')
plt.plot(t[len(train_diff):][:100], model_test_hat[:100], label='$ARMA(2,2)$ Forecast')
plt.legend(fontsize=14)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Traffic Volume', fontsize=16)
plt.title('The $ARMA(2,2)$ Forecast on the Differenced Dataset (Stationary) \n(Last 100 samples of training set and first 100 samples of testing set)', fontsize=20)
plt.show()

# Forecast Function
# ARMA(2,2)
# ARIMA(2,1,2)xARIMA(0,1,0)_s168
# (1-q^-1)(1-q^-168)(1-(a1q^-1)-(a2q^-2))y(t)=(1-(b1q^-1)-(b2q^-2))e(t)
# y(t)-1.1934y(t-1)-0.152y(t-2)+0.3454y(t-3)-y(t-168)+1.1934y(t-169)+0.152y(t-170)-0.3454y(t-171) = e(t)+0.7514e(t-1)+0.2159e(t-2)
# y(t) = 1.1934y(t-1)+0.152y(t-2)-0.3454y(t-3)+y(t-168)-1.1934y(t-169)-0.152y(t-170)+0.3454y(t-171)+e(t)+0.7514e(t-1)+0.2159e(t-2)

# Original Function: y(t)=1.1934y(t-1)+0.152y(t-2)-0.3454y(t-3)+y(t-168)-1.1934y(t-169)-0.152y(t-170)+0.3454y(t-171)+e(t)+0.7514e(t-1)+0.2159e(t-2)
def step_ahead_prediction(y,test):
    pred = np.zeros(len(test))
    for i in range(len(pred)):
        t = 42039
        h = i + 1
        if i == 0:
            pred[i] = 1.193*y[t+h-1]+0.152*y[t+h-2]-0.3454*y[t+h-3]+y[t+h-168]-1.1934*y[t+h-169]-0.152*y[t+h-170]+0.3454*y[t+h-171]+0.7514*(y[t+h-1])+0.2159*(y[t+h-2])
        elif i == 1:
            pred[i] = 1.193*pred[i-1]+0.152*y[t+h-2]-0.3454*y[t+h-3]+y[t+h-168]-1.1934*y[t+h-169]-0.152*y[t+h-170]+0.3454*y[t+h-171]+0.2159*(y[t+h-2])
        elif i == 2:
            pred[i] = 1.193*pred[i-1]+0.152*pred[i-1]-0.3454*y[t+h-3]+y[t+h-168]-1.1934*y[t+h-169]-0.152*y[t+h-170]+0.3454*y[t+h-171]
        elif 3 <= i <= 167:
            pred[i] = 1.193*pred[i-1]+0.152*pred[i-1]-0.3454*pred[i-1]+y[t+h-168]-1.1934*y[t+h-169]-0.152*y[t+h-170]+0.3454*y[t+h-171]
        elif i == 168:
            pred[i] = 1.193*pred[i-1]+0.152*pred[i-1]-0.3454*pred[i-1]+pred[i-1]-1.1934*y[t+h-169]-0.152*y[t+h-170]+0.3454*y[t+h-171]
        elif i == 169:
            pred[i] = 1.193*pred[i-1]+0.152*pred[i-1]-0.3454*pred[i-1]+pred[i-1]-1.1934*pred[i-1]-0.152*y[t+h-170]+0.3454*y[t+h-171]
        elif i == 170:
            pred[i] = 1.193*pred[i-1]+0.152*pred[i-1]-0.3454*pred[i-1]+pred[i-1]-1.1934*pred[i-1]-0.152*pred[i-1]+0.3454*y[t+h-171]
        else:
            pred[i] = 1.193*pred[i-1]+0.152*pred[i-1]-0.3454*pred[i-1]+pred[i-1]-1.1934*pred[i-1]-0.152*pred[i-1]+0.3454*pred[i-1]
    return pred

# SARIMA h-step Forecast on Raw Dataset (Non-Stationary)
SARIMA_h_ahead_pred = step_ahead_prediction(train, test)
SARIMA_h_ahead_pred = pd.DataFrame(SARIMA_h_ahead_pred).set_index(test.index)

# Calculate the forecast error, MSE, Q value, Mean, Variance
forecast_error_SARIMA = test.values-np.ndarray.flatten(SARIMA_h_ahead_pred.values)
MSE_SARIMA = np.round(np.square(forecast_error_SARIMA).mean(),3)
Q_SARIMA = np.round(Q_value(forecast_error_SARIMA,lags),3)
Mean_SARIMA = np.round(np.mean(forecast_error_SARIMA),3)
Var_SARIMA = np.round(np.var(forecast_error_SARIMA),3)
print('The mean of the forecast error of ARIMA(2,1,2)xARIMA(0,1,0)_s168 is', Mean_SARIMA)
print('The variance of the forecast error of ARIMA(2,1,2)xARIMA(0,1,0)_s168 is', Var_SARIMA)
print('The Q value of the forecast error of ARIMA(2,1,2)xARIMA(0,1,0)_s168 is', Q_SARIMA)
print('MSE for ARIMA(2,1,2)xARIMA(0,1,0)_s168 Forecast is', MSE_SARIMA)
print()

# Plot train set, test set and h-step forecast of ARIMA(2,1,2)xARIMA(0,1,0)_s168 model
fig, ax = plt.subplots(figsize=(20,10))
ax.plot(train[-250:], label='Train')
ax.plot(test[:250], label='Test')
ax.plot(SARIMA_h_ahead_pred[:250], label="Holt's Linear Trend Method Forecast")
plt.legend(fontsize=14)
plt.title('$ARIMA(2,1,2)xARIMA(0,1,0)_{168}$'+ 'h-step Forecast with the MSE = {}\n(Last 250 samples of training set and first 250 samples of testing set)'.format(MSE_SARIMA), fontsize=20)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Traffic Volume', fontsize=16)
plt.show()

# ============================== Base-Models ===========================================================================================================================================================================
print('# ======================= Base-Models ========================================================================================================================================================================== #')

# Average Method (h-step ahead)
y_hat_avg = np.zeros(len(test))
for i in range(len(test)):
    y_hat_avg[i] = np.mean(train)
y_hat_avg = pd.DataFrame(y_hat_avg).set_index(test.index)

# Calculate the forecast error, MSE, Q value, Mean, Variance
forecast_error_Avg = test.values-np.ndarray.flatten(y_hat_avg.values)
MSE_Avg = np.round(np.square(forecast_error_Avg).mean(),3)
Q_Avg = np.round(Q_value(forecast_error_Avg,lags),3)
Mean_Avg = np.round(np.mean(forecast_error_Avg),3)
Var_Avg = np.round(np.var(forecast_error_Avg),3)
print('The mean of the forecast error of Average Method is', Mean_Avg)
print('The variance of the forecast error of Average Method is', Var_Avg)
print('The Q value of the forecast error of Average Method is', Q_Avg)
print('MSE for Average Method Forecast is', MSE_Avg)
print()

# Plot the test set, train set and the h-step forecast of average method
fig,ax = plt.subplots(figsize=(20,10))
ax.plot(train[-250:], label='Train')
ax.plot(test[:250], label='Test')
ax.plot(y_hat_avg[:250], label='Average Method Forecast')
plt.legend(fontsize=14)
plt.title('Average Method h-step Forecast with MSE = {}\n(Last 250 samples of training set and first 250 samples of testing set)'.format(MSE_Avg), fontsize=20)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Traffic Volume', fontsize=16)
plt.show()

# Naive Method (h-step ahead)
y_hat_naive = np.zeros(len(test))
for i in range(len(test)):
    y_hat_naive[i] = train[-1]
y_hat_naive = pd.DataFrame(y_hat_naive).set_index(test.index)

# Calculate the forecast error, MSE, Q value, Mean, Variance
forecast_error_Naive = test.values-np.ndarray.flatten(y_hat_naive.values)
MSE_Naive = np.round(np.square(forecast_error_Naive).mean(),3)
Q_Naive = np.round(Q_value(forecast_error_Naive,lags),3)
Mean_Naive = np.round(np.mean(forecast_error_Naive),3)
Var_Naive = np.round(np.var(forecast_error_Naive),3)
print('The mean of the forecast error of Naive Method is', Mean_Naive)
print('The variance of the forecast error of Naive Method is', Var_Naive)
print('The Q value of the forecast error of Naive Method is', Q_Naive)
print('MSE for Naive Method Forecast is', MSE_Naive)
print()

# Plot the test set, training set and the h-step forecast of naive method
fig,ax = plt.subplots(figsize=(20,10))
ax.plot(train[-250:], label='Train')
ax.plot(test[:250], label='Test')
ax.plot(y_hat_naive[:250], label='Naive Method Forecast')
plt.legend(fontsize=14)
plt.title('Naive Method h-step Forecast with MSE = {}\n(Last 250 samples of training set and first 250 samples of testing set)'.format(MSE_Naive), fontsize=20)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Traffic Volume', fontsize=16)
plt.show()

# Seasonal Naive Method (h-step ahead)
y_hat_seasonal_naive = np.zeros(len(test))
for i in range(len(test)):
    k = int(i/168)
    m_k = 168*(k+1)
    y_hat_seasonal_naive[i] = train[i-m_k]
y_hat_seasonal_naive = pd.DataFrame(y_hat_seasonal_naive).set_index(test.index)

# Calculate the forecast error, MSE, Q value, Mean, Variance
forecast_error_Seasonal_Naive = test.values-np.ndarray.flatten(y_hat_seasonal_naive.values)
MSE_Seasonal_Naive = np.round(np.square(forecast_error_Seasonal_Naive).mean(),3)
Q_Seasonal_Naive = np.round(Q_value(forecast_error_Seasonal_Naive,lags),3)
Mean_Seasonal_Naive = np.round(np.mean(forecast_error_Seasonal_Naive),3)
Var_Seasonal_Naive = np.round(np.var(forecast_error_Seasonal_Naive),3)
print('The mean of the forecast error of Seasonal Naive Method is', Mean_Seasonal_Naive)
print('The variance of the forecast error of Seasonal Naive Method is', Var_Seasonal_Naive)
print('The Q value of the forecast error of Seasonal Naive Method is', Q_Seasonal_Naive)
print('MSE for Seasonal Naive Method Forecast is', MSE_Seasonal_Naive)
print()

# Plot the test set, training set and the h-step forecast of Seasonal Naive method
fig,ax = plt.subplots(figsize=(20,10))
ax.plot(train[-250:], label='Train')
ax.plot(test[:250], label='Test')
ax.plot(y_hat_seasonal_naive[:250], label='Seasonal Naive Method Forecast')
plt.legend(fontsize=14)
plt.title('Seasonal Naive Method h-step Forecast with MSE = {}\n(Last 250 samples of training set and first 250 samples of testing set)'.format(MSE_Seasonal_Naive), fontsize=20)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Traffic Volume', fontsize=16)
plt.show()


# Drift Method (h-step ahead)
y_hat_drift = np.zeros(len(test))
for i in range(len(test)):
    slope = (train[-1]-train[0])/(len(train)-1)
    y_hat_drift[i] = train[-1]+(i+1)*slope
y_hat_drift = pd.DataFrame(y_hat_drift).set_index(test.index)

# Calculate the forecast error, MSE, Q value, Mean, Variance
forecast_error_Drift = test.values-np.ndarray.flatten(y_hat_drift.values)
MSE_Drift = np.round(np.square(forecast_error_Drift).mean(),3)
Q_Drift = np.round(Q_value(forecast_error_Drift,lags),3)
Mean_Drift = np.round(np.mean(forecast_error_Drift),3)
Var_Drift = np.round(np.var(forecast_error_Drift),3)
print('The mean of the forecast error of Drift Method is', Mean_Drift)
print('The variance of the forecast error of Drift Method is', Var_Drift)
print('The Q value of the forecast error of Drift Method is', Q_Drift)
print('MSE for Drift Method Forecast is', MSE_Drift)
print()

# Plot the test set, train set and the h-step forecast of Drift method
fig,ax = plt.subplots(figsize=(20,10))
ax.plot(train[-250:], label='Train')
ax.plot(test[:250], label='Test')
ax.plot(y_hat_drift[:250], label='Drift Method Forecast')
plt.legend(fontsize=14)
plt.title('Drift Method h-step Forecast with MSE = {}\n(Last 250 samples of training set and first 250 samples of testing set)'.format(MSE_Drift), fontsize=20)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Traffic Volume', fontsize=16)
plt.show()

# SES (Simple Exponential Smoothing)
holtt = ets.ExponentialSmoothing(train, trend=None, damped_trend=False, seasonal=None).fit()
holt_SES_pred = holtt.forecast(steps=len(test))
holt_SES_pred = pd.DataFrame(holt_SES_pred).set_index(test.index)

# Calculate the forecast error, MSE, Q value, Mean, Variance
forecast_error_SES= test.values-np.ndarray.flatten(holt_SES_pred.values)
MSE_SES = np.round(np.square(forecast_error_SES).mean(),3)
Q_SES = np.round(Q_value(forecast_error_SES,lags),3)
Mean_SES = np.round(np.mean(forecast_error_SES),3)
Var_SES = np.round(np.var(forecast_error_SES),3)
print('The mean of the forecast error of Simple Exponential Smoothing Method is', Mean_SES)
print('The variance of the forecast error of Simple Exponential Smoothing Method is', Var_SES)
print('The Q value of the forecast error of Simple Exponential Smoothing Method is', Q_SES)
print('MSE for Drift Method Forecast is', MSE_SES)
print()

# Plot the test set, train set and the h-step forecast of Simple Exponential Smoothing
fig,ax = plt.subplots(figsize=(20,10))
ax.plot(train[-250:], label='Train')
ax.plot(test[:250], label='Test')
ax.plot(holt_SES_pred[:250], label='Simple Exponential Smoothing Forecast')
plt.legend(fontsize=14)
plt.title('Simple Exponential Smoothing Method h-step Forecast with MSE = {}\n(Last 250 samples of training set and first 250 samples of testing set)'.format(MSE_SES), fontsize=20)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Traffic Volume', fontsize=16)
plt.show()

# Holt’s Linear Trend method
holtt = ets.ExponentialSmoothing(train, trend='additive', damped_trend=True, seasonal=None).fit()
holt_Linear_pred = holtt.forecast(steps=len(test))
holt_Linear_pred = pd.DataFrame(holt_Linear_pred).set_index(test.index)
# MSE_holt_linear = np.square(np.subtract(test.values, np.ndarray.flatten(holt_Linear_pred.values))).mean()

# Calculate the forecast error, MSE, Q value, Mean, Variance
forecast_error_holt_Linear= test.values-np.ndarray.flatten(holt_Linear_pred.values)
MSE_holt_Linear = np.round(np.square(forecast_error_holt_Linear).mean(),3)
Q_holt_Linear = np.round(Q_value(forecast_error_holt_Linear,lags),3)
Mean_holt_Linear = np.round(np.mean(forecast_error_holt_Linear),3)
Var_holt_Linear = np.round(np.var(forecast_error_holt_Linear),3)
print('The mean of the forecast error of Holt Linear Trend Method is', Mean_holt_Linear)
print('The variance of the forecast error of Holt Linear Trend Method is', Var_holt_Linear)
print('The Q value of the forecast error of Holt Linear Trend Method is', Q_holt_Linear)
print('MSE for Drift Method Forecast is', MSE_holt_Linear)
print()

# Plot the test set, train set and the h-step forecast of Holt's Linear Method
fig,ax = plt.subplots(figsize=(20,10))
ax.plot(train[-250:], label='Train')
ax.plot(test[:250], label='Test')
ax.plot(holt_Linear_pred[:250], label="Holt's Linear Trend Method Forecast")
plt.legend(fontsize=14)
plt.title("Holt's Linear Trend h-step Forecast with MSE = {}\n(Last 250 samples of training set and first 250 samples of testing set".format(MSE_holt_Linear), fontsize=20)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Traffic Volume', fontsize=16)
plt.show()


# ============================== Final Model Selection ===========================================================================================================================================================================
print('# ======================= Final Model Selection ========================================================================================================================================================================== #')

# Compare the 9 Forecast Method (Q values, MSE, mean of prediction errors, variance of prediction error).
method = ['Average Method', 'Naive Method', 'Seasonal Naive Method', 'Drift Method', 'Simple Exponential Method', "Holt's Linear Trend Method", 'Holt_Winter Method', 'Multiple Linear Regression', 'ARIMA(2,1,2)xARIMA(0,1,0)_s168']
Q = [Q_Avg, Q_Naive, Q_Seasonal_Naive, Q_Drift, Q_SES, Q_holt_Linear, Q_holt_winter, Q_forecast_error_MLR, Q_SARIMA]
MSE = [MSE_Avg,MSE_Naive, MSE_Seasonal_Naive, MSE_Drift, MSE_SES, MSE_holt_Linear,MSE_holt_winter,MSE_MLR,MSE_SARIMA]
Mean = [Mean_Avg, Mean_Naive, Mean_Seasonal_Naive, Mean_Drift, Mean_SES, Mean_holt_Linear, Mean_holt_winter, Mean_forecast_error_MLR, Mean_SARIMA]
Var = [Var_Avg, Var_Naive, Var_Seasonal_Naive, Var_Drift, Var_SES, Var_holt_Linear, Var_holt_winter, Var_forecast_error_MLR, Var_SARIMA]

d = {'Method': method,
     'Q value': Q,
     'MSE': MSE,
     'Mean of Prediction Error': Mean,
     'Variance of Prediction Error': Var
    }
df_compare = pd.DataFrame(d, columns = ['Method', 'Q value', 'MSE', 'Mean of Prediction Error', 'Variance of Prediction Error'])
print(df_compare)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))