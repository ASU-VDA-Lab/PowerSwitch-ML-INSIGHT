import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import matplotlib.pyplot as plt

data1 = pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw3_50_1p10p_leak10m.csv_filtered", on_bad_lines='warn')
data2 = pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw3_150_1p10p_leak10m.csv_filtered", on_bad_lines='warn')
data3 = pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw3_200_1p10p_leak10m.csv_filtered", on_bad_lines='warn')
data4 = pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw3_150_1p10p_leak10m.csv_filtered", on_bad_lines='warn')
data = pd.concat([data1, data3], axis = 0, ignore_index=True)
print(data.shape)
print(data.columns)

rows_with_nan = data[data.isnull().any(axis=1)]
print(rows_with_nan)
data.dropna(inplace=True)

X = data.iloc[:,:10].values
Y = data.iloc[:,10].values
print(X.shape,Y.shape)
print(X[0],Y[0])

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=1)
print(X_train[1:3],Y_train[3])

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print('Mean:',sc.mean_)
print('Scale:',sc.scale_)

model = Sequential()
model.add(Dense(64, input_dim=10, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.fit(X_train, Y_train, verbose=0, epochs=100)
model.save_weights('/home/vgopal18/python/ML_data/model_weights_spice3stages_minmax.h5')

#Testing
print(X_test.shape,Y_test.shape)
Y_predict = model.predict(X_test)
print("Ans:",Y_predict.shape)
mse = mean_squared_error(Y_test, Y_predict)
mae = mean_absolute_error(Y_test,Y_predict)
rmse = np.sqrt(mse)
max_err = max_error(Y_test, Y_predict)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Maximum Error:", max_err)
print(min(Y_test))
Y_predict = Y_predict.reshape(-1)
print(Y_predict.shape,Y_test.shape)
errors = Y_predict - Y_test
error_percentages = (errors / Y_test) * 100
print(errors.shape, error_percentages.shape)
print("Maximum Error:", max(errors))
print("Maximum Error %:", max(error_percentages))
plt.scatter(Y_test, error_percentages)
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Actual Value')
plt.ylabel('Error Percent')
plt.title('Error Difference of Regression Model')
plt.show()
