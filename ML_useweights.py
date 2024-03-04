import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import matplotlib.pyplot as plt

data = pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw3_200_1p10p_leak10m.csv", on_bad_lines='warn')
X = data.iloc[:,:10].values
Y = data.iloc[:,10].values
scale = 1000/X.shape[0]
X_train,X_test1,Y_train,Y_test = train_test_split(X,Y,test_size=scale,random_state=1)
sc = StandardScaler()

#scales for spice models 4 switch counts
Mean = [4.63617199e+01,4.63556812e+01,4.64200910e+01,3.00000000e+00,1.39137492e+02,5.49798157e-12,3.99483896e-02,3.99432579e-02,3.99979934e-02,1.00000000e-02]
Scale = [3.36415078e+01,3.36474938e+01,3.36630523e+01,1.00000000e+00,2.19234917e+01,2.87260159e-12,2.85885533e-02,2.85936403e-02,2.86068618e-02,1.00000000e+00]

sc.mean_ = np.array(Mean)
sc.scale_ = np.array(Scale)
X_test = sc.transform(X_test1)
model = Sequential()
model.add(Dense(64, input_dim=10, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.load_weights('/home/vgopal18/python/ML_data/model_weights_spice3stages_2.h5')
#model.load_weights('/home/vgopal18/python/ML_data/model_weights_filtercombined_3456.h5')
Y_predict = model.predict(X_test)
print("Ans:",Y_predict.shape)
mse = mean_squared_error(Y_test, Y_predict)
mae = mean_absolute_error(Y_test,Y_predict)
rmse = np.sqrt(mse)
max_err = max_error(Y_test, Y_predict)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
#print("Maximum Error:", max_err)
print(min(Y_test))

Y_predict = Y_predict.reshape(-1)
print(Y_predict.shape,Y_test.shape)
errors = Y_predict - Y_test
abs_errors = abs(errors)
#print(abs_errors)
error_percentages = (abs_errors / Y_test) * 100
print(errors.shape, error_percentages.shape)
print("Max Error : ", max(abs_errors))
print("Maximum Error %:", max(error_percentages))

df = pd.DataFrame(X_test1, columns = data.columns[:10])
df['Actual I'] = Y_test
df['Predict I'] = Y_predict
df['Error'] = errors
df['Err %'] = error_percentages
df.to_csv('sw7_error.csv', index=False)

#plt.scatter(Y_test, error_percentages)
plt.scatter(Y_test, errors)
plt.axhline(y=0, color='black', linestyle='--')
#print(errors)
#plt.xticks(ticks=[i for i in range(len(errors))], labels=Y_predict)
#plt.scatter(np.arange(len(errors)+1),errors)
plt.xlabel('Actual Value')
plt.ylabel('Error Percent')
plt.ylabel('Error Value')
plt.title('Error Difference of Regression Model')
plt.show()
