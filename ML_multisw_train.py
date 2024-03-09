import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras import backend as K
import random

# Set random seeds
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
K.clear_session()

data = []
data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw3_100_5p_leak10m.csv", on_bad_lines='warn'))
data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw3_110_5p_leak10m.csv", on_bad_lines='warn'))
data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw3_120_5p_leak10m.csv", on_bad_lines='warn'))

for i in range(len(data)):
  #dropping Cap, Ileak, Stage number columns
    data[i] = data[i].drop(data[i].columns[9],axis=1)
    #data[i] = data[i].drop(data[i].columns[8],axis=1)
    #data[i] = data[i].drop(data[i].columns[7],axis=1)
    #data[i] = data[i].drop(data[i].columns[6],axis=1)
    data[i] = data[i].drop(data[i].columns[5],axis=1)
    #data[i] = data[i].drop(data[i].columns[4],axis=1)
    data[i] = data[i].drop(data[i].columns[3],axis=1)
    rows_with_nan = data[i][data[i].isnull().any(axis=1)]
    print(data[i].shape)
    print(data[i].columns)
    print(rows_with_nan)
    data[i].dropna(inplace=True)
data_c = pd.concat(data, axis = 0, ignore_index=True)
print(data_c.shape,data_c.columns)
X = data_c.iloc[:,:7].values
Y = data_c.iloc[:,7].values
print(X.shape,Y.shape)

X_train,X_test1,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state=1)
df = pd.DataFrame(X_test1, columns = data_c.columns[:7])
df['Actual I'] = Y_test
df.to_csv('test_data.csv', index=False)
df = pd.DataFrame(X_train, columns = data_c.columns[:7])
df['Actual I'] = Y_train
df.to_csv('train_data.csv', index=False)

#X_train = X
#Y_train = Y
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test1)
print('Mean:',sc.mean_)
print('Scale:',sc.scale_)

model = Sequential()
model.add(Dense(64, input_dim=7, kernel_initializer='he_uniform', activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
#early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, Y_train, verbose=1,epochs=200)
model.save_weights('/home/vgopal18/python/ML_data/model_weights_spice3stage_nocap_comb3_l12.h5')

#Plot training metrics
history_df = pd.DataFrame(history.history)
history_df.to_csv('history.csv', index=False)
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('loss_plot.png')
plt.close()
plt.plot(history.history['mae'], label='Training MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('Training MAE')
plt.savefig('mae_plot.png')
plt.close()

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
abs_errors = abs(errors)
error_percentages = (abs_errors / Y_test) * 100
print(errors.shape, error_percentages.shape)
print("Maximum Error:", max(abs_errors))
print("Maximum Error %:", max(error_percentages))

fig, axs = plt.subplots(2)
axs[0].scatter(Y_test, error_percentages)
axs[1].scatter(Y_test, errors)
plt.axhline(y=0, color='black', linestyle='--')
axs[0].set_xlabel('Actual Value')
axs[1].set_xlabel('Actual Value')
axs[0].set_ylabel('Error Percent')
axs[1].set_ylabel('Error Value')
axs[1].set_title('Error Difference of Regression Model')
axs[0].set_title('% Error Difference of Regression Model')
plt.tight_layout()
plt.show()

df = pd.DataFrame(X_test1, columns = data_c.columns[:7])
df['Actual I'] = Y_test
df['Predict I'] = Y_predict
df['Error'] = errors
df['Err %'] = error_percentages
df.to_csv('sw1_error.csv', index=False)
