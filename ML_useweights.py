import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import matplotlib.pyplot as plt

#data = pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw1_2000.csv", on_bad_lines='warn')
data = pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw3_250_5p_leak10m.csv", on_bad_lines='warn')
data = data.drop(data.columns[9],axis=1)
#data = data.drop(data.columns[8],axis=1)
#data = data.drop(data.columns[7],axis=1)
#data = data.drop(data.columns[6],axis=1)
data = data.drop(data.columns[5],axis=1)
#data = data.drop(data.columns[4],axis=1)
data = data.drop(data.columns[3],axis=1)

print(data.columns)
X = data.iloc[:,:7].values
Y = data.iloc[:,7].values
print(X.shape,Y.shape)
print(X[1])
scale = 1000/X.shape[0]
X_train,X_test1,Y_train,Y_test = train_test_split(X,Y,test_size=scale,random_state=1)
print(X_train[0],Y_train[0])
sc = StandardScaler()

#scales for combined ML model 4 and 5
#Mean = [2.91428629e+01,2.91266090e+01,2.91480973e+01,4.81183389e+00,1.40867560e+02,4.57005426e-12,2.53158049e+01,2.53019923e+01,2.53202531e+01,1.04965581e-03]
#Scale = [3.19818628e+01,3.19647957e+01,3.19914219e+01,4.04715024e-01,1.93189167e+01,2.70250195e-12,2.71781870e+01,2.71636834e+01,2.71863103e+01,6.66660748e-04]

#scales of combined ML model 4,5 and 6
#Mean = [1.86176974e+01,1.86183083e+01,1.86063702e+01,5.46700633e+00,1.00000000e+02,5.08406618e-12,1.63715193e+01,1.63720384e+01,1.63618934e+01,1.02547231e-03]
#Scale = [2.05131740e+01,2.05170541e+01,2.05124647e+01,6.68106750e-01,1.00000000e+00,2.90918216e-12,1.74320952e+01,1.74353926e+01,1.74314925e+01,4.78123362e-04]

#scales of filtered and combined ML model 4,5 and 6
#Mean = [5.36820993e+01,2.67990594e+01,1.36374051e+01,4.68638534e+00,1.00000000e+02,5.04228512e-12,4.61692480e+01,2.33240407e+01,1.21392668e+01,1.44224629e-03]
#Scale = [1.33045533e+01,8.72862270e+00,6.95305186e+00,8.41690624e-01,1.00000000e+00,2.89354342e-12,1.13062094e+01,7.41758357e+00,5.90870347e+00,1.94541379e-03]

#scales of spice model
#Mean = [3.33999433e+01,3.32938312e+01,3.33062255e+01,3.00000000e+00,1.00000000e+02,5.49458874e-12,2.89334718e-02,2.88432977e-02,2.88538304e-02,1.00000000e-03]
#Scale = [2.31934380e+01,2.31963725e+01,2.32313861e+01,1.00000000e+00,1.00000000e+00,2.87365747e-12,1.97097836e-02,1.97122774e-02,1.97420319e-02,1.00000000e+00]

#scales for spice models 4 switch counts
#Mean = [4.63617199e+01,4.63556812e+01,4.64200910e+01,3.00000000e+00,1.39137492e+02,5.49798157e-12,3.99483896e-02,3.99432579e-02,3.99979934e-02,1.00000000e-02]
#Scale = [3.36415078e+01,3.36474938e+01,3.36630523e+01,1.00000000e+00,2.19234917e+01,2.87260159e-12,2.85885533e-02,2.85936403e-02,2.86068618e-02,1.00000000e+00]

#Scales for filtered model
#Mean = [4.64191065e+01,4.64368506e+01,4.64220989e+01,3.00000000e+00,1.39278056e+02,5.49590232e-12,3.99971567e-02,4.00122356e-02,3.99996997e-02,1.00000000e-02]
#Scale = [1.77912049e+01,1.77779605e+01,1.78133980e+01,1.00000000e+00,2.16906372e+01,2.87233599e-12,1.51189659e-02,1.51077108e-02,1.51378256e-02,1.00000000e+00]

#Scles for 50_200 model
#Mean = [6.40266996e+01,6.40622196e+01,6.39434584e+01,3.00000000e+00,1.92032378e+02,5.49561147e-12,5.49600893e-02,5.49902742e-02,5.48893509e-02,1.00000000e-02]
#Scale = [2.52464004e+01,2.52259118e+01,2.52270136e+01,1.00000000e+00,3.36401598e+01,2.87579940e-12,2.14543910e-02,2.14369798e-02,2.14379162e-02,1.00000000e+00]

#Scales for 1 stage
#Mean = [5.49191605e+02,5.52000000e-12,4.67253226e-01,1.00000000e-02]
#Scale = [2.59729297e+02,2.87315717e-12,2.20717956e-01,1.00000000e+00]

#scales for 1 stage 2k sw
#Mean = [5.70845611e+02,5.49574520e-12,4.85654801e-01,1.00000000e-02]
#Scale = [3.12552139e+02,2.87388603e-12,2.65606808e-01,1.00000000e+00]

#Scales for 3 stages 100 swicthes spice model
#Mean = [3.34351938e+01,3.32720573e+01,3.32927489e+01,3.00000000e+00,1.00000000e+02,5.49458874e-12,2.89634277e-02,2.88247943e-02,2.88423780e-02,1.00000000e-02]
#Scale = [2.32657078e+01,2.31469217e+01,2.32082393e+01,1.00000000e+00,1.00000000e+00,2.87365747e-12,1.97711985e-02,1.96702541e-02,1.97223618e-02,1.00000000e+00]

#Scales 3 stages 100 sw no cap
#Mean = [3.33556701e+01,3.33912371e+01,3.32530928e+01,3.00000000e+00,1.00000000e+02,2.88958485e-02,2.89260733e-02,2.88086782e-02]
#Scale = [2.33240903e+01,2.32346951e+01,2.30912883e+01,1.00000000e+00,1.00000000e+00,1.98208120e-02,1.97448439e-02,1.96229768e-02]

#Scales 3 stages 100 sw no cap no slew
#Mean = [3.33556701e+01,3.33912371e+01,3.32530928e+01]
#Scale = [2.33240903e+01,2.32346951e+01,2.30912883e+01]
#Mean = [33.33333333,33.33333333,33.33333333]
#Scale = [23.21398046,23.21398046,23.21398046]

#Scales 3 stages 100 sw with slew
#Mean = [3.32749141e+01,3.33686140e+01,3.33564719e+01,2.88272220e-02,2.89068482e-02,2.88965299e-02]
#Scale = [2.32573338e+01,2.32326049e+01,2.32280557e+01,1.97640822e-02,1.97430676e-02,1.97392018e-02]
#Mean = [3.33556701e+01,3.33912371e+01,3.32530928e+01,2.88958485e-02,2.89260733e-02,2.88086782e-02]
#Scale = [2.33240903e+01,2.32346951e+01,2.30912883e+01,1.98208120e-02,1.97448439e-02,1.96229768e-02]

#Scales 3 stages 100 110 150 with slew
#Mean = [4.28215562e+01,4.27568161e+01,4.24523639e+01,3.69399585e-02,3.68849423e-02,3.66262189e-02]
#Scale = [3.11794211e+01,3.12392293e+01,3.11222281e+01,2.64962721e-02,2.65470971e-02,2.64476694e-02]
#Scales 3 stages 100,110,120
#Mean = [3.70235890e+01,3.72372669e+01,3.69398699e+01,3.20128460e-02,3.21944294e-02,3.19417014e-02]
#Scale = [2.60763418e+01,2.61345037e+01,2.60860402e+01,2.21596753e-02,2.22091012e-02,2.21679170e-02]

#Scale 3 100, 110, 120 with slew, total sw
Mean = [3.70235890e+01,3.72372669e+01,3.69398699e+01,1.08208797e+02,3.20128460e-02,3.21944294e-02,3.19417014e-02]
Scale = [2.60763418e+01,2.61345037e+01,2.60860402e+01,9.51185050e+00,2.21596753e-02,2.22091012e-02,2.21679170e-02]

#X_test1 = [50.0,33.0,17.0,3.0,200.0,2e-12,0.0430402,0.065736,0.0149968,0.01]
#X_test1 = np.array(X_test1).reshape(1,10)
sc.mean_ = np.array(Mean)
sc.scale_ = np.array(Scale)
X_test = sc.transform(X_test1)
print(X_test1,X_test)
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
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
#model.load_weights('/home/vgopal18/python/ML_data/model_weights_spice1stage_2.h5')
#model.load_weights('/home/vgopal18/python/ML_data/model_weights_spice3stage_nocap_l14.h5')
model.load_weights('/home/vgopal18/python/ML_data/model_weights_spice3stage_nocap_comb3_l18.h5')
#model.load_weights('/home/vgopal18/python/ML_data/model_weights_filtercombined_3456.h5')
print(Y_test[1])
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
#print(min(Y_test))

Y_predict = Y_predict.reshape(-1)
print(Y_predict.shape,Y_test.shape)
errors = Y_predict - Y_test
abs_errors = abs(errors)
#print(abs_errors)
error_percentages = (abs_errors / Y_test) * 100
#print(errors.shape, error_percentages.shape)
print("Max Error : ", max(abs_errors))
print("Maximum Error %:", max(error_percentages))

df = pd.DataFrame(X_test1, columns = data.columns[:7])
df['Actual I'] = Y_test
df['Predict I'] = Y_predict
df['Error'] = errors
df['Err %'] = error_percentages
df.to_csv('sw3_error.csv', index=False)

fig, axs = plt.subplots(2)

axs[0].scatter(Y_test, error_percentages)
axs[1].scatter(Y_test, errors)
plt.axhline(y=0, color='black', linestyle='--')
#axs[1].axhline(y=0, color='black', linestyle='--')
#print(errors)
#plt.xticks(ticks=[i for i in range(len(errors))], labels=Y_predict)
#plt.scatter(np.arange(len(errors)+1),errors)
axs[0].set_xlabel('Actual Value')
axs[1].set_xlabel('Actual Value')
axs[0].set_ylabel('Error Percent')
axs[1].set_ylabel('Error Value')
axs[1].set_title('Error Difference of Regression Model')
axs[0].set_title('% Error Difference of Regression Model')
plt.tight_layout()
plt.show()
