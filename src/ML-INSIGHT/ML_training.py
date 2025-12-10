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
import random, math, time
import matplotlib.ticker as ticker
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import joblib

# Set random seeds
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
K.clear_session()

#data = pd.read_csv("/home/vgopal18/python/filtered_train_data.csv", on_bad_lines='warn')
#data = pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw1_1000_5p_leak10m.csv", on_bad_lines='warn')
#data = []
#data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_master_sw6_100_1p10p_leak15u.csv", on_bad_lines='warn'))
#data.append(pd.read_csv("/home/vgopal18/python/ML_data/sw4_100_ileak_1m/sw4_100_1m_2.csv", on_bad_lines='warn'))
#data.append(pd.read_csv("/home/vgopal18/python/ML_data/sw7_100_ileak_1m/sw7_100_1m_2.csv", on_bad_lines='warn'))
#data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw3_100_1p24p_leakAllu.csv", on_bad_lines='warn'))
#data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw3_150_1p24p_leakAllu.csv", on_bad_lines='warn'))
#data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw3_200_1p24p_leakAllu.csv", on_bad_lines='warn'))

files = []

files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw8_100_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw8_100_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw8_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw8_1k_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw83_100_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw83_100_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw83_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw83_1k_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw83_500_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw83_500_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw84_100_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw84_100_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw84_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw84_1k_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw84_500_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw84_500_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw8_500_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw8_500_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw85_100_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw85_100_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw85_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw85_1k_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw85_500_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw85_500_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw86_100_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw86_100_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw86_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw86_1k_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw86_500_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw86_500_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw87_100_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw87_100_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw87_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw87_1k_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw87_500_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_gini_sw87_500_1p10p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sub2_sw8_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sub2_sw83_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sub2_sw84_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sub2_sw85_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sub2_sw86_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sub2_sw87_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sub_sw8_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sub_sw83_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sub_sw84_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sub_sw85_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sub_sw86_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sub_sw87_1k_10p24p_15u.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sw63_100_1p10p_15u_d.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sw63_150_1p24p_15u_d.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sw63_200_1p24p_15u_d.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sw64_100_1p10p_15u_d.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sw64_150_1p24p_15u_d.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sw64_200_1p24p_15u_d.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sw65_100_1p10p_15u_d.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sw65_150_1p24p_15u_d.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sw65_200_1p24p_15u_d.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sw66_100_1p10p_15u_d.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sw66_150_1p24p_15u_d.csv")
files.append("/data/vgopal18/ML_data/V1_data/results_wave_sw66_200_1p24p_15u_d.csv")

data = []

for file in files:
    data.append(pd.read_csv(file, on_bad_lines='warn'))
    #data[i] = data[i].drop(data[i].columns[9],axis=1)
    #data[i] = data[i].drop(data[i].columns[8],axis=1)
    #data[i] = data[i].drop(data[i].columns[7],axis=1)
    #data[i] = data[i].drop(data[i].columns[6],axis=1)
    #data[i] = data[i].drop(data[i].columns[4],axis=1)
    #data[i] = data[i].drop(data[i].columns[3],axis=1)
    #data[i] = data[i].drop(data[i].columns[0],axis=1)

data_c = pd.concat(data, axis = 0, ignore_index=True)
print(data_c.shape,data_c.columns)
data_c = data_c[data_c['MaxI']<=10e-3]

print(data_c.shape,data_c.columns)
X = data_c.iloc[:,:10].values
Y = data_c.iloc[:,10].values
print(X.shape,Y.shape)

X_train,X_test1,Y_train,Y_test = train_test_split(X,Y,test_size=0.4,random_state=1)
#df = pd.DataFrame(X_test1, columns = data_c.columns[:10])
#df['Actual I'] = Y_test
#df.to_csv('test_data.csv', index=False)
#df = pd.DataFrame(X_train, columns = data_c.columns[:10])
#df['Actual I'] = Y_train
#df.to_csv('train_data.csv', index=False)

#X_train = X
#Y_train = Y
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test1)
print('Mean:',sc.mean_)
print('Scale:',sc.scale_)

epochs = 5
batch_size = 64
steps_per_epoch = math.ceil(X_train.shape[0]/batch_size)
steps_per_epoch = 100
print("Epochs :",epochs)
print("Batch Size :",batch_size)
print("Steps per epoch :",steps_per_epoch)
st = time.time()

ml_type = "mlp"

if (ml_type == "mlp"):
    model = Sequential()
    model.add(Dense(64, input_dim=10, kernel_initializer='he_uniform', activation='relu'))
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
    #early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, Y_train, verbose=1,epochs=epochs,batch_size=batch_size,steps_per_epoch=steps_per_epoch,shuffle=True)
    model.save_weights('/home/vgopal18/python/ML_data/mlp_new_lt.h5')
    #model.save_weights('/home/vgopal18/python/ML_data/model_weights_spice6sage_l14.h5')
    print("Time to train model: ",time.time()-st)
    #Plot training metrics
#    history_df = pd.DataFrame(history.history)
#    history_df.to_csv('history.csv', index=False)
#    plt.plot(history.history['loss'], label='Training Loss')
#    plt.xlabel('Epochs')
#    plt.ylabel('Loss')
#    plt.title('Training Loss')
#    plt.savefig('loss_plot.png')
#    plt.close()
#    plt.plot(history.history['mae'], label='Training MAE')
#    plt.xlabel('Epochs')
#    plt.ylabel('Mean Absolute Error')
#    plt.title('Training MAE')
#    plt.savefig('mae_plot.png')
#    plt.close()
elif (ml_type == "xgb"):
    model = xgb.XGBRegressor(
        objective ='reg:squarederror',
        learning_rate = 0.01,
        max_depth = 6,
        n_estimators = 2500,
        colsample_bytree = 1.0,
        subsample = 1.0
    )
    model.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], verbose=10)
    model.save_model("xgb_model_exp_gt.json")
elif (ml_type == "rf"):
    model = RandomForestRegressor(n_estimators=1000, max_depth =6, random_state=42, verbose = 10)
    model.fit(X_train, Y_train)
    joblib.dump(model, 'random_forest_model2.joblib')
elif (ml_type == "svr"):
    #model = SVR(kernel='rbf', C=0.01, epsilon=0.01, gamma='scale')
    model = SVR(kernel='poly', degree=3, C=1.0, coef0=1)
    model.fit(X_train, Y_train)
    
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
plt.savefig('error_diff.png')
plt.close()

df = pd.DataFrame(X_test1, columns = data_c.columns[:10])
df['Actual I'] = Y_test
df['Predict I'] = Y_predict
df['Error'] = errors
df['Err %'] = error_percentages
df.to_csv('sw1_error.csv', index=False)

plt.scatter(Y_test,Y_predict)
llimit = np.min(np.minimum(Y_predict, Y_test))-0.0005
mlimit = np.max(np.maximum(Y_predict,Y_test))+0.0005
print(llimit,mlimit)
t = []
t.append(llimit)
t.append(mlimit)
plt.plot(t, t, color='red', linestyle='--', label='x=y')
plt.xlabel('Actual $I_{rush}$ (mA)', fontsize=30)
plt.ylabel('Predicted $I_{rush}$ (mA)', fontsize=30)
#plt.title('Max Inrush I Comparison', fontsize=30)
scale_x = 1e-3
scale_y = 1e-3
plt.xlim(llimit,mlimit)
plt.ylim(llimit,mlimit)
ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
plt.gca().xaxis.set_major_formatter(ticks_x)
ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
plt.gca().yaxis.set_major_formatter(ticks_y)
plt.legend(fontsize=30)
plt.show()
