import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xgboost as xgb
import time
import joblib

#data = pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw1_2000.csv", on_bad_lines='warn')
data = []
data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_ariane.csv", on_bad_lines='warn'))
data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_aes256.csv", on_bad_lines='warn'))
data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_mempool.csv", on_bad_lines='warn'))
#data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_ibex.csv", on_bad_lines='warn'))
#data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_jpeg.csv", on_bad_lines='warn'))
#data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_a128.csv", on_bad_lines='warn'))
#data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_ethmac.csv", on_bad_lines='warn'))
#data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_ravensha.csv", on_bad_lines='warn'))
#data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_marr.csv", on_bad_lines='warn'))
#data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_amber.csv", on_bad_lines='warn'))

designs = ["ariane", "aes256", "mempool"]
#designs = ["ibex", "jpeg", "aes128", "ethmac", "ravensha", "marr", "amber"]
#designs = ["ariane", "aes256", "mempool", "ibex", "jpeg", "aes128", "ethmac", "ravensha", "marr", "amber"]
data_sampled = []

for i in range(len(data)):
    data[i] = data[i].drop('EndTime', axis=1)
    data_sampled.append(data[i].sample(n=2000, random_state=42).reset_index(drop=True))
    #data_sampled.append(data[i])

sc = StandardScaler()
sc_y = StandardScaler()
sc2 = StandardScaler()
sc_y2 = StandardScaler()

# Model 10 lt 10m
#Mean = [7.59719499e+01,7.76572080e+01,7.79495141e+01,5.21264219e+00,5.16064530e-12,6.51111631e-02,6.65432954e-02,6.67916971e-02,1.50000000e-05,3.78803207e+00,1.70197747e+00]
#Scale = [1.55627830e+02,1.58587637e+02,1.59036632e+02,1.68055276e+00,3.33857946e-12,1.32252530e-01,1.34767774e-01,1.35149330e-01,1.00000000e+00,2.98659675e+00,2.44471987e+00]
#YMean = [4.7305353,1.59244441]
#YScale = [2.64562442,0.63050819]

# Model 12 gt 10m
Mean = [1.43227570e+02,1.38864087e+02,1.27665023e+02,5.92283191e+00,1.79051427e-11,1.22264989e-01,1.18556901e-01,1.09039936e-01,1.50000000e-05,1.06473904e+01]
Scale = [2.46370632e+02,2.42663157e+02,2.31416593e+02,1.50047250e+00,3.97149963e-12,2.09365763e-01,2.06215151e-01,1.96657821e-01,1.00000000e+00,4.37663137e+00]
YMean= [12.83201396]
YScale= [2.0102293]

# Model 12 lt 10m
Mean2 = [5.71653903e+01,5.83387965e+01,5.81635055e+01,4.69863795e+00,8.28617431e-12,4.91293487e-02,5.01265093e-02,4.99775470e-02,1.50000000e-05,3.02252474e+00]
Scale2 = [1.25479598e+02,1.28338975e+02,1.28910594e+02,1.58241846e+00,5.75619539e-12,1.06632563e-01,1.09062461e-01,1.09548223e-01,1.00000000e+00,2.84168490e+00]
YMean2 = [4.25612982]
YScale2 = [2.41709109]

# Model 15 lt 10m
#Mean = [5.49349650e+01,5.59503195e+01,5.53295028e+01,4.88232706e+00,9.11311103e-12,4.72339332e-02,4.80967815e-02,4.75692115e-02,1.50000000e-05,3.35345446e+00]
#Scale = [1.17394679e+02,1.20103305e+02,1.20583432e+02,1.64432471e+00,6.20626859e-12,9.97619982e-02,1.02063789e-01,1.02471800e-01,1.00000000e+00,2.98884340e+00]
#YMean = [4.58158578]
#YScale = [2.59561707]

# Model 15 gt 10m
Mean2 = [1.41592828e+02,1.37474657e+02,1.26083356e+02,6.01468120e+00,1.79515644e-11,1.20875785e-01,1.17376164e-01,1.07695836e-01,1.50000000e-05,1.07513735e+01]
Scale2 = [2.45765547e+02,2.42355649e+02,2.31026134e+02,1.50201724e+00,3.99165885e-12,2.08851562e-01,2.05953830e-01,1.96326009e-01,1.00000000e+00,4.32359322e+00]
YMean2 = [12.85552629]
YScale2 = [2.04070111]

sc.mean_ = np.array(Mean)
sc.scale_ = np.array(Scale)
sc_y.mean_ = np.array(YMean)
sc_y.scale_ = np.array(YScale)

sc2.mean_ = np.array(Mean2)
sc2.scale_ = np.array(Scale2)
sc_y2.mean_ = np.array(YMean2)
sc_y2.scale_ = np.array(YScale2)

#for x in X_test:
#    X_test_scaled.append(sc.transform(x))

#X_test11 = sc.transform(X_test1)
#X_test22 = sc.transform(X_test2)
#X_test33 = sc.transform(X_test3)
#X_test44 = sc.transform(X_test4)
#X_test55 = sc.transform(X_test5)
#X_test66 = sc.transform(X_test6)

ml_type = "rf"

if (ml_type == "xgb"):
    model = xgb.XGBRegressor()
    model.load_model("/home/vgopal18/python/xgb_v2_multistage_model15_lt10m.json")
    model2 = xgb.XGBRegressor()
    model2.load_model("/home/vgopal18/python/xgb_v2_multistage_model15_gt10m.json")
elif (ml_type == "mlp"):
    model = Sequential()
    model.add(Dense(64, input_dim=10, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(128, activation='relu'))
#    model.add(Dense(128, activation='relu'))
#    model.add(Dense(128, activation='relu'))
#    model.add(Dense(128, activation='relu'))
#    model.add(Dense(128, activation='relu'))
#    model.add(Dense(128, activation='relu'))
#    model.add(Dense(128, activation='relu'))
#    model.add(Dense(128, activation='relu'))
#    model.add(Dense(128, activation='relu'))
#    model.add(Dense(128, activation='relu'))
#    model.add(Dense(128, activation='relu'))
#    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    model.load_weights('/home/vgopal18/python/ML_data/mlp_l14_2.h5')
    model2 = model
elif (ml_type == "rf"):
    model2 = joblib.load("random_forest_final_model.joblib")

def cascade_predict(data, sc, sc_y, model):
    Y_predict = np.array([])
    size = 10

    print(data.columns)
    stages = data['Stages'].values[0]
    print("Predicting for stages:",stages)

    for i in range(stages-2):
        data1 = data[[data.columns[i],data.columns[i+1],data.columns[i+2],data.columns[stages],data.columns[stages+2],data.columns[stages+3+i],data.columns[stages+4+i],data.columns[stages+5+i],data.columns[stages*2+3],data.columns[stages*2+4]]]
        data1.insert(9,'PrevI',0)
        #data1.insert(10,'PrevWl',0)
        data1['Stages'] = i+3
        if Y_predict.size == 0:
            data1['PrevI'] = 0
            #data1['PrevWl'] = 0
        else:
            data1['PrevI'] = Y_predict
            #data1['PrevWl'] = Y_predict[:,1]
        X_test = data1.iloc[:,:size].values
        print(X_test[0,:])
        Y_test = data1.iloc[:,size:].values
        print(data1.columns)   
        print(X_test.shape,Y_test.shape)
        print(X_test[1])
    
        X_test1 = sc.transform(X_test)
        print(X_test[0])
        #X_test1 = X_test/Max
        print(X_test1[0])
        #Y_predict = sc_y.inverse_transform(xgb_reg.predict(X_test1))*[1e-3,1e-9]
        Y_predict = sc_y.inverse_transform(model.predict(X_test1).reshape(-1,1))
        #Y_predict = model.predict(X_test1)
        print(Y_predict[0])
        #Y_predict = Y_predict*YMax
        print("After stage:",i+3)
        print("Ans:",Y_predict.shape)
    return Y_predict

def evaluate(Y_test, Y_predict):
    error_list = []

    for i in range(Y_test.shape[1]):
        y_true = Y_test[:, i]
        y_pred = Y_predict[:, i]
        #mse = mean_squared_error(y_true, Y_pred)
        mae = mean_absolute_error(y_true,y_pred)
        #rmse = np.sqrt(mse)
        max_err = max_error(y_true, y_pred)
        errors = y_pred - y_true
        abs_errors = abs(errors)
        error_percentages = (abs_errors / y_true) * 100
        mean_percent = sum(error_percentages)/len(error_percentages)
        temp = []
        temp.append(mae)
        temp.append(max_err)
        temp.append(mean_percent)        
        temp.append(max(error_percentages))
        error_list.append(temp)
    return error_list

run_times = []
Y_predict = []
Y_actual = []
for x in data_sampled:
    print(x)
    cap = x['Cap'][0]
    if cap > 15e-12:
        st = time.time()
        Y_predict.append(cascade_predict(x, sc2, sc_y2, model2))
        et = time.time()
        run_times.append(et-st)
    else:
        st = time.time()        
        Y_predict.append(cascade_predict(x, sc, sc_y, model))
        et = time.time()
        run_times.append(et-st)
        
    Y_temp = x[['MaxI']].to_numpy()*[1e3]
    Y_temp = Y_temp.reshape(-1, 1)
    Y_actual.append(Y_temp)

fp = open('ML_metrics_rf_Irush_gt.csv','w')
fp.write('Design,MAE,Max Error Value,Mean Error %,Max Error %\n')
for i in range(len(data)):
    errors = evaluate(Y_actual[i], Y_predict[i])
    lst = []
    lst.append(designs[i])
    lst.extend([str(x) for x in errors[0]])
    str1 = ",".join(lst)
    fp.write(str1+'\n')
fp.close()
run = "\n".join([str(x) for x in run_times])
print(run)
