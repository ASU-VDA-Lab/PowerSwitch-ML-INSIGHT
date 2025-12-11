import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xgboost as xgb
import time

data = []
data.append(pd.read_csv("../../data/testcases//results_ariane.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases//results_aes256.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases//results_mempool.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases//results_ibex.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases//results_jpeg.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases//results_a128.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases//results_ethmac.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases//results_ravensha.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases//results_marr.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases//results_amber.csv", on_bad_lines='warn'))

designs = ["ariane", "aes256", "mempool", "ibex", "jpeg", "aes128", "ethmac", "ravensha", "marr", "amber"]
data_sampled = []

for i in range(len(data)):
    data[i] = data[i].drop('EndTime', axis=1)
    data_sampled.append(data[i].sample(n=2000, random_state=42).reset_index(drop=True))

sc = StandardScaler()
sc_y = StandardScaler()
sc2 = StandardScaler()
sc_y2 = StandardScaler()

# Model 15 lt 10m
Mean = [5.49349650e+01,5.59503195e+01,5.53295028e+01,4.88232706e+00,9.11311103e-12,4.72339332e-02,4.80967815e-02,4.75692115e-02,1.50000000e-05,3.35345446e+00]
Scale = [1.17394679e+02,1.20103305e+02,1.20583432e+02,1.64432471e+00,6.20626859e-12,9.97619982e-02,1.02063789e-01,1.02471800e-01,1.00000000e+00,2.98884340e+00]
YMean = [4.58158578]
YScale = [2.59561707]

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

ml_type = "xgb"

if (ml_type == "xgb"):
    model = xgb.XGBRegressor()
    model.load_model("../../models/xgb_inrush_model15_lt10m.json")
    model2 = xgb.XGBRegressor()
    model2.load_model("../../models/xgb_inrush_model15_gt10m.json")

def cascade_predict(data, sc, sc_y, model):
    Y_predict = np.array([])
    size = 10

    print(data.columns)
    stages = data['Stages'].values[0]
    print("Predicting for stages:",stages)

    for i in range(stages-2):
        data1 = data[[data.columns[i],data.columns[i+1],data.columns[i+2],data.columns[stages],data.columns[stages+2],data.columns[stages+3+i],data.columns[stages+4+i],data.columns[stages+5+i],data.columns[stages*2+3],data.columns[stages*2+4]]]
        data1.insert(9,'PrevI',0)
        data1['Stages'] = i+3
        if Y_predict.size == 0:
            data1['PrevI'] = 0
        else:
            data1['PrevI'] = Y_predict
        X_test = data1.iloc[:,:size].values
        Y_test = data1.iloc[:,size:].values
    
        X_test1 = sc.transform(X_test)
        Y_predict = sc_y.inverse_transform(model.predict(X_test1).reshape(-1,1))
    return Y_predict

def evaluate(Y_test, Y_predict):
    error_list = []

    for i in range(Y_test.shape[1]):
        y_true = Y_test[:, i]
        y_pred = Y_predict[:, i]
        mae = mean_absolute_error(y_true,y_pred)
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

fp = open('ML_metrics_xgb_Irush_gt.csv','w')
fp.write('Design,MAE,Max Error Value,Mean Error %,Max Error %\n')
for i in range(len(data)):
    errors = evaluate(Y_actual[i], Y_predict[i])
    lst = []
    lst.append(designs[i])
    lst.extend([str(x) for x in errors[0]])
    str1 = ",".join(lst)
    fp.write(str1+'\n')
fp.close()
