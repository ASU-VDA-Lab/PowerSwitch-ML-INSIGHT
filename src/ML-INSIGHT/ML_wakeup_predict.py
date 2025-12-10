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
from scipy.stats import skew, kurtosis
import time
import joblib

#data = pd.read_csv("/home/vgopal18/python/ML_data/spice_data/results_sw1_2000.csv", on_bad_lines='warn')
data = []
data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_ariane.csv", on_bad_lines='warn'))
data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_aes256.csv", on_bad_lines='warn'))
data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_mempool.csv", on_bad_lines='warn'))
data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_ibex.csv", on_bad_lines='warn'))
data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_jpeg.csv", on_bad_lines='warn'))
data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_a128.csv", on_bad_lines='warn'))
data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_ethmac.csv", on_bad_lines='warn'))
data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_ravensha.csv", on_bad_lines='warn'))
data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_marr.csv", on_bad_lines='warn'))
data.append(pd.read_csv("/home/vgopal18/python/ML_data/spice_data/wave_data/test_data/results_amber.csv", on_bad_lines='warn'))

#designs = ["ibex", "jpeg", "aes128", "ethmac", "ravensha", "marr", "amber"]
designs = ["ariane", "aes256", "mempool", "ibex", "jpeg", "aes128", "ethmac", "ravensha", "marr", "amber"]
data_sampled = []

for i in range(len(data)):
    data_sampled.append(data[i].sample(n=2000, random_state=42))
    #data_sampled.append(data[i])

sc = StandardScaler()
sc_y = StandardScaler()

# Model stats

#Mean = [2.91723017e+01,2.37326476e+01,6.09993688e+00,6.45160483e+01,4.08535043e-01,-1.21497763e+00,4.16388271e-01,1.34972143e+00,2.42196145e+00,2.28270077e+00,2.53408220e-02,2.01680039e-02,5.73392636e-03,5.53759378e-02,4.08535043e-01,-1.21497763e+00,4.05905201e-01,1.38418434e+00,2.42196145e+00,2.28270077e+00,3.87196466e+00,1.12169796e+02,8.46595446e-12,1.53439093e-05]
#Scale = [8.84314422e+00,1.19836301e+01,6.77526681e+00,2.71922091e+01,5.59825218e-01,5.12285980e-01,1.47927761e-01,3.98325486e-01,1.20197969e+00,1.12664089e+00,7.51490396e-03,1.01836888e-02,5.75762173e-03,2.31079393e-02,5.59825218e-01,5.12285980e-01,1.44224050e-01,3.83664032e-01,1.20197969e+00,1.12664089e+00,9.27307701e-01,3.95322165e+01,6.04781678e-12,2.24507589e-06]
#YMean = [3.10165829, 4.81525278]
#YScale = [1.1843393, 5.96079519]

#Mean = [9.63905961e+01,1.41991798e+02,3.57706475e+01,4.34626838e+02,1.79890006e+00,1.64851636e+00,4.75308744e-01,1.65329678e+00,3.70860732e+00,2.12179230e+00,8.24629286e-02,1.20664630e-01,3.09480963e-02,3.69896087e-01,1.79890006e+00,1.64851636e+00,4.70908062e-01,1.67157210e+00,3.70860732e+00,2.12179230e+00,6.68461509e+00,6.29942943e+02,5.50123660e-12,1.50000000e-05]
#Scale = [4.76012964e+01,1.09678166e+02,2.81662548e+01,3.00784036e+02,6.37852084e-01,1.31712928e+00,2.58960664e-01,7.83199335e-01,2.02707966e+00,1.36772118e+00,4.04515816e-02,9.32045054e-02,2.39356833e-02,2.55606274e-01,6.37852084e-01,1.31712928e+00,2.57373173e-01,7.71965593e-01,2.02707966e+00,1.36772118e+00,1.03371795e+00,2.97648633e+02,2.87231976e-12,1.00000000e+00]
#YMean = [5.37850562,1.48912059]
#YScale = [2.5731844,1.63281495]

# Model Min Max MinStage MaxStage StageN SlewN stages switches xgb_multistage_modelWl_StageNMaxNt.json
Mean = [2.37797094e+01,4.03038754e+02,3.19473662e+00,2.06472403e+00,9.84455241e+01,2.07581971e-02,3.43052533e-01,3.19473662e+00,2.06472403e+00,8.42092063e-02,5.48808272e+00,5.50439054e+02,9.45354765e-12,1.50000000e-05]
Scale = [2.95635931e+01,3.15251453e+02,1.82446330e+00,1.21446682e+00,1.96758908e+02,2.51231414e-02,2.67900685e-01,1.82446330e+00,1.21446682e+00,1.67205720e-01,1.65908911e+00,3.44697376e+02,6.84716814e-12,1.00000000e+00]
YMean = [1.76569098]
YScale = [0.60064835]

# Model Min Max MinStage MaxStage StageN SlewN stages switches xgb_multistage_modelWl_StageNMaxNt_2.json

#Mean = [1.83866814e+01,3.39530975e+02,3.18403649e+00,2.03993284e+00,8.47878294e+01,1.61752018e-02,2.89083622e-01,3.18403649e+00,2.03993284e+00,7.26028975e-02,5.44134170e+00,4.71448511e+02,9.92909710e-12,1.50000000e-05]
#Mean = [1.83866814e+01,3.39530975e+02,3.18403649e+00,2.03993284e+00,1.61752018e-02,2.89083622e-01,3.18403649e+00,2.03993284e+00,5.44134170e+00,4.71448511e+02,9.92909710e-12,1.50000000e-05]
#Scale = [2.68575789e+01,2.96066171e+02,1.82104052e+00,1.17283903e+00,1.74650874e+02,2.28235705e-02,2.51597032e-01,1.82104052e+00,1.17283903e+00,1.48418313e-01,1.67873003e+00,3.33488289e+02,6.75369480e-12,1.00000000e+00]
#Scale = [2.68575789e+01,2.96066171e+02,1.82104052e+00,1.17283903e+00,2.28235705e-02,2.51597032e-01,1.82104052e+00,1.17283903e+00,1.67873003e+00,3.33488289e+02,6.75369480e-12,1.00000000e+00]
#YMean = [2.0781746]
#YScale = [0.85464984]
#YMean = [4.81525278]
#YScale = [5.96079519]
#Irush
#YMean = [3.10165829]
#YScale = [1.1843393]

# Model Cascade Wl only
#Mean = [2.92994986e+01,2.91926136e+01,2.89880384e+01,3.87170624e+00,8.45550047e-12,2.54489139e-02,2.53580831e-02,2.51842350e-02,1.53414239e-05,1.66596225e+00]
#Scale = [2.80671388e+01,2.79680539e+01,2.78811262e+01,9.28810982e-01,6.04185799e-12,2.38514545e-02,2.37672522e-02,2.36933811e-02,2.23713836e-06,1.66888195e+00]
##YMean = [3.10494534,4.78291605]
#YMean = [3.10494534]
##YScale = [1.18269804,5.8457812]
#YScale = [1.18269804]

sc.mean_ = np.array(Mean)
sc.scale_ = np.array(Scale)

sc_y.mean_ = np.array(YMean)
sc_y.scale_ = np.array(YScale)


ml_type = "mlp"

if (ml_type == "xgb"):
    model = xgb.XGBRegressor()
    #model.load_model("/home/vgopal18/python/xgb_v2_clean_multistage_model_stat_comb.json")
    #model.load_model("/home/vgopal18/python/xgb_multistage_modelWl_StageNMaxNt_2.json")
    model.load_model("/home/vgopal18/python/xgb_multistage_modelWl_FE_noN.json")
elif (ml_type == "mlp"):
    model = Sequential()
    model.add(Dense(64, input_dim=14, kernel_initializer='he_uniform', activation='relu'))
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
    model.load_weights('/home/vgopal18/python/ML_data/mlp_wl_l14_1.h5')
elif (ml_type == "rf"):
    model = joblib.load("random_forest_wl_final_model.joblib")

def get_stats(pattern):
    temp = []
    #temp.append(np.mean(pattern))
    #temp.append(np.std(pattern))
    temp.append(np.min(pattern))
    temp.append(np.max(pattern))
    #temp.append(skew(pattern))
    #temp.append(kurtosis(pattern))
    #temp.append(np.sum(np.abs(pattern[:, None] - pattern[None, :])) / (2 * len(pattern) * np.sum(pattern)))
    #temp.append(-np.sum((pattern / np.sum(pattern)) * np.log2(pattern / np.sum(pattern))))
    temp.append(np.argmax(pattern)+1)
    temp.append(np.argmin(pattern)+1)
    temp.append(pattern[-1])
    return temp

def wl_predict(data, sc, model):
    Y_predict = np.array([])
    #size = 14
    size = 14
    #data_out = data[["Stages","Switches","Cap","Ileak"]]
    data_out = data[["Stages","Switches", "Cap","Ileak"]]
    data_out.reset_index(drop=True, inplace=True)
    print("Data: ",data_out)
    stages = data['Stages'].values[0]
    patterns = data.iloc[:,:stages].to_numpy()
    slews = data.iloc[:,stages+3:stages*2+3]
    print("Patterns: ",patterns.shape)
    print("Slews: ",slews.shape)
    stats = np.apply_along_axis(get_stats, axis=1, arr=patterns)
    slew_stats = np.apply_along_axis(get_stats, axis=1, arr=slews)
    #stats_df = pd.DataFrame(stats,columns=["StageN"])
    stats_df = pd.DataFrame(stats,columns=["Min","Max","MaxStage","MinStage", "StageN"])
    #slew_stats_df = pd.DataFrame(slew_stats,columns=["SlewN"])
    slew_stats_df = pd.DataFrame(slew_stats,columns=["slew_Min","slew_Max","MaxSlewStage", "MinSlewStage", "SlewN"])
    print(stats_df.shape, slew_stats_df.shape,data_out.shape)
    print("Stats: ",stats_df)
    print("Stats: ",slew_stats_df)
    data_final = pd.concat([stats_df,slew_stats_df,data_out],axis=1)
    print(data_final.shape)
    print(data_final.columns)
    
    X_test = data_final.iloc[:,:size].values
    print(X_test[0,:])
    
    X_test1 = sc.transform(X_test)
    Y_predict = sc_y.inverse_transform(model.predict(X_test1).reshape(-1, 1))
    return Y_predict

def evaluate(Y_test, Y_predict):

    y_true = Y_test
    y_pred = Y_predict
    print(y_true.shape, y_pred.shape)
    #mse = mean_squared_error(y_true, Y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    #rmse = np.sqrt(mse)
    max_err = max_error(y_true, y_pred)
    errors = y_pred - y_true
    print(errors.shape)
    abs_errors = abs(errors)
    error_percentages = (abs_errors / y_true) * 100
    mean_percent = np.mean(error_percentages)
    temp = []
    temp.append(mae)
    temp.append(max_err)
    temp.append(mean_percent)        
    temp.append(np.max(error_percentages))
    return temp

def evaluate2(Y_test, Y_predict):
    error_list = []

    for i in range(1):
        y_true = Y_test
        y_pred = Y_predict
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

Y_predict = []
Y_actual = []
run_times = []
for x in data_sampled:
    st = time.time()
    Y_predict.append(wl_predict(x, sc, model))
    et = time.time()
    run_times.append(et-st)
    Y_actual.append(x[['EndTime']].to_numpy()*[1e9])

fp = open('ML_metrics_mlp_wl.csv','w')
fp.write('Design,MAE,Max Error Value,Mean Error %,Max Error %\n')
error_sum = 0
error_max = 0

for i in range(len(data)):
    errors = evaluate2(Y_actual[i], Y_predict[i])
    #range_list = []
    #range_list.append(min(Y_predict[i][:, 0]))
    #range_list.append(max(Y_predict[i][:, 0]))
    #range_list.append(min(Y_predict[i][:, 1]))
    #range_list.append(max(Y_predict[i][:, 1]))
    lst = []
    lst.append(designs[i])
    #lst.extend([str(x) for x in errors])
    lst.extend([str(x) for x in errors[0]])
    #lst.extend([str(x) for x in errors[1]])
    #lst.extend([str(x) for x in range_list])
    str1 = ",".join(lst)
    fp.write(str1+'\n')
    error_sum = error_sum + errors[0][2]
    error_max = errors[0][3] if errors[0][3] > error_max else error_max
    
fp.close()
run = "\n".join([str(x) for x in run_times])
print("\nFinal Mean error percent: ",error_sum/len(data))
print("Final Max error percent: ", error_max)

