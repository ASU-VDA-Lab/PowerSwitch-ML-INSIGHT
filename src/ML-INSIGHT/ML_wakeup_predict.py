import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xgboost as xgb
from scipy.stats import skew, kurtosis
import time

data = []
data.append(pd.read_csv("../../data/testcases/results_ariane.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases/results_aes256.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases/results_mempool.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases/results_ibex.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases/results_jpeg.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases/results_a128.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases/results_ethmac.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases/results_ravensha.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases/results_marr.csv", on_bad_lines='warn'))
data.append(pd.read_csv("../../data/testcases/results_amber.csv", on_bad_lines='warn'))

designs = ["ariane", "aes256", "mempool", "ibex", "jpeg", "aes128", "ethmac", "ravensha", "marr", "amber"]
data_sampled = []

for i in range(len(data)):
    data_sampled.append(data[i].sample(n=2000, random_state=42))

sc = StandardScaler()
sc_y = StandardScaler()


# Model Min Max MinStage MaxStage StageN SlewN stages switches xgb_multistage_modelWl_StageNMaxNt.json
Mean = [1.83866814e+01,3.39530975e+02,3.18403649e+00,2.03993284e+00,8.47878294e+01,1.61752018e-02,2.89083622e-01,3.18403649e+00,2.03993284e+00,7.26028975e-02,5.44134170e+00,4.71448511e+02,9.92909710e-12,1.50000000e-05]
Scale = [2.68575789e+01,2.96066171e+02,1.82104052e+00,1.17283903e+00,1.74650874e+02,2.28235705e-02,2.51597032e-01,1.82104052e+00,1.17283903e+00,1.48418313e-01,1.67873003e+00,3.33488289e+02,6.75369480e-12,1.00000000e+00]
YMean = [2.0781746]
YScale = [0.85464984]

sc.mean_ = np.array(Mean)
sc.scale_ = np.array(Scale)

sc_y.mean_ = np.array(YMean)
sc_y.scale_ = np.array(YScale)


ml_type = "xgb"

if (ml_type == "xgb"):
    model = xgb.XGBRegressor()
    model.load_model("../../models/xgb_wakeup_stat_model.json")

def get_stats(pattern):
    temp = []
    temp.append(np.min(pattern))
    temp.append(np.max(pattern))
    temp.append(np.argmax(pattern)+1)
    temp.append(np.argmin(pattern)+1)
    temp.append(pattern[-1])
    return temp

def wl_predict(data, sc, model):
    Y_predict = np.array([])
    size = 14
    data_out = data[["Stages","Switches", "Cap","Ileak"]]
    data_out.reset_index(drop=True, inplace=True)
    stages = data['Stages'].values[0]
    patterns = data.iloc[:,:stages].to_numpy()
    slews = data.iloc[:,stages+3:stages*2+3]
    stats = np.apply_along_axis(get_stats, axis=1, arr=patterns)
    slew_stats = np.apply_along_axis(get_stats, axis=1, arr=slews)
    stats_df = pd.DataFrame(stats,columns=["Min","Max","MaxStage","MinStage", "StageN"])
    slew_stats_df = pd.DataFrame(slew_stats,columns=["slew_Min","slew_Max","MaxSlewStage", "MinSlewStage", "SlewN"])
    data_final = pd.concat([stats_df,slew_stats_df,data_out],axis=1)
    
    X_test = data_final.iloc[:,:size].values
    
    X_test1 = sc.transform(X_test)
    Y_predict = sc_y.inverse_transform(model.predict(X_test1).reshape(-1, 1))
    return Y_predict

def evaluate(Y_test, Y_predict):
    error_list = []

    for i in range(1):
        y_true = Y_test
        y_pred = Y_predict
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

Y_predict = []
Y_actual = []
run_times = []
for x in data_sampled:
    st = time.time()
    Y_predict.append(wl_predict(x, sc, model))
    et = time.time()
    run_times.append(et-st)
    Y_actual.append(x[['EndTime']].to_numpy()*[1e9])

fp = open('ML_metrics_xgb_wakeup.csv','w')
fp.write('Design,MAE,Max Error Value,Mean Error %,Max Error %\n')
error_sum = 0
error_max = 0

for i in range(len(data)):
    errors = evaluate(Y_actual[i], Y_predict[i])
    lst = []
    lst.append(designs[i])
    lst.extend([str(x) for x in errors[0]])
    str1 = ",".join(lst)
    fp.write(str1+'\n')
    error_sum = error_sum + errors[0][2]
    error_max = errors[0][3] if errors[0][3] > error_max else error_max
    
fp.close()
run = "\n".join([str(x) for x in run_times])
print("\nFinal Mean error percent: ",error_sum/len(data))
print("Final Max error percent: ", error_max)

