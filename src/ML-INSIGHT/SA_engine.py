import json, csv, math, random, time
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error
import matplotlib.pyplot as plt
import xgboost as xgb

def get_cascade_inrush(test, stages):
    global model
    global sc
    global sc_y

    Y_predict = np.array([])
    for i in range(stages-2):
        data1 = [test[i],test[i+1],test[i+2],test[stages],test[stages+2],test[stages+3+i],test[stages+4+i],test[stages+5+i],test[stages*2+3]]
        data1.insert(9,0)
        data1[3] = i+3
        if Y_predict.size != 0:
            data1[9] = Y_predict[0,0]
        Xtest = sc.transform(np.array(data1).reshape(1,-1))
        Y_predict = sc_y.inverse_transform(model.predict(Xtest).reshape(1, -1))
    return Y_predict[0][0]

def get_stats(pattern):
    temp = []
    temp.append(np.min(pattern))
    temp.append(np.max(pattern))
    temp.append(np.argmax(pattern)+1)
    temp.append(np.argmin(pattern)+1)
    temp.append(pattern[-1])
    return temp

def wl_predict(test, stages):
    global wl_model
    global wl_sc
    global wl_sc_y
    pattern = test[:stages]
    slews = test[stages+3:stages*2+3]
    stats = get_stats(pattern)
    slew_stats = get_stats(slews)
    specs = test[[stages, stages+1, stages+2, stages*2+3]]
    final = np.concatenate((stats,slew_stats,specs))
    X_test1 = wl_sc.transform(final.reshape(1, -1))
    Y_predict = wl_sc_y.inverse_transform(wl_model.predict(X_test1).reshape(1, -1))
    return Y_predict[0][0]

def calc_cost(data,sw_per_stage):
    global model
    global sc
    global fpw
    global Ilimit
    global Wllimit

    test = []
    test.extend(sw_per_stage)
    test.append(len(sw_per_stage))
    test.append(sum(sw_per_stage))
    test.append(data['load_cap'])
    slew = [(1.4+0.8498*(x-1))*0.001 for x in sw_per_stage]
    test.extend(slew)
    test.append(data['operating_leakage_current'])
    test = np.array(test)
    irush_pred = get_cascade_inrush(test, len(sw_per_stage))
    wl_pred = wl_predict(test, len(sw_per_stage))

    fpw.write(",".join([str(x) for x in test]))
    fpw.write(","+str(irush_pred)+","+str(wl_pred))
    w1 = 1
    w2 = 1
    w3 = 100
    w4 = 100

    if wl_pred<0:
        cost_sum = -1
    else:
        cost_sum = irush_pred*w1 + wl_pred*w2 + max(0, (irush_pred - Ilimit))*w3 + max(0, (wl_pred - Wllimit))*w4
    fpw.write(","+str(cost_sum))
    return cost_sum, irush_pred, wl_pred

def move_switch(sol):
    stages = len(sol)
    incr_stage = random.randint(0,stages-1)
    decr_stage = random.randint(0,stages-1)
    while sol[decr_stage] <= 1:
        decr_stage = random.randint(0,stages-1)
    sol[incr_stage] += 1
    sol[decr_stage] -= 1
    return sol

def merge_stage(sol):
    stages = len(sol)
    stage1 = random.randint(0,stages-1)
    stage2 = random.randint(0,stages-1)
    while(stage1 == stage2):
        stage2 = random.randint(0,stages-1)
    sol[stage1] += sol[stage2]
    sol.pop(stage2)
    return sol

def split_stage(sol):
    stages = len(sol)
    stage = random.randint(0,stages-1)
    while sol[stage]<10:
        stage = random.randint(0,stages-1)
    half = sol[stage]//2
    sol.insert(stage,half)    
    sol[stage+1] -= half
    return sol    

def move(sol):
    global max_stage
    if len(sol) <= 3:
        mtype = random.randint(0,1)
    elif len(sol) >= max_stage:
        mtype = random.choice([0, 2])
    else:
        mtype = random.randint(0,2)
    if (mtype == 0):
        sol = move_switch(sol)
    elif (mtype == 1):
        sol = split_stage(sol)
    elif (mtype == 2):
        sol = merge_stage(sol)
    return sol

k = 1.3806E-16
def isAccept(cost, T):
    global k
    if cost < 0:
        return True
    else:
        temp = -1*cost/(k*T)
        boltz = math.exp(temp)
        r = random.random()
        if r < boltz:
            return True
        else:
            return False

def get_model(cap, ml_type="xgb"):
    model = xgb.XGBRegressor()
    wl_model = xgb.XGBRegressor()

    sc = StandardScaler()
    wl_sc = StandardScaler()
    sc_y = StandardScaler()
    wl_sc_y = StandardScaler()
    
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

    # Model Min Max MinStage MaxStage StageN SlewN stages switches xgb_multistage_modelWl_StageNMaxNt.json
    Mean3 = [2.37797094e+01,4.03038754e+02,3.19473662e+00,2.06472403e+00,9.84455241e+01,2.07581971e-02,3.43052533e-01,3.19473662e+00,2.06472403e+00,8.42092063e-02,5.48808272e+00,5.50439054e+02,9.45354765e-12,1.50000000e-05]
    Scale3 = [2.95635931e+01,3.15251453e+02,1.82446330e+00,1.21446682e+00,1.96758908e+02,2.51231414e-02,2.67900685e-01,1.82446330e+00,1.21446682e+00,1.67205720e-01,1.65908911e+00,3.44697376e+02,6.84716814e-12,1.00000000e+00]
    YMean3 = [1.76569098]
    YScale3 = [0.60064835]

    if cap > 15e-12:
        model.load_model("../../models/xgb_inrush_model15_gt10m.json")
        sc.mean_ = np.array(Mean2)
        sc.scale_ = np.array(Scale2)
        sc_y.mean_ = np.array(YMean2)
        sc_y.scale_ = np.array(YScale2)
    else:
        model.load_model("../../models/xgb_inrush_model15_lt10m.json")
        sc.mean_ = np.array(Mean)
        sc.scale_ = np.array(Scale)
        sc_y.mean_ = np.array(YMean)
        sc_y.scale_ = np.array(YScale)


    wl_model.load_model("../../models/xgb_wakeup_stat_model.json")    
    wl_sc.mean_ = np.array(Mean3)
    wl_sc.scale_ = np.array(Scale3)
    wl_sc_y.mean_ = np.array(YMean3)
    wl_sc_y.scale_ = np.array(YScale3)

    return model, sc, sc_y, wl_model, wl_sc, wl_sc_y

def SA_engine(curSol):
    global fpw
    cost_list = []
    temp_list = []

    T0 = 1E20
    tf = 1E-5
    slope = 0.999
    T = T0
    mov_per_step = 10
    cost_cursol, _, _ = calc_cost(data,curSol[:])
    print("Initial cost: ",cost_cursol)
    fpw.write("\n")
    while(T>tf):
        for i in range(mov_per_step):
            next_sol = move(curSol[:])
            #print(curSol)
            cost_nextsol, _, _ = calc_cost(data,next_sol[:])
            if (cost_nextsol == -1):
                t = False
            else:
                cost = cost_nextsol - cost_cursol
                t = isAccept(cost,T)
            fpw.write(","+str(t)+"\n")
            if t:
                curSol = next_sol
                cost_cursol = cost_nextsol
        cost_list.append(cost_cursol)
        T = slope*T
        temp_list.append(T)
    return curSol, temp_list, cost_list


start_time = time.time()
config = "/home/vgopal18/python/config.json"

designs = {
"Ariane": [1000, 28.653, 26.1],
"AES256": [930, 22, 23.71],
"Mempool": [700, 19.49, 17.14],
"Ibex": [500, 2.783, 3.04],
"JPEG": [440, 6.13, 6.57],
"AES128": [410, 2.048, 1.89],
"Ethmac": [380, 9.148, 11.74],
"Raven_SHA": [320, 3.69, 3.71],
"Mock-array": [220, 10.198, 0.47],
"Amber": [200, 1, 0.68]
}

design = "AES128"

switches = designs[design][0]

init_stage = 3
max_stage = 8

cap = designs[design][1]*1e-12
ileak = designs[design][2]*1e-6
Ilimit = 20
Wllimit = 20
print("Starting for design ",design,"\nNt: ",switches,"\nCap: ",cap,"\nIleak: ",ileak)

fp = open(config)
data = json.load(fp)
data['load_cap'] = cap 
data['operating_leakage_current'] = ileak

fpw = open("SA_log_"+design+".csv",'w')

#ML
model, sc, sc_y, wl_model, wl_sc, wl_sc_y = get_model(cap)


rem = switches - init_stage - 98
initial_sol = [rem]+[100]+[1]*(init_stage-2)

print("Initial Solution : ")
print(initial_sol)

curSol = initial_sol
finalSol, t_list, c_list = SA_engine(curSol[:])
print("Final Solution:",finalSol)
print("Final cost: ",calc_cost(data, finalSol[:]))
x = range(len(c_list))
fig, ax1 = plt.subplots()

print("Min cost seen: ",min(c_list))
print("Max cost seen: ",max(c_list))

ax1.plot(x, c_list, label="Cost", color="blue")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Cost")

ax2 = ax1.twinx()
ax2.plot(x, t_list, label="Temperature", color="red")
ax2.set_ylabel("Temperature")

# Labels and legend
ax1.legend()

# Show plot
plt.savefig("SA_"+design+".png")

fpw.close()
end_time = time.time()
print("Time taken: ",str(end_time-start_time))




