import json, csv, math, random, time
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import xgboost as xgb

def get_cascade_inrush(test, stages):
    global model
    global sc
    global sc_y
    Y_predict = np.array([])
    for i in range(stages-2):
        data1 = [test[i],test[i+1],test[i+2],test[stages],test[stages+1],test[stages+2+i],test[stages+3+i],test[stages+4+i],test[stages*2+2]]
        data1.insert(9,0)
        data1.insert(10,0)
        data1[3] = i+3
        if Y_predict.size != 0:
            data1[9] = Y_predict[0,0]
            data1[10] = Y_predict[0,1]
        data1 = sc.transform(np.array(data1).reshape(1,-1))
        Y_predict = sc_y.inverse_transform(model.predict(data1))
        #print("At stage ",i+2,Y_predict)
    return Y_predict
    
def calc_cost(data,sw_per_stage):
    global model
    global sc
    global fpw
    global Ilimit
    global Wllimit

    test = []
    test.extend(sw_per_stage)
    test.append(len(sw_per_stage))
    #test.append(sum(sw_per_stage))
    test.append(data['load_cap'])
    slew = [(1.4+0.8498*(x-1))*0.001 for x in sw_per_stage]
    test.extend(slew)
    test.append(data['operating_leakage_current'])
    test = np.array(test)
    pred = get_cascade_inrush(test, len(sw_per_stage))
    fpw.write(",".join([str(x) for x in test]))
    fpw.write(","+str(pred[0,0])+","+str(pred[0,1]))
    w1 = 1
    w2 = 1
    w3 = 1000
    w4 = 1000

    if pred[0,1]<1:
        cost_sum = -1
    else:
        cost_sum = pred[0,0]*w1 + pred[0,1]*w2 + max(0, (pred[0,0] - Ilimit))*w3 + max(0, (pred[0,1] - Wllimit))*w4
    return cost_sum, pred[0,0], pred[0,1]

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
        #print(cost, boltz, T)
        if r < boltz:
            return True
        else:
            return False

def get_model(ml_type="xgb"):
    model = xgb.XGBRegressor()
    model.load_model("/home/vgopal18/python/xgb_v2_clean_multistage_model7.json")
    sc = StandardScaler()
    sc_y = StandardScaler()
    Mean= [2.85167979e+01, 2.84284070e+01, 2.81884693e+01, 3.96423129e+00, 9.35657228e-12, 2.47837748e-02, 2.47086603e-02, 2.45047612e-02, 2.24570787e-04, 1.74113518e+00, 3.49024562e+00]
    Scale= [2.82105944e+01, 2.81359983e+01, 2.80467193e+01, 9.85303160e-01, 6.43102742e-12, 2.39733631e-02, 2.39099714e-02, 2.38341020e-02, 4.02798472e-04, 1.67476341e+00, 6.27315376e+00]
    YMean= [3.13215204, 4.65101527]
    YScale= [1.18155887, 4.59859609]
    sc.mean_ = np.array(Mean)
    sc.scale_ = np.array(Scale)
    sc_y.mean_ = np.array(YMean)
    sc_y.scale_ = np.array(YScale)
    return model, sc, sc_y

def SA_engine(curSol):
    global fpw
    cost_list = []
    temp_list = []

    T0 = 1E20
    tf = 0.01
    slope = 0.99
    T = T0
    mov_per_step = 5
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
"Ariane": [180, 28.653, 25000],
"AES256": [175, 22, 23.71],
"Mempool": [160, 19.49, 3700],
"Ibex": [145, 2.783, 3.04],
"JPEG": [140, 6.13, 6.57],
"AES128": [140, 2.048, 1.89],
"Ethmac": [140, 9.148, 11.74],
"Raven_SHA": [135, 3.69, 3.71],
"Mock-array": [120, 10.198, 0.47],
"Amber": [110, 1, 0.68]
}
design = "Amber"

switches = designs[design][0]

init_stage = 3
max_stage = 8

cap = designs[design][1]*1e-12
ileak = designs[design][2]*1e-6
Ilimit = 8
Wllimit = 8
print("Starting for design ",design,"\nNt: ",switches,"\nCap: ",cap,"\nIleak: ",ileak)

fp = open(config)
data = json.load(fp)
data['load_cap'] = cap 
data['operating_leakage_current'] = ileak

fpw = open("SA_log2_"+design+".csv",'w')
#ML
model, sc, sc_y = get_model()


sw_avg = switches//init_stage
rem = switches%init_stage
initial_sol = [sw_avg]*init_stage
initial_sol[init_stage-1] += rem
print("Initial Solution : ")
print(initial_sol)

curSol = initial_sol
finalSol, t_list, c_list = SA_engine(curSol[:])
print("Final Solution:",finalSol)
print("Final cost: ",calc_cost(data, finalSol[:]))

print("Min cost seen: ",min(c_list))
print("Max cost seen: ",max(c_list))

x = range(len(c_list))
fig, ax1 = plt.subplots()

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




