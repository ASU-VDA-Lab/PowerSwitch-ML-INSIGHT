import json, csv, math
import matplotlib.pyplot as plt

config = "C:/Users/gvikr/OneDrive/Desktop/Inrush_automation/config.json"
fp = open(config)
data = json.load(fp)
inrush_max = 2
operating_leakage_current_per_sw = data['operating_leakage_current'] / data['trickle_switches']
inrush_data = []
current_values = []
#wakeup_latency = stages*data['delay']+data['Response']
sw_rem = data['trickle_switches']
Idsat = data['Idsat']
stage = 0
sw_per_stage = []
time = 0
tr_golden = 0.03356
Cap = data['load_cap']
def inrush(data,sw_per_stage,t):
    stages = len(sw_per_stage)
    operating_leakage_current_per_sw = data['operating_leakage_current'] / data['trickle_switches']
    inrush_data = [0]
    for j in range(stages):
        switch_current = 0
        input_signal_time = data['delay']*j
        complete_turnon_time = input_signal_time + data['Response']
        stage_leakage = sw_per_stage[j]*data['Ileak']
        if (t < input_signal_time):
            switch_current = stage_leakage
        elif (t > complete_turnon_time):
            switch_current = operating_leakage_current_per_sw * sw_per_stage[j]
        else:
            switch_current = (Idsat - (t - input_signal_time)*data['gradient'])*sw_per_stage[j]
        inrush_data.append(switch_current)
    return inrush_data

while sw_rem > 0:
    print("stage :"+str(stage))
    Icurrent = inrush(data,sw_per_stage,time)
    print("Stage "+str(stage)+" Current :")
    print(Icurrent)
    Icurrent = sum(Icurrent)
    sw = math.floor((inrush_max - Icurrent) / Idsat)
    if sw<sw_rem:
        sw_per_stage.append(sw)
    else:
        sw_per_stage.append(sw_rem)
    swon = sum(sw_per_stage)
    trise = tr_golden*Cap/swon
    if trise < time:
        print("Reached 95% VDD at Stage "+str(stage))
    sw_rem = sw_rem - sw
    time = time + data['delay']
    stage = stage + 1

print("Optimum switch pattern :")
print(sw_per_stage)
stages = len(sw_per_stage)
wakeup_latency = stages*data['delay']+data['Response']
print("Number of switches: "+str(sum(sw_per_stage)))
print("Number of stages: "+str(len(sw_per_stage)))
print("Wakeup latency : "+str(wakeup_latency))
