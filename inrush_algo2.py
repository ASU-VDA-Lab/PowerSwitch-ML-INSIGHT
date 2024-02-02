import json, csv, math
import itertools

def inrush(data, sw_per_stage):
    stages = len(sw_per_stage)
    operating_leakage_current_per_sw = data['operating_leakage_current'] / data['trickle_switches']
    inrush_data = []
    current_values = []
    wakeup_latency = stages*data['delay']+data['Response']
    step_size = 250
    stop_time = math.ceil(wakeup_latency/step_size)*step_size
    sim_time = range(0,stop_time,step_size)
    print("Stop Time : "+str(stop_time))
    print("Step Size : "+str(step_size))
    print("Total Time Steps : "+str(len(sim_time)))
    print("Total Stages : "+str(stages))
    print("Wakeup latency: "+str(wakeup_latency))
    for j in range(stages):
        stage_current = [str(j)]
        for current_time  in sim_time:
            input_signal_time = data['delay']*j
            complete_turnon_time = input_signal_time + data['Response']
            stage_leakage = sw_per_stage[j]*data['Ileak']
            if (current_time < input_signal_time):
                switch_current = stage_leakage
            elif (current_time > complete_turnon_time):
                switch_current = operating_leakage_current_per_sw *sw_per_stage[j]
            else:
                switch_current = (data['Idsat'] - (current_time - input_signal_time)*data['gradient'])*sw_per_stage[j]
            stage_current.append(switch_current)
        inrush_data.append(stage_current)
    #print(current_data)

    for i in range(1,len(sim_time)+1):
        value = 0
        for j in range(stages):
            value = value + float(inrush_data[j][i])
        current_values.append(value)

    print("Max Inrush Current: "+str(max(current_values)))
    return [max(current_values),wakeup_latency]

config = "C:/Users/gvikr/OneDrive/Desktop/Inrush_automation/config.json"
fp = open(config)
data = json.load(fp)
wakeup_latency = 1145
stages = math.floor((wakeup_latency-data['Response'])/data['delay'])
switches = data['trickle_switches']
numbers = range(1,switches+1)
patterns = []
for i in range(stages,0,-1):
    for seq in itertools.combinations_with_replacement(numbers, i):
        if sum(seq) == switches:
            for j in itertools.permutations(seq):
                patterns.append(j)

print("Number of switches: "+str(switches))
print("Number of stages: "+str(stages))
print("Wakeup latency : "+str(wakeup_latency))
print("Possible patterns:")
print(patterns)
operating_leakage_current_per_sw = data['operating_leakage_current'] / data['trickle_switches']
inrush_data = []
current_values = []
#wakeup_latency = stages*data['delay']+data['Response']
sw_rem = data['trickle_switches']
Idsat = data['Idsat']
stage = 0
sw_per_stage = []
time = 0
max_current = []
wl = []
pattern_idx = 0
inrush_best = 9999999999999999999999999
j = 0

for i in patterns:
    t = inrush(data,i)
    max_current.append(t[0])
    wl.append(t[1])
    if (t[0]<inrush_best):
        inrush_best = t[0]
        pattern_idx = j
    j=j+1


print(patterns)
print(max_current)
print("Max inrush current :"+str(max_current[pattern_idx]))
print("Wakeup latency: "+str(wl[pattern_idx]))
print("Optimum switch pattern :")
print(patterns[pattern_idx])
