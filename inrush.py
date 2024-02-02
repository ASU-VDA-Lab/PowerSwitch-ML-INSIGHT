import json, csv, math
import matplotlib.pyplot as plt

config = "C:/Users/gvikr/OneDrive/Desktop/Inrush_automation/config.json"
fp = open(config)
data = json.load(fp)
#data["sw_per_stage"] = [40-x for x in range(40)]
data['sw_per_stage'] = [192, 8, 9, 10, 9, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 16, 16, 17, 18, 19, 20, 20, 21, 2, 13, 13, 13, 13, 14, 13, 14, 13, 14, 14, 13, 14, 13, 14, 13, 13, 13, 13, 12, 12, 11, 11, 13, 12, 12, 12, 11, 12, 12, 11, 11, 12, 6]
#data["trickle_switches"] = sum(data["sw_per_stage"])
if (type(data['sw_per_stage']) == type([])):
    stages = len(data['sw_per_stage'])
else:
    stages = math.ceil(data['trickle_switches']/data['sw_per_stage'])
    data['sw_per_stage'] = [data['sw_per_stage'] for i in range(stages)]

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
        stage_leakage = data['sw_per_stage'][j]*data['Ileak']
        if (current_time < input_signal_time):
            switch_current = stage_leakage
        elif (current_time > complete_turnon_time):
            switch_current = operating_leakage_current_per_sw * data['sw_per_stage'][j]
        else:
            switch_current = (data['Idsat'] - (current_time - input_signal_time)*data['gradient'])*data['sw_per_stage'][j]
        stage_current.append(switch_current)
    inrush_data.append(stage_current)
#print(current_data)

for i in range(1,len(sim_time)+1):
    value = 0
    for j in range(stages):
        value = value + float(inrush_data[j][i])
    current_values.append(value)

print("Max Inrush Current: "+str(max(current_values)))
print("Switches per stage : "+" ".join([str(i) for i in data['sw_per_stage']]))
plt.plot(sim_time,current_values)
plt.title("Inrush Plot")
plt.xlabel("Time")
plt.ylabel("Current")
plt.show()

with open('C:/Users/gvikr/OneDrive/Desktop/Inrush_automation/inrush.csv','w',newline='') as fpw:
    csvwriter = csv.writer(fpw)
    sim_time2 = list(sim_time)
    sim_time2.insert(0,'Stages')
    csvwriter.writerow(sim_time2)
    csvwriter.writerows(inrush_data)
    current_values.insert(0,' ')
    csvwriter.writerow(current_values)
