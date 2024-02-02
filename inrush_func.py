import json, csv, math
import matplotlib.pyplot as plt

config = "C:/Users/gvikr/OneDrive/Desktop/Inrush_automation/config.json"
fp = open(config)
data = json.load(fp)

def inrush(data):
    stages = math.ceil(data['trickle_switches']/data['sw_per_stage'])
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
            stage_leakage = data['sw_per_stage']*data['Ileak']
            if (current_time < input_signal_time):
                switch_current = stage_leakage
            elif (current_time > complete_turnon_time):
                switch_current = operating_leakage_current_per_sw * data['sw_per_stage']
            else:
                switch_current = (data['Idsat'] - (current_time - input_signal_time)*data['gradient'])*data['sw_per_stage']
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
    #plt.plot(sim_time,current_values)
    #plt.title("Inrush Plot")
    #plt.xlabel("Time")
    #plt.ylabel("Current")
    #plt.show()

    #with open('C:/Users/gvikr/OneDrive/Desktop/Inrush_automation/inrush.csv','w',newline='') as fpw:
    #    csvwriter = csv.writer(fpw)
    #    sim_time2 = list(sim_time)
    #    sim_time2.insert(0,'Stages')
    #    csvwriter.writerow(sim_time2)
    #    csvwriter.writerows(inrush_data)
    #    current_values.insert(0,' ')
    #    csvwriter.writerow(current_values)

sw_per_stage = range(1,41,1)
stages_list = list(map(lambda x: math.ceil(data['trickle_switches']/x), sw_per_stage))
inrush_current = []
wl = []
for i in sw_per_stage:
    data['sw_per_stage'] = i
    t = inrush(data)
    inrush_current.append(t[0])
    wl.append(t[1])

print(stages_list)
print(inrush_current)
print(wl)

fig,ax = plt.subplots()
ax1 = ax.twinx()
ax.plot(sw_per_stage,inrush_current,color='red',label='Max Inrush Current')
ax1.plot(sw_per_stage,wl,color='green',label='Wakeup Latency')
ax.set_xlabel("Switches Per Stage")
ax.set_ylabel("Max Inrush Current")
ax1.set_ylabel("Wakeup Latency")
plt.title("Max Current and Wakeup latency")
ax.legend()
ax1.legend()
plt.show()