import json, csv, math
import matplotlib.pyplot as plt

lut_file = "C:/Users/gvikr/OneDrive/Desktop/Inrush_automation/lut.txt"
lut = dict()
with open(lut_file) as fp:
    for line in fp.readlines():
        lut[float(line.split()[0])] = float(line.split()[1])
print(lut)
def lookup(vds,lut):
    if vds in lut:
        return lut[vds]
    else:
        i = 0
        for k in lut.keys():
            if vds < k:
                if k == 0:
                    print("Error: vds less than 0")
                    return
                else:
                    low = list(lut.keys())[i-1]
                    high = list(lut.keys())[i]
                    ids = ((high - vds)*lut[low] + (vds - low)*lut[high])/(high-low)
                    return ids
            i = i + 1
        if vds > list(lut.keys())[i-1]:
            print("vds is greater then available voltages...doing extrapolation")
            low = list(lut.keys())[i-2]
            high = list(lut.keys())[i-1]
            ids = lut[low]+(((vds - low)/(high-low))*(lut[high] - lut[low]))
            return ids

config = "C:/Users/gvikr/OneDrive/Desktop/Inrush_automation/config.json"
fp = open(config)
data = json.load(fp)
#data["sw_per_stage"] = [40-x for x in range(40)]
#data['sw_per_stage'] = [20, 2, 3]
#data['sw_per_stage'] = [6, 6, 6, 7, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 15, 15]
data['sw_per_stage'] = [36, 8, 11, 17, 28]
#data['sw_per_stage'] = [34,6,10]
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
step_size = 5
stop_time = math.ceil(wakeup_latency/step_size)*step_size
#stop_time = 2600
sim_time = range(0,stop_time,step_size)
print("Stop Time : "+str(stop_time))
print("Step Size : "+str(step_size))
print("Total Time Steps : "+str(len(sim_time)))
print("Total Stages : "+str(stages))
print("Wakeup latency: "+str(wakeup_latency))
time_current = []
vol = []
vdd_sw = 0
vdd = data['supply_voltage']
vds = vdd - vdd_sw
for current_time in sim_time:
    stage_current = []
    for j in range(stages):
        #stage_current = [str(j)]
        input_signal_time = data['delay']*j
        stage_leakage = data['sw_per_stage'][j]*data['Ileak']
        if (current_time < input_signal_time):
            switch_current = stage_leakage
        else:
            print(vds)
            print(vdd_sw)
            switch_current = lookup(vds,lut)*data['sw_per_stage'][j]
            print(switch_current)
        stage_current.append(switch_current)
    inrush_data.append(stage_current)
    total_current = sum(stage_current)
    time_current.append(total_current)
    print(sum(time_current)*step_size)
    vdd_sw = (sum(time_current)*step_size*(10**-12))/data['load_cap']
    if (vdd_sw >= vdd):
        print("Switch supply reached 95% vdd at time "+str(current_time))
        #break
    vds = vdd - vdd_sw
    vol.append(vdd_sw)
print(stage_current)
print(inrush_data)
print(vol)

print("Max Inrush Current: "+str(max(time_current)))
print("Switches per stage : "+" ".join([str(i) for i in data['sw_per_stage']]))
plt.plot(sim_time,time_current)
plt.title("Inrush Plot")
plt.xlabel("Time")
plt.ylabel("Current")
plt.show()

with open('C:/Users/gvikr/OneDrive/Desktop/Inrush_automation/inrush.csv','w',newline='') as fpw:
    csvwriter = csv.writer(fpw)
    sim_time2 = list(sim_time)
    #sim_time2.insert(0,'Stages')
    csvwriter.writerow(sim_time2)
    id_transpose = list(map(lambda *x: list(x), *inrush_data))
    csvwriter.writerows(id_transpose)
    csvwriter.writerow(time_current)
    csvwriter.writerow([])
    csvwriter.writerow([])
    csvwriter.writerow(vol)