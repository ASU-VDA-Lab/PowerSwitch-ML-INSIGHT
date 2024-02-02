import json, csv, math
import matplotlib.pyplot as plt

lut_file = "C:/Users/gvikr/OneDrive/Desktop/Inrush_automation/lut2.txt"
lut2 = dict()
vin_keys = [0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]
print(vin_keys)
with open(lut_file) as fp:
    for line in fp.readlines():
        lut2[float(line.split()[0])] = {}
        for k in range(len(vin_keys)):
            lut2[float(line.split()[0])][vin_keys[k]] = float(line.split()[k+1])
print(lut2)

def lookup2d(vds, vin, lut):
    if vds in lut and vin in lut[vds]:
        return lut[vds][vin]
    else:
        i = 0
        for k in lut.keys():
            if vds <= k:
                if k == 0:
                    vdslow = list(lut.keys())[i]
                    vdshigh = list(lut.keys())[i+1]
                    return
                else:
                    vdslow = list(lut.keys())[i-1]
                    vdshigh = list(lut.keys())[i]
                    break
            i = i + 1
        i = 0
        for k in lut[vdslow].keys():
            if vin >= k:
                if k == 0.7:
                    vinlow = list(lut[vdslow].keys())[i+1]
                    vinhigh = list(lut[vdslow].keys())[i]
                else:
                    vinlow = list(lut[vdslow].keys())[i]
                    vinhigh = list(lut[vdslow].keys())[i-1]
                    break
            i = i+1
        f11 = lut[vdslow][vinlow]
        f12 = lut[vdslow][vinhigh]
        f21 = lut[vdshigh][vinlow]
        f22 = lut[vdshigh][vinhigh]
        #print(str(vdslow)+" "+str(vdshigh)+" "+str(vinhigh)+" "+str(vinlow))
        #print(str(f11)+" "+str(f12)+" "+str(f21)+" "+str(f22))
        #print(lut[vdslow].keys())
        ids = (f11*(vdshigh - vds)*(vinhigh - vin) + f21*(vds-vdslow)*(vinhigh - vin) + f12*(vdshigh - vds)*(vin - vinlow) + f22*(vds-vdslow)*(vin - vinlow))/((vdshigh-vdslow)*(vinhigh-vinlow))
        return ids

def integrate(current,step):
    n = 5
    mid_pts = list(step/(2*n)+(step*i/n) for i in range(n))
    l = len(current)
    sum = 0
    for i in range(l):
        if i == l - 1:
            sum = sum + current[i]*step
        else:
            t1 = i*step
            t2 = (i+1)*step
            temp_sum = 0
            for m in mid_pts:
                mid = m+t1
                temp_sum = temp_sum + (current[i]+(((mid - t1)/(t2-t1))*(current[i+1] - current[i])))*step/n
                #print(temp_sum)
            sum = sum + temp_sum
        #print(sum)
    return sum
    
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
vin = 0.7
slew = 20
for current_time in sim_time:
    stage_current = []
    for j in range(stages):
        #stage_current = [str(j)]
        input_signal_time = data['delay']*j
        input_signal_time_end = input_signal_time + slew
        stage_leakage = data['sw_per_stage'][j]*data['Ileak']
        if (current_time < input_signal_time):
            switch_current = lookup2d(vds,vin,lut2)*data['sw_per_stage'][j]
        else:
            if (current_time >= input_signal_time_end):
                vin_now = 0
            else:
                vin_now = ((input_signal_time_end - current_time)*vin)/slew
            print(vin)     
            print(vds)
            print(vdd_sw)
            switch_current = lookup2d(vds,vin_now,lut2)*data['sw_per_stage'][j]
            print(switch_current)
        stage_current.append(switch_current)
    inrush_data.append(stage_current)
    total_current = sum(stage_current)
    time_current.append(total_current)
    print(sum(time_current)*step_size)
    #vdd_sw = (sum(time_current)*step_size*(10**-12))/data['load_cap']
    vdd_sw = integrate(time_current,step_size*(10**-12))/data['load_cap']
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