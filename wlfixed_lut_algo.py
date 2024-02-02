import json, csv, math
import itertools

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

def inrush(data, sw_per_stage):
    inrush_data = []
    stages = len(sw_per_stage)
    wakeup_latency = stages*data['delay']+data['Response']
    step_size = 5
    stop_time = math.ceil(wakeup_latency/step_size)*step_size
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
            stage_leakage = sw_per_stage[j]*data['Ileak']
            if (current_time < input_signal_time):
                switch_current = stage_leakage
            else:
                print(vds)
                print(vdd_sw)
                switch_current = lookup(vds,lut)*sw_per_stage[j]
                print(switch_current)
            stage_current.append(switch_current)
        inrush_data.append(stage_current)
        total_current = sum(stage_current)
        time_current.append(total_current)
        #print(sum(time_current)*step_size)
        vdd_sw = (sum(time_current)*step_size*(10**-12))/data['load_cap']
        if (vdd_sw >= vdd):
            print("Switch supply reached 95% vdd at time "+str(current_time))
            #break
        vds = vdd - vdd_sw
        vol.append(vdd_sw)
    #print(stage_current)
    #print(inrush_data)
    #print(vol)
    #print("Max Inrush Current: "+str(max(time_current)))
    return [max(time_current),wakeup_latency]

config = "C:/Users/gvikr/OneDrive/Desktop/Inrush_automation/config.json"
fp = open(config)
data = json.load(fp)
wakeup_latency = 1245
stages = math.floor((wakeup_latency-data['Response'])/data['delay'])
stages = 3
#switches = data['trickle_switches']
switches = 32
numbers = range(1,switches+1)
patterns = []
for i in range(stages,0,-1):
    for seq in itertools.combinations_with_replacement(numbers, i):
        if sum(seq) == switches and len(seq) == stages:
            for j in itertools.permutations(seq):
                if j not in patterns:
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
