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
inrush_max = 0.002
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
vol = []
current_total = 0
vdd_sw = 0
vdd = data['supply_voltage']
step_size = 10
while sw_rem > 0:
    print("stage :"+str(stage))
    vds = vdd - vdd_sw
    Icurrent = lookup(vds,lut)
    if (time >= stage*data['delay']):
        print("Stage "+str(stage)+" Current :")
    #Icurrent = sum(Icurrent)
        sw = math.floor(inrush_max / Icurrent)
        if sw<sw_rem:
            sw_per_stage.append(sw)
        else:
            sw_per_stage.append(sw_rem)
        stage = stage + 1
        sw_rem = sw_rem - sw
    swon = sum(sw_per_stage)
    current_stage = Icurrent*swon
    current_total = current_total + current_stage*step_size*(10**-12)
    vdd_sw = current_total/data['load_cap']
    vol.append(vdd_sw)
    if vdd_sw > 0.95*vdd:
        print("Reached 95% VDD at Stage "+str(stage))
    time = time + step_size

print("Optimum switch pattern :")
print(sw_per_stage)
stages = len(sw_per_stage)
wakeup_latency = stages*data['delay']+data['Response']
print("Number of switches: "+str(sum(sw_per_stage)))
print("Number of stages: "+str(len(sw_per_stage)))
print("Wakeup latency : "+str(wakeup_latency))
print(vol)
