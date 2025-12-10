import json, csv, math, os, time
import itertools
import multiprocessing as mp
import threading

lut_file = "/home/vgopal18/python/lut2.txt"
lut2 = dict()
vin_keys = [0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]
print(vin_keys)
with open(lut_file) as fp:
    for line in fp.readlines():
        lut2[float(line.split()[0])] = {}
        for k in range(len(vin_keys)):
            lut2[float(line.split()[0])][vin_keys[k]] = float(line.split()[k+1])
#print(lut2)

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

def inrush(data, sw_per_stage):
    inrush_data = []
    stages = len(sw_per_stage)
    step_size = 5
    #stop_time = math.ceil(wakeup_latency/step_size)*step_size
    stop_time = int(1E4)
    sim_time = range(0,stop_time,step_size)
    #print("Stop Time : "+str(stop_time))
    #print("Step Size : "+str(step_size))
    #print("Total Time Steps : "+str(len(sim_time)))
    #print("Total Stages : "+str(stages))
    #print("Wakeup latency: "+str(wakeup_latency))
    slew = [1.4+0.8498*(x-1) for x in sw_per_stage]
    #wakeup_latency = stages*data['delay']+slew[-1]
    time_current = []
    vol = []
    vdd_sw = 0
    vdd = data['supply_voltage']
    vds = vdd - vdd_sw
    vin = 0.7
    vth = 0.3
    exit_loop = 0
    operating_leakage_current_per_sw = data['operating_leakage_current'] / sum(sw_per_stage)
    for current_time in sim_time:
        stage_current = []
        for j in range(stages):
            #stage_current = [str(j)]
            input_signal_time = data['delay']*j
            input_signal_time_end = input_signal_time + slew[j]
            stage_leakage = sw_per_stage[j]*data['Ileak']
            if (current_time < input_signal_time):
                switch_current = lookup2d(vds,vin,lut2)*sw_per_stage[j]
            else:
                if (current_time >= input_signal_time_end):
                    vin_now = 0
                else:
                    vin_now = ((input_signal_time_end - current_time)*vin)/slew[j]
                #print(vds)
                #print(vdd_sw)
                if (vds < vdd-vth):
                    switch_current = operating_leakage_current_per_sw*sw_per_stage[j]
                    wakeup_latency = current_time
                    exit_loop = 1
                else:
                    switch_current = lookup2d(vds,vin_now,lut2)*sw_per_stage[j]
                #switch_current = lookup(vds,lut)*sw_per_stage[j]
                #print(switch_current)
            stage_current.append(switch_current)
        inrush_data.append(stage_current)
        total_current = sum(stage_current)
        time_current.append(total_current)
        #print(sum(time_current)*step_size)
        #vdd_sw = (sum(time_current)*step_size*(10**-12))/data['load_cap']
        vdd_sw = integrate(time_current,step_size*(10**-12))/data['load_cap']
        if (vdd_sw >= vdd):
            print("Switch supply reached 95% vdd at time "+str(current_time))
            #break
        vds = vdd - vdd_sw
        vol.append(vdd_sw)
        if(exit_loop):
            break
    #print(stage_current)
    #print(inrush_data)
    #print(vol)
    #print("Max Inrush Current: "+str(max(time_current)))
    return max(time_current),wakeup_latency

start = time.time()
config = "/home/vgopal18/python/config.json"
fp = open(config)
data = json.load(fp)
i = [56,34,10]

#Ariane	1000	8	28.653	26.1
#AES256	930	7	22	23.71
#Mempool	710	7	19.49	17.14
#Ariane	180	8	28.653	25000
#AES256	175	7	22	23.71
#Mempool	160	7	19.49	3700
#Ibex	145	6	2.783	3.04
#JPEG	140	6	6.13	6.57
#AES128	140	5	2.048	1.89
#Ethmac	140	4	9.148	11.74
#Raven_SHA	135	4	3.69	3.71
#Mock-array	120	3	10.198	0.47
#Amber	110	3	1	0.68

design = "marr"
patterns = "/home/vgopal18/python/ML_data/python_data/patterns/"+design+"_patterns.txt"
cap = 10.198e-12
ileak = 0.47e-6
out = "/home/vgopal18/python/ML_data/python_data/clean_data/new/results_"+design+".csv"
stage = 3

head = ",".join(["Stage"+str(i) for i in range(1,stage+1)])
head = head + ',MaxI,Wl\n'

fpw = open(out,'w')
fp = open(patterns,'r')
fpw.write(head)

data['load_cap'] = cap 
data['operating_leakage_current'] = ileak
fp.readline()
for l in fp.readlines():
    pattern = [int(x) for x in l.rstrip().split(',')]
    irush, wl = inrush(data,pattern)
    fpw.write(l.rstrip()+','+str(irush)+','+str(wl)+'\n')

fp.close()
fpw.close()
end = time.time()
diff = end -start
print("Finished writing ",out)
print("Run Time: ",diff,' seconds')

