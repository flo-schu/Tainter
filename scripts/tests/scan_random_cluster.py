import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import os as os
import itertools as it

#random.seed(1)
#import networkx as nx
# os.chdir(os.getcwd()+'\\Tainter\\Models\\tf5_cluster')

try:
    id = int(os.environ['SLURM_ARRAY_TASK_ID']) + 1
except:
    id = 1

if os.path.exists("./data/temp/"+str(id)):
    tempfiles = os.listdir("./data/temp/"+str(id))
    for f in tempfiles:
        os.remove('./data/temp/'+str(id)+"/"+f)
    os.rmdir('./data/temp/'+str(id))

os.makedirs("./data/temp/"+str(id))

################################################################################
############################### MODEL ##########################################
################################################################################

# import sys;sys.path.append(os.getcwd()+'\\Tainter\\Models\\tf5')
from tainter_function_5_0 import tf5 as tainter_expl
import tainter_function_blocks as tblocks
import tainter_plots as tplots

def store_output():
    global par1range, par2range, par3range, par4range, par5range
    global survivaltime, args_record, maxenergy, wellbeing, other
    global k, p, exploration, threshold, efficiency, t, args, merun, wb
    global id, p1, i

    par1range.append(k)
    par2range.append(p)
    par3range.append(exploration)
    par4range.append(threshold)
    par5range.append(efficiency)
    survivaltime.append(t+1) # starts with 0
    args_record.append(args)
    maxenergy.append(merun)
    wellbeing.append(wb)
    struc_net = list()
    struc_net.append(np.mean(np.array(history["Administration"])/args["N"]))
    struc_net.append( np.std(np.array(history["Administration"])/args["N"]))
    struc_net.append(np.mean(np.array(history["coordinated Labourers"])/args["N"]))
    struc_net.append( np.std(np.array(history["coordinated Labourers"])/args["N"]))
    #TODO: Save ECAP
    struc_net
    other.append(struc_net)

    if (i+1) % save_every == 0 and i > 0:

        tblocks.save_data(folder_name = "temp/"+str(id), filename= str(p1), d1 = par1range, d2 = par2range, d3 = par3range, d4 = par4range, d5 = par5range, d6 = survivaltime, d7 = maxenergy, d8 = wellbeing, d9 = args_record, d10 = other)
        p1 +=1
        survivaltime = list()
        maxenergy = list()
        wellbeing = list()
        other     = list()
        struc_net = list()
        par1range = list()
        par2range = list()
        par3range = list()
        par4range = list()
        par5range = list()
        args_record = list()

def write_message():
    global survivaltime, args_record, maxenergy, wellbeing
    global k, p, exploration, threshold, efficiency, t, args, merun, wb
    global functime, logtime
    global id, p1, i
    logmessage = "k: %s | p: %s | expl: %s | thresh: %s | eff: %s | Run: %s | ME: %s | ST: %s | WB: %s | Runtime: %ss | Log: %ss" % (k, round(p,2), round(exploration,2), round(threshold,2), round(efficiency,2), i, round(merun,0), t+1, wb,round(functime,2), round(logtime,2))
    loglength = len(logmessage)
    print(" "*loglength + "\r" + logmessage)

par1name = "links"
par2name = "link.density"
par3name = "exploration"
par4name = "threshold"
par5name = "eff"
fct = "tf5"

# Initialize List
survivaltime = list()
maxenergy = list()
wellbeing = list()
other     = list()
par1range = list()
par2range = list()
par3range = list()
par4range = list()
par5range = list()
args_record = list()


# WRITE LOG
tblocks.save_data(folder_name = "temp/"+str(id), filename= "log", d1 = {"time":time.strftime("%Y-%m-%d-%H%M"),"fct":fct,"par1name":par1name, "par2name":par2name, "par3name": par3name, "par4name":par4name,"par5name":par5name})

tmax = 10000
save_every = 5
tempfiles = os.listdir("./data/temp/"+str(id))
p1 = len(tempfiles)
# anzahl knoten erhöhenm
# expand grid formel schreiben

#rk = np.arange(5,10)
rk = np.array([0])
rp = np.linspace(0.0,0.10,11)

rexploration = np.linspace(0.0,1.0,400)
# exp1 = np.linspace(0,0.0475,20)
# exp2 = np.linspace(0.05, 0.75, 11)
# exp3 = np.linspace(0.7525 , 1.0, 100)
# rexploration = np.concatenate((exp1, exp2[(exp2 > exp1.max()) & (exp2 < exp3.min())], exp3))

rthreshold = np.array([1])
refficiency = np.linspace(1.0,1.3,13)
rrep = np.arange(100)



# create grid for parameter loops:
pargrid = list(it.product(rk,rp,rexploration,rthreshold,refficiency,rrep))
pargrid = pd.DataFrame(pargrid)
pargrid = pargrid.rename(columns = {0:"k",1:"p",2:"expl",3:"thresh",4:"eff",5:"rep"})

# calculate range in dependence of the cluster id length of the parameter grid and
# number of clusters
n = len(pargrid)
nc = 3575

# calculate optimum number of clusters
# while n % nc != 0:
#     nc -= 1
lpc = n / nc

assert n % nc == 0

print(f'runs: {n},',f'jobs: {nc},', f'runs per job: {lpc}')



for i in range(int((int(id)-1)*lpc),int(lpc*int(id))):
    starttime = time.time()
    k = pargrid.loc[i,"k"]
    p = pargrid.loc[i,"p"]
    exploration = pargrid.loc[i,"expl"]
    threshold   = pargrid.loc[i,"thresh"]
    efficiency  = pargrid.loc[i,"eff"]

    history, t, args, fct, merun, wb = tainter_expl(
                              network = "erdos" ,
                              N = 400,
                              k = k,
                              p = p,
                              layout = "fixed",
                              first_admin = "highest degree" ,#???
                              choice = "topcoordinated",#????
                              exploration = exploration,
                              mepercent = 0.75,
                              a = 1.0 ,
                              stress = ["off","linear",0.01] ,
                              shock = ["on","beta",[1,15]],
                              tmax = tmax,
                              threshold = threshold ,
                              eff = efficiency ,
                              death_energy_level = 0.25,
                              print_every = None)

    functime = time.time() - starttime # Time used by Tainter Function
    store_output()
    logtime = time.time() - starttime - functime # Time used to log results
    write_message()

print("Model complete. Now processing output...")

################################################################################
######################## DATA PROCESSING #######################################
################################################################################

tempfiles = os.listdir("./data/temp/"+str(id))
survivaltime = list()
maxenergy = list()
wellbeing = list()
mean_A    = list()
sd_A      = list()
mean_LC   = list()
sd_LC     = list()
par1range = list()
par2range = list()
par3range = list()
par4range = list()
par5range = list()
log = tblocks.load_data("./data/temp/"+str(id)+"/log.pkl")[0]
par1name = log["par1name"]
par2name = log["par2name"]
par3name = log["par3name"]
par4name = log["par4name"]
par5name = log["par5name"]

for i in range(1,len(tempfiles)):
    d = tblocks.load_data("./data/temp/"+str(id)+"/"+str(i)+".pkl")
    par1range +=d[0]
    par2range += d[1]
    par3range += d[2]
    par4range += d[3]
    par5range += d[4]
    survivaltime += d[5]
    maxenergy += d[6]
    wellbeing += d[7]
    mean_A  += list(np.transpose(d[9])[0])
    sd_A    += list(np.transpose(d[9])[1])
    mean_LC += list(np.transpose(d[9])[2])
    sd_LC   += list(np.transpose(d[9])[3])
    print(i,end= "\r")

len(wellbeing)
len(mean_A)
data = pd.DataFrame(np.transpose([par1range,par2range,par3range, par4range, par5range,survivaltime,maxenergy, wellbeing,mean_A,sd_A,mean_LC,sd_LC]))
data.columns = [par1name, par2name, par3name,par4name,par5name,'survivaltime', 'tot.energy', 'wellbeing','mean_A','sd_A','mean_LC','sd_LC']

#OPTIONAL (HISTORY, means)
# history = pd.DataFrame()
# for i in range(len(tempfiles)-1):
#     d = tblocks.load_data("./Tainter/data/temp/"+str(i)+".pkl")
#     for j in range(random_repeat):
#         history = history.append(pd.DataFrame(d[5][j]).mean(), ignore_index = True)
#         print(i,j, end = "\r")

# Get arguments
args = pd.DataFrame(tblocks.load_data("./data/temp/"+str(id)+"/1.pkl")[8][0])

# Save as PICKLE
if not os.path.exists("./data/temp/output"):
    os.makedirs("./data/temp/output")
tblocks.save_data(folder_name = "temp/output", filename = str(id)+"_"+time.strftime("%Y-%m-%d-%H%M"), d1 = log, d2 = args, d3 = data)

# Save as CSV
data.to_csv('./data/temp/output'+'/' + str(id)+"_"+'data_'+ time.strftime("%Y-%m-%d-%H%M")+'.csv',na_rep = 'NA')

# Save logfile
loglist = [a+": "+b for a,b in zip(log.keys(),log.values())]
f = open('./data/temp/output/'+str(id)+"_"+'log_'+time.strftime("%Y-%m-%d-%H%M")+'.txt', 'w')
for item in loglist:
    f.write("%s\n" % item)
f.write("\n Arguments passed to the model: \n")
f.write(args.to_string())
f.write("\n Note that arguments from parnames are just randomly assigned! \n Explanations: \n links: k \n rewiring: p")
f.close()

tempfiles = os.listdir("./data/temp/"+str(id))
for f in tempfiles:
    os.remove('./data/temp/'+str(id)+"/"+f)
os.rmdir('./data/temp/'+str(id))
