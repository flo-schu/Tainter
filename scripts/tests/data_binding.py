import os
import pandas as pd
from shutil import copyfile
import time
# os.chdir(os.getcwd()+'\\Tainter\\Models\\tf5')
import tainter_function_blocks as tblocks
import tainter_plots as tplots

filelist = os.listdir("./data/temp/output")
#select only csv
datalist = [i for i in filelist if "pkl" in i]
loglist = [i for i in filelist if "txt" in i]

data = pd.DataFrame()
for i in datalist:
    data = data.append(tblocks.load_data("./data/temp/output/"+i)[2], ignore_index = True)
    print(i)

#example log and args
log = loglist[0]
copyfile("./data/temp/output/"+log,"./data/log_"+ time.strftime("%Y-%m-%d-%H%M") +".txt")

# Save as CSV
data.to_csv('./data/'+'data_'+ time.strftime("%Y-%m-%d-%H%M")+'.csv',na_rep = 'NA')

#history.to_csv('./Tainter/data/'+folder_name+'/history_'+ time.strftime("%Y-%m-%d-%H%M")+'.csv',na_rep = 'NA')

print("Done! Files are in:", os.getcwd()+"\\data")

tempfiles = os.listdir("./data/temp/output")
for f in tempfiles:
        os.remove('./data/temp/output/'+f)
os.rmdir("./data/temp/output")
