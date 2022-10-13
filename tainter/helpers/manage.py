import time
import shutil
import os

def icreate_folder_date(path):
    folder = time.strftime("%Y%m%d_%H%M")
    path = os.path.join(path, folder)
    if os.path.isdir(path):
        if input("dir exists. overwrite? (y/n) ") == "y":
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)
    
    return path