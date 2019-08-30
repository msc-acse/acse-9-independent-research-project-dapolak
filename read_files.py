# -*- coding: utf-8 -*-
"""
Joining files script, to be used prior to uploading the data on DataBricks
"""

import pandas as pd
import glob

# Name of the sensors whch files need to be joined
sensors = ["well_wh_p_", "well_dh_p_", "well_wh_t_", "well_dh_t", "well_wh_choke_"]


# loop through each sensor and find all files with sensor as name
for csvfile in sensors:
    path = r'C:/Users/well_data/' #change path accordingly
    all_files = glob.glob(path + "/" + csvfile + "*" )
    
    print(len(all_files)) # Print total number of files present in repository
    li = [] # Create list to store .csv data during joining
        
    # get data out of each .csv file
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    
    print("All files read for sensor", csvfile)
    
    # concatenate the files together
    frame = pd.concat(li, axis=0, ignore_index=True, sort=False)
    
    print("Concatenated and now to save")
    
    # save sensor data as one file
    frame.to_csv(path_or_buf = path + "/" + csvfile +"total.csv", index=[" ", "ts", "name", "value"])
