# library imports
import os
import numpy as np
import pandas as pd
import CAM_NWB as cn


# function definitions
def makeDF(camdir):

    # creates data frame for CAM experiments, and returns paths to all experiments

    list_of_dirs = os.walk(camdir).next()[1]
    list_of_paths = []
    for i in list_of_dirs:
        nwb_path = camdir + i + '/' + i + '.nwb'
        list_of_paths.append(nwb_path)

    CAM_df = pd.DataFrame(index=range(len(list_of_paths),),columns=cn.getMetaData(list_of_paths[0]))
    row_counter = 0
    for path in list_of_paths:
        CAM_df.loc[row_counter] = pd.Series(cn.getMetaData(path))
        row_counter += 1

    # add a metric of running speed to that data frame
    mean_speed = []
    for path in list_of_paths:
        run_data = cn.getRunningSpeed(path)
        mean_speed.append(np.nanmean(run_data[0]))
    CAM_df['mean speed'] = mean_speed
    CAM_df['path_to_exp'] = list_of_paths

    return list_of_paths, CAM_df


def getResponses(stim_table, celltraces, num_cells, intersweep, sweeplength):

    # create an empty data frame with columns for each ROI and index (rows) for each stimulus sweep
    sweep_response = pd.DataFrame(index=stim_table.index.values, columns=np.array(range(num_cells)).astype(str))

    # populate data frame with dF/F trace of each ROI in response to each grating
    for index, row in stim_table.iterrows():
        start = row['start'] - intersweep  # UNCOMMENT TO RETRIEVE FULL TRACE (INCLUDING INTERSWEEP)
        end = row['start'] + sweeplength + intersweep
        for nc in range(num_cells):
            temp = celltraces[nc,start:end]
            sweep_response[str(nc)][index] = 100*((temp/np.mean(temp[:intersweep]))-1) # locally defined dF/F

    return sweep_response