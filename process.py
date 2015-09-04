# library imports
import os
import numpy as np
import pandas as pd
import CAM_NWB as cn
from scipy.stats import ks_2samp


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


def getConditions(stimulus_table, sweep_response, intersweep, sweeplength, orivals, tfvals, number_cells):

    # average cell response data for stimulus trials
    conditions = np.ndarray(shape=(len(orivals),len(tfvals)), dtype=list, order='C') # numpy array to store all cell responses for each condition
    for oi, ori in enumerate(orivals): # loop over orientations
        df = stimulus_table[stimulus_table['orientation']==ori]
        for ti, tf in enumerate(tfvals): # loop over temporal frequencies, for a given orientation
            data = df[df['temporal_frequency']==tf] # data block corresponding to given orientation and temporal freq
            start = data.index.values[0]
            stop = data.index.values[-1]
            cr = sweep_response[start:stop][np.array(range(number_cells)).astype(str)].values # cell responses
            cr = cr.mean(axis=0) # find average response over all trials for condition
            cr = [cr[i][int(intersweep):int(intersweep+sweeplength)] for i in range(len(cr))] # restrict to stimulus presentation times
            conditions[oi,ti] = cr

    return conditions


def findSigROIs(conditions, orivals, tfvals, blanks, PTHRESH):

    # finds significantly responding cells
    pdvals = np.ndarray(shape=(len(orivals),len(tfvals)), dtype=np.ndarray, order='C') # numpy array that will contain D parameters and p-values
    sig_rois = [] # list that will contain ids of significantly responding rois (those that respond to at least one stimulus condition)
    for (oi,ti), data in np.ndenumerate(conditions):
        pdvec = np.ndarray(shape=(1,len(data)), dtype=list, order='C')
        for cell_num, cell_data in enumerate(data):
            stim = np.asarray(cell_data)
            blank = np.asarray(blanks[cell_num])
            D, pval = ks_2samp(stim, blank)
            pdvec[0,cell_num] = [D,pval]
        pdvals[oi,ti] = pdvec
        these_rois = [i for i,x in enumerate(pdvals[oi,ti][0]) if x[1]<PTHRESH/float(conditions.size)]
        sig_rois = np.union1d(sig_rois,these_rois) # find unique values in union of all significantly responding rois so far, and this condition's

    return sig_rois