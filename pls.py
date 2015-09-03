import numpy as np
import pandas as pd

def pls(stimulus_table, sweep_response, num_cells):

    # takes raw data in the form of "data", where data is a data frame containing N*K rows and P columns, for K
    # conditions, N trials per condition and P cells.  each data frame entry is a time series of Tf fluorescence data
    # points.  executes partial least squares algorithm to generate:
    #
    # Udev: voxel/pixel saliences
    # Sdev: LV singular values
    # Vdev: task saliences
    # B: brain scores

    # calculate constants
    t1 = stimulus_table['temporal_frequency'].unique()[0]
    o1 = stimulus_table['orientation'].unique()[0]
    N = stimulus_table[(stimulus_table['orientation']==o1)&(stimulus_table['temporal_frequency']==t1)].shape[0] # number of trials per condition
    P = num_cells # number of cells (pixels) in dataset

    orivals = stimulus_table.orientation.dropna().unique() # find unique orientation values
    tfvals = stimulus_table.temporal_frequency.dropna().unique() # find unique temporal frequency values
    orivals.sort()
    tfvals.sort()
    K = len(orivals)*len(tfvals) # number of experimental conditions

    
    Tf = # get minimum length of trial in cell traces
    number_cells = np.size(celltraces,0)
    sweeplength = stimulus_table['end'][1] - stimulus_table['start'][1]
    intersweep = stimulus_table['start'][2] - stimulus_table['end'][1]

    sweep_response = pd.DataFrame(index=stimulus_table.index.values, columns=np.array(range(number_cells)).astype(str))
    for index, row in stimulus_table.iterrows():
        start = row['start'] - intersweep  # UNCOMMENT TO RETRIEVE FULL TRACE (INCLUDING INTERSWEEP)
        end = row['start'] + sweeplength + intersweep # TODO: figure this out
        for nc in range(number_cells):
            temp = celltraces[nc,start:start+Tf#TODO???]
            sweep_response[str(nc)][index] = 100*((temp/np.mean(temp[:intersweep]))-1) # locally defined dF/F

