# library imports
import numpy as np


# function definitions
def pls(stimulus_table, sweep_response, number_cells, intersweep, sweeplength):

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
    orivals = stimulus_table.orientation.dropna().unique() # find unique orientation values
    tfvals = stimulus_table.temporal_frequency.dropna().unique() # find unique temporal frequency values
    K = len(orivals)*len(tfvals)+1 # number of experimental conditions
    P = number_cells # number of cells (pixels) in dataset
    Tf = sweeplength

    # get blank trials
    blanks = sweep_response[stimulus_table['blank_sweep']==1].values # blank stimulus trials
    blanks = blanks.mean(axis=0) # get average response over all trials for each cell

    # construct data matrix T (averaged over trials)
    T = np.zeros((K, P*Tf))
    for cell_index in range(P):
        li = 0 # linear index over conditions
        for oi, ori in enumerate(orivals):
            for ti, tf in enumerate(tfvals):
                tf_mask = stimulus_table['temporal_frequency'] == tf
                ori_mask = stimulus_table['orientation'] == ori
                trial_arr = np.array(sweep_response[tf_mask & ori_mask][str(cell_index)].tolist())
                lens = [len(trial) for trial in trial_arr]
                if min(lens) < max(lens): # justify trial lengths if one or more has a different length (so we can index into array later)
                    trial_arr = np.array([trial[0:min(lens)] for trial in trial_arr])
                if not np.all(np.isfinite(trial_arr)):
                    print("trial error (inf or NaN): excluding data for orientation = %f degrees, temporal frequency = %f" % (ori,tf))
                    continue
                T[li, cell_index*Tf:(cell_index+1)*Tf] = trial_arr[:, intersweep:intersweep+Tf].mean() # take mean over trials
                li += 1 # update linear index across conditions
        T[K-1, cell_index*Tf:(cell_index+1)*Tf] = blanks[cell_index][intersweep:intersweep+Tf] # append blank trial average

    # perform singular value decomposition and calculate brain scores
    T = np.matrix(T)
    v = np.matrix(np.ones((1,K)))
    vT = np.transpose(v)
    T = T-np.dot(vT, np.dot(v, T)/float(K))
    U,S,V = np.linalg.svd(np.transpose(T),full_matrices=False) # calculate svd
    B = np.dot(T,U)

    return U,S,V,B

