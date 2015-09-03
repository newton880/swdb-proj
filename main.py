# authors: Taylor Newton and Daniela Saderi
# date: 27.8.2015
# institute: Allen Institute for Brain Science 2015 Summer Workshop on the Dynamic Brain
# description: this script extracts visually "responsive" ROIs from Ca2+ imaging data
# provided by the Allen Institue (TODO: flesh out file explanation).


# library imports
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import CAM_NWB as cn
from scipy.stats import ks_2samp
import sparseness as sp


# constants
MXROWS = 10 # maximum number of columns to display for pandas
MXCOLS = 6 # maximum number of rows to display for pandas
EXPID = str(479182359) # experiment ID number
CAMDIR = r'/Volumes/Brain2015/CAM/'
NTRIALS = 15 # number of trials per sweep
PTHRESH = 0.05 # p-value cutoff above which results are not significant
DTHRESH = 0.01 # D-statistic for K-S test below which results are not significant
NBINS = 20 # number of bins for histogram plot
DF_THRESH = 15 # percent threshold (dF/F) for cutting off non-responsive cells


# get experiment data and path to experiments
list_of_dirs = os.walk(CAMDIR).next()[1]
list_of_paths = []
for i in list_of_dirs:
    nwb_path = CAMDIR + i + '/' + i + '.nwb'
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

# LOOP OVER ALL EXPERIMENTS AND CALCULATE HISTOGRAM OF LIFETIME SPARSENESSES FOR EACH
life_spar_list = []
pop_spar_list = []
for path in list_of_paths:
    print path
    # extract interesting information
    meta = cn.getMetaData(path) # get meta data for experiment of interest
    proj = cn.getMaxProjection(path) # getMaxProjection returns a 512x512 array of the maximum projection of the 2P movie
    print meta

    # returns an array of raw fluorescence traces for each ROI and the timestamps for each time point.
    timestamps, celltraces = cn.getFluorescenceTraces(path)

    number_cells = np.size(celltraces,0)
    acquisition_rate = 1/(timestamps[1]-timestamps[0])
    print "Number of cells: ", number_cells
    print "Acquisition rate: %f Hz" % acquisition_rate

    # returns data frame of stimulus conditions
    stimulus_table = cn.getStimulusTable(path)
    number_sweeps = len(stimulus_table)

    # find unique orientation values
    orivals = stimulus_table.orientation.dropna().unique()
    orivals.sort()

    # find unique temporal frequency values
    tfvals = stimulus_table.temporal_frequency.dropna().unique()
    tfvals.sort()

    # calculate length of stimulus sweeps and inter-sweep intervals
    sweeplength = stimulus_table['end'][1] - stimulus_table['start'][1]
    intersweep = stimulus_table['start'][2] - stimulus_table['end'][1]

    # create an empty data frame with columns for each ROI and index (rows) for each stimulus sweep
    sweep_response = pd.DataFrame(index=stimulus_table.index.values, columns=np.array(range(number_cells)).astype(str))

    # populate data frame with dF/F trace of each ROI in response to each grating (restricted to stimulus presentation)
    sorted_table = stimulus_table.sort(['orientation','temporal_frequency'])
    sorted_table = sorted_table.reset_index(drop=True)
    for index, row in sorted_table.iterrows():
        start = row['start'] - intersweep  # UNCOMMENT TO RETRIEVE FULL TRACE (INCLUDING INTERSWEEP)
        end = row['start'] + sweeplength + intersweep
        for nc in range(number_cells):
            temp = celltraces[nc,start:end]
            sweep_response[str(nc)][index] = 100*((temp/np.mean(temp[:intersweep]))-1) # locally defined dF/F

    # find blank stimulus presentations
    blank = stimulus_table[stimulus_table['blank_sweep']==1]
    print "Number of blank trials: ", len(blank)

    # find visually "responsive" neurons: those whose response was significantly different from blank to at
    # least one condition (see Rust & DiCarlo, 2012), and whose dF/F is > DF_THRESH.

    # begin by averaging data for blank stimulus trials
    blanks = sweep_response[sorted_table['blank_sweep']==1].values # blank stimulus trials
    blanks = blanks.mean(axis=0) # get average response over all trials

    # now average cell response data for stimulus trials
    conditions = np.ndarray(shape=(len(orivals),len(tfvals)), dtype=list, order='C') # numpy array to store all cell responses for each condition
    for oi, ori in enumerate(orivals): # loop over orientations
        df = sorted_table[sorted_table['orientation']==ori]
        for ti, tf in enumerate(tfvals): # loop over temporal frequencies, for a given orientation
            data = df[df['temporal_frequency']==tf] # data block corresponding to given orientation and temporal freq
            start = data.index.values[0]
            stop = data.index.values[-1]
            cr = sweep_response[start:stop][np.array(range(number_cells)).astype(str)].values # cell responses
            cr = cr.mean(axis=0) # find average response over all trials for condition
            cr = [cr[i][int(intersweep):int(intersweep+sweeplength)] for i in range(len(cr))] # restrict to stimulus presentation times
            conditions[oi,ti] = cr
    print "done with averaging"

    # compute two-way Kolmogorov-Smirnov test on blank vs. stimulus trial for each cell for each sweep to determine significance
    # find lifetime sparsenesses of signficantly responding cells
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

    print "found sig rois"
    # find lifetime and population sparsenesses
    pop, life = sp.sparseness(intersweep, sweeplength, orivals, tfvals, number_cells, stimulus_table, sweep_response)
    sig_life_sparseness = np.asarray([life[int(i)] for i in sig_rois]) # numpy array of lifetime sparsenesses of significantly responding cells
    life_spar_list.append(sig_life_sparseness)
    pop_spar_list.append(np.asarray(pop))
    print "found lifetime and population sparsenesses"



# plot results
print "plotting results"
sbplts = np.zeros((4,5))

f1, axarr1 = plt.subplots(4, 5)
for (x,y), data in np.ndenumerate(sbplts):
    li = np.ravel_multi_index((x,y), dims=(4,5), order='C')
    axarr1[x, y].hist(life_spar_list[li],bins=NBINS,normed=1,histtype='bar',rwidth=0.8)

f2, axarr2 = plt.subplots(4, 5)
for (x,y), data in np.ndenumerate(sbplts):
    li = np.ravel_multi_index((x,y), dims=(4,5), order='C')
    axarr2[x, y].hist(pop_spar_list[li],bins=NBINS,normed=1,histtype='bar',rwidth=0.8)

plt.show()


