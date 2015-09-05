# authors: Taylor Newton and Daniela Saderi
# date: 27.8.2015
# institute: Allen Institute for Brain Science 2015 Summer Workshop on the Dynamic Brain
# description: this script extracts visually "responsive" ROIs from Ca2+ imaging data
# provided by the Allen Institue (TODO: flesh out file explanation).


# library imports
import numpy as np
import cPickle
import CAM_NWB as cn
import sparseness as sp
import process as pr
import pls


# constants
CAMDIR = r'/Volumes/Brain2015/CAM/'
PTHRESH = 0.05 # p-value cutoff above which results are not significant
NBINS = 20 # number of bins for histogram plot
DF_THRESH = 15 # percent threshold (dF/F) for cutting off non-responsive cells
NPERMS = 100 # number of permutations to use for permutation test of significance for LVs
NBOOTS = 100 # number of iterations to perform for bootsrapping procedure for salience stabilities


# get experiment data and path to experiments
path_list, CAM_df = pr.makeDF(CAMDIR)

# LOOP OVER ALL EXPERIMENTS AND CALCULATE HISTOGRAM OF LIFETIME SPARSENESSES FOR EACH
life_spar_list = []
pop_spar_list = []
meta_list = []
pval_list = []
sal_list = []
for path in path_list:

    # extract interesting information
    meta = cn.getMetaData(path) # get meta data for experiment of interest
    proj = cn.getMaxProjection(path) # getMaxProjection returns a 512x512 array of the maximum projection of the 2P movie
    print meta
    meta_list.append(meta)

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

    # create data frame with dF/F trace of each ROI in response to each grating
    sweep_response = pr.getResponses(stimulus_table, celltraces, number_cells, intersweep, sweeplength)

    # find blank stimulus presentations
    blank = stimulus_table[stimulus_table['blank_sweep']==1]
    print "Number of blank trials: ", len(blank)

    # find visually "responsive" neurons: those whose response was significantly different from blank to at
    # least one condition (see Rust & DiCarlo, 2012), and whose dF/F is > DF_THRESH.

    # begin by averaging data for blank stimulus trials
    blanks = sweep_response[stimulus_table['blank_sweep']==1].values # blank stimulus trials
    blanks = blanks.mean(axis=0) # get average response over all trials

    # now average cell response data for stimulus trials
    conditions = pr.getConditions(stimulus_table, sweep_response, intersweep, sweeplength, orivals, tfvals, number_cells)
    print "done with averaging"

    # compute two-way Kolmogorov-Smirnov test on blank vs. stimulus trial for each cell for each sweep to determine significance
    # find lifetime sparsenesses of signficantly responding cells
    sig_rois = pr.findSigROIs(conditions, orivals, tfvals, blanks, PTHRESH)
    print "found sig rois"

    # find lifetime and population sparsenesses
    pop, life = sp.sparseness(intersweep, sweeplength, orivals, tfvals, number_cells, stimulus_table, sweep_response)
    sig_life_sparseness = np.asarray([life[int(i)] for i in sig_rois]) # numpy array of lifetime sparsenesses of significantly responding cells
    life_spar_list.append(zip(sig_rois, sig_life_sparseness))
    pop_spar_list.append(np.asarray(pop))
    print "found lifetime and population sparsenesses"

    # compute partial least squares and p-values corresponding to LVs
    U,S,V,B,M,N = pls.pls(stimulus_table, sweep_response, number_cells, intersweep, sweeplength)
    pvals = pls.permuteTest(U, S, M, N, orivals, tfvals, NPERMS)
    pval_list.append(pvals)

    # find significant cell saliences, and save them
    Usig = pls.bootstrap(NBOOTS, N, M, U, S, orivals, tfvals)
    U1 = Usig[:,0] # cell saliences of first principal component
    saliences = np.zeros((number_cells, 1))
    for cell_index in range(number_cells):
        x = U1[cell_index*sweeplength:(cell_index+1)*sweeplength]
        saliences[cell_index] = max(x.min(), x.max(), key=abs) # take maximum of saliences
    sal_list.append(saliences) # append results

# Save the variables in a pickle file. Plot data in plot_sparseness.py
names_dict = dict(lift_spar_list=life_spar_list, pop_spar_list=pop_spar_list, meta_list=meta_list, pval_list=pval_list, sal_list=sal_list)
pkl_file = open('names_dict','wb')
cPickle.dump(names_dict, pkl_file)
pkl_file.close()


