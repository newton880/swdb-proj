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
    N = len(stimulus_table[(stimulus_table['temporal_frequency']==t1)&(stimulus_table['orientation']==o1)]) # number of trials per condition
    orivals = stimulus_table.orientation.dropna().unique() # find unique orientation values
    tfvals = stimulus_table.temporal_frequency.dropna().unique() # find unique temporal frequency values
    K = len(orivals)*len(tfvals)+1 # number of experimental conditions (plus one blank)
    P = number_cells # number of cells (pixels) in dataset
    Tf = sweeplength

    # get blank trials
    blanks = sweep_response[stimulus_table['blank_sweep']==1] # blank stimulus trials
    num_blanks = len(blanks) # number of blank trials

    # construct data matrix T (averaged over trials)
    M = np.zeros(((K-1)*N+num_blanks, P*Tf)) # full data matrix (non-blank trials plus number of blank trials)
    T = np.zeros((K, P*Tf)) # trial-averaged data
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
                M[li*N:(li+1)*N, cell_index*Tf:(cell_index+1)*Tf] = trial_arr[:, intersweep:intersweep+Tf] #
                T[li, cell_index*Tf:(cell_index+1)*Tf] = trial_arr[:, intersweep:intersweep+Tf].mean() # take mean over trials
                li += 1 # update linear index across conditions
        this_blank = np.array(blanks[str(cell_index)].tolist())
        M[(K-1)*N:(K-1)*N+num_blanks, cell_index*Tf:(cell_index+1)*Tf] = this_blank[:, intersweep:intersweep+Tf]
        T[K-1, cell_index*Tf:(cell_index+1)*Tf] = this_blank[:, intersweep:intersweep+Tf].mean() # append blank trial average

    # perform singular value decomposition and calculate brain scores
    T = np.matrix(T)
    v = np.matrix(np.ones((1,K)))
    vT = np.transpose(v)
    T = T-np.dot(vT, np.dot(v, T)/float(K))
    U,S,V = np.linalg.svd(np.transpose(T),full_matrices=False) # calculate svd
    B = np.dot(T,U)

    return U,S,V,B,M,N


def permuteTest(U, S, M, N, orivals, tfvals, nperms):

    # permute rows of data matrix M to test significance of effects represeted in LVs
    S = np.diag(S) # convert to matrix from vector
    K = len(orivals)*len(tfvals)+1 # number of experimental conditions (plus one blank)
    nk = M.shape[0] # number of rows in M
    ncols = M.shape[1]
    SMAT = np.zeros((S.shape[0],nperms)) # data structure to store singular values
    for i in range(nperms):
        idx = np.random.permutation(nk) # vector for randomly permuting rows of M
        Mp = M[idx, :] # permuted data matrix
        Tp = np.zeros((K, ncols)) # trial-averaged permuted data matrix
        li = 0 # linear index over conditions
        for oi in range(len(orivals)):
            for ti in range(len(tfvals)):
                Tp[li, :] = Mp[li*N:(li+1)*N, :].mean(axis=0)
                li += 1 # update linear index
        Tp[K-1,:] = Mp[(li+1)*N:, :].mean(axis=0)

        # perform singular value decomposition
        Tp = np.matrix(Tp)
        v = np.matrix(np.ones((1,K)))
        vT = np.transpose(v)
        Tp = Tp-np.dot(vT, np.dot(v, Tp)/float(K))
        Up,Sp,_ = np.linalg.svd(np.transpose(Tp),full_matrices=False) # calculate svd
        Sp = np.diag(Sp) # convert to matrix from vector

        # perform Procrustes rotation
        Wp = np.dot(Up,Sp) # permuted principal coordinates
        W0 = np.dot(U,S) # original principal coordinates
        Um,_,Vm = np.linalg.svd(np.dot(np.transpose(Wp),W0),full_matrices=False)
        Q = np.dot(Um,np.transpose(Vm))
        Wp_ = np.dot(Wp,Q) # rotate permuted principal scores
        Sp_ = np.sqrt(np.sum(np.square(Wp_),axis=0)) # rotated singular values
        SMAT[:,i] = Sp_ # store results

    # compare results of permutations to original data and calculate p-values
    S = np.diag(S)
    mask = np.zeros((SMAT.shape))
    for i in range(nperms):
        mask[:, i] = SMAT[:, i] >= S
    mask = np.sum(mask,axis=1)
    pvals = mask/float(nperms)

    return pvals


def bootstrap(niters, U):

    # permute rows of data matrix M to test significance of effects represeted in LVs
    return 0

