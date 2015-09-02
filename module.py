import numpy as np

# returns lifetime sparseness for each cell in a given experimental trial (8 orientations by 5 temporal frequencies)
def lifetimeSparseness(intersweep, sweeplength, orivals, tfvals, num_cells, stimulus_table, sweep_response):
    #calculate lifetime sparsensess
    numerator = np.zeros(num_cells)
    denominator = np.zeros(num_cells)

    trial_start, trial_end = (intersweep, intersweep + sweeplength)
    N = len(orivals) * len(tfvals)
    # Loop over cells
    for cell_index in range(num_cells):
        #get time and trial average for each stimulus condition
        # Loop over orientations and times
        cell_response = []
        for ori in orivals:
            for tf in tfvals:
                tf_mask = stimulus_table['temporal_frequency'] == tf
                ori_mask = stimulus_table['orientation'] == ori
                trial_arr = np.array(sweep_response[tf_mask & ori_mask][str(cell_index)].tolist()) #TODO clean this up
                if not np.all(np.isfinite(trial_arr)):
                    print("trial error: excluding data for orientation = %f degrees, temporal frequency = %f" % (ori,tf))
                    continue

                trial_mean = trial_arr[:, trial_start:trial_end].mean()
                cell_response.append(trial_mean)

        numerator[cell_index] = np.array(cell_response).mean() ** 2
        denominator[cell_index] = np.sum((np.array(cell_response) ** 2 / N))

    a_list = numerator/denominator
    assert np.all(a_list < 1.0),"something's wrong..."
    sparseness_list = (1.0-a_list)/(1.0-(1.0/N))

    return sparseness_list
