# library imports
import numpy as np


# calculates population and lifetime sparseness
def sparseness(intersweep, sweeplength, orivals, tfvals, num_cells, stimulus_table, sweep_response):

    #calculate population sparseness. Same equation as lft sparseness but now ri is the response of the ith
    #neuron to a given stimulus averaged across trials and N is the number of cells in the exp.
    popnum = np.zeros(len(orivals)*len(tfvals))
    popden = np.zeros(len(orivals)*len(tfvals))

    lifenum = np.zeros(num_cells) # for lifetime sparseness of individual cells
    lifeden = np.zeros(num_cells)

    trial_start, trial_end = (intersweep, intersweep + sweeplength)

    # Loop over stimulus condition
    stimulus_index = 0
    for ori in orivals:
        for tf in tfvals:
            pop_response = [] # for population sparseness
            tf_mask = stimulus_table['temporal_frequency'] == tf
            ori_mask = stimulus_table['orientation'] == ori
            this_resp = sweep_response[tf_mask & ori_mask]
            # Loop over cells
            for cell_index in range(num_cells):
                trial_arr = np.array(this_resp[str(cell_index)].tolist())
                lens = [len(trial) for trial in trial_arr]
                if min(lens) < max(lens): # justify trial lengths if one or more has a different length
                    trial_arr = np.array([trial[0:min(lens)] for trial in trial_arr])
                if not np.all(np.isfinite(trial_arr)):
                    print("trial error (inf or NaN): excluding data for orientation = %f degrees, temporal frequency = %f" % (ori,tf))
                    continue

                trial_mean = trial_arr[:, trial_start:min(trial_end,min(lens))].mean()
                pop_response.append(trial_mean)

                lifenum[cell_index] += trial_mean
                lifeden[cell_index] += trial_mean ** 2

            popnum[stimulus_index] = np.array(pop_response).mean() ** 2
            popden[stimulus_index] = np.sum((np.array(pop_response) ** 2 / num_cells))
            stimulus_index += 1

    lifenum = (lifenum / (len(orivals)*len(tfvals))) ** 2
    lifeden = lifeden / (len(orivals)*len(tfvals))

    pop_a_list = popnum/popden
    assert np.all(pop_a_list < 1.0),"something's wrong..."
    pop_sparseness = (1.0 - pop_a_list)/(1.0-(1.0/num_cells))

    life_a_list = lifenum/lifeden
    assert np.all(life_a_list < 1.0),"something's wrong..."
    life_sparseness = (1.0-life_a_list)/(1.0-(1.0/(len(orivals)*len(tfvals))))

    return pop_sparseness, life_sparseness