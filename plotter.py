import cPickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
lsp = open('names_dict', 'rb')
names_dict = cPickle.load(lsp)

NBINS = 20


lfts_list = names_dict['lift_spar_list']
pops_list = names_dict['pop_spar_list']

# store mean, median, std of each lft sp value for each exp in a list
lfts_params = []
for exp in range(len(lfts_list)):
    mean_lfts = np.mean(lfts_list[exp])
    median_lfts = np.median(lfts_list[exp])
    std_lfts = np.std(lfts_list[exp])
    lfts_param = (mean_lfts, median_lfts, std_lfts)
    lfts_params.append(lfts_param)

# plot lfts results
plt.close('all')

# plot life time sparseness distribution for each experiment
figure, axarr = plt.subplots(nrows=5, ncols=4, figsize=(9,9))
figure.canvas.set_window_title('Lifetime sparseness')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for li, ax in enumerate(axarr.flatten()):
    ax.hist(lfts_list[li],bins=NBINS,normed=1,histtype='bar',rwidth=0.8, color='c')
    title = names_dict['meta_list'][li]['HVA'] + ' ' + names_dict['meta_list'][li]['depth'] + \
        ' ' + names_dict['meta_list'][li]['Cre']
    ax.set_title(title, fontsize=10)
    #ax.vlines(median_lfts[li]), 0, lift_spar_list[li].max(), color='r', lw=1.5, linestyle='dashed')
plt.xlabel('Lifetime Sparseness')
plt.ylabel('Count')
plt.show()

# plot population sparseness distribution for each experiment
figure, axarr = plt.subplots(nrows=5, ncols=4, figsize=(9,9))
figure.canvas.set_window_title('Population sparseness')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for li, ax in enumerate(axarr.flatten()):
    ax.hist(pops_list[li],bins=NBINS,normed=1,histtype='bar',rwidth=0.8, color='b')
    title = names_dict['meta_list'][li]['HVA'] + ' ' + names_dict['meta_list'][li]['depth'] + \
        ' ' + names_dict['meta_list'][li]['Cre']
    ax.set_title(title, fontsize=10)
    #ax.vlines(median_lfts[li]), 0, lift_spar_list[li].max(), color='r', lw=1.5, linestyle='dashed')
plt.xlabel('Population Sparseness')
plt.ylabel('Count')
plt.show()
