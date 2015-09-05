import cPickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.linalg import norm
from scipy.spatial.distance import euclidean

BINS = np.arange(0,1.2,0.05)  # number of bins for histogram plot
XLIM = [0, 1.2]
lsp = open('names_dict', 'rb')
names_dict = cPickle.load(lsp)

lfts_list_tuples = names_dict['lift_spar_list']
lfts_list = []
for list in lfts_list_tuples:
    lfts_list.append([var[1] for var in list])

pops_list = names_dict['pop_spar_list']

# store mean, median, std of each lft sp value for each exp in a list
lfts_params = []
lfts_median_list = []
for exp in range(len(lfts_list)):
    mean_lfts = np.mean(lfts_list[exp])
    median_lfts = np.median(lfts_list[exp])
    lfts_median_list.append(median_lfts)
    std_lfts = np.std(lfts_list[exp])
    lfts_param = (mean_lfts, median_lfts, std_lfts)
    lfts_params.append(lfts_param)

pops_params = []
pops_median_list = []
for exp in range(len(pops_list)):
    mean_pops = np.mean(pops_list[exp])
    median_pops = np.median(pops_list[exp])
    pops_median_list.append(median_pops)
    std_pops = np.std(pops_list[exp])
    skw_pops = 3*((mean_pops-median_pops)/std_pops)
    pops_param = (mean_pops, median_pops, std_pops)
    pops_params.append(pops_param)

# plot lfts results
plt.close('all')

# plot life time sparseness distribution for each experiment
figure, axarr = plt.subplots(nrows=5, ncols=4, figsize=(9,9))
figure.canvas.set_window_title('Lifetime sparseness')
plt.title('Lifetime sparseness')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for li, ax in enumerate(axarr.flatten()):
    ax.hist(lfts_list[li], bins=BINS, normed=True, histtype='bar', color='c',rwidth=0.8)
    title = names_dict['meta_list'][li]['HVA'] + ' ' + names_dict['meta_list'][li]['depth'] + \
        ' ' + names_dict['meta_list'][li]['Cre']
    ax.set_title(title, fontsize=10)
    ax.locator_params(axis='x', nbins=5)
    ax.set_xlim(XLIM)
    ax.axvline(lfts_median_list[li], color='r', lw=1.5, linestyle='dashed')
plt.xlabel('Sparseness')
plt.ylabel('Count')
figure.savefig('life.png')

# plot population sparseness distribution for each experiment
figure, axarr = plt.subplots(nrows=5, ncols=4, figsize=(9,9))
figure.canvas.set_window_title('Population sparseness')
plt.title('Population sparseness')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for li, ax in enumerate(axarr.flatten()):
    ax.hist(pops_list[li], bins=BINS, normed=True, histtype='bar', color='b', rwidth=0.8)
    title = names_dict['meta_list'][li]['HVA'] + ' ' + names_dict['meta_list'][li]['depth'] + \
        ' ' + names_dict['meta_list'][li]['Cre']
    ax.set_title(title, fontsize=10)
    ax.locator_params(axis='x', nbins=5)
    ax.set_xlim(XLIM)
    ax.axvline(pops_median_list[li], color='r', lw=1.5, linestyle='dashed')
plt.xlabel('Sparseness')
plt.ylabel('Count')
figure.savefig('pops.png')

# plot the lfts and pops on the same plots
figure, axarr = plt.subplots(nrows=5, ncols=4, figsize=(9,9))
figure.canvas.set_window_title('Lifetime and population sparseness')
plt.title('Lifetime and population sparseness')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for li, ax in enumerate(axarr.flatten()):
    ax.hist(lfts_list[li], BINS, normed=True, histtype='bar', color='c', alpha=0.5, rwidth=0.8,label='LFTS')
    ax.hist(pops_list[li], BINS, normed=True, histtype='bar', color='b', alpha=0.5, rwidth=0.8,label='POPS')
    title = names_dict['meta_list'][li]['HVA'] + ' ' + names_dict['meta_list'][li]['depth'] + \
        ' ' + names_dict['meta_list'][li]['Cre']
    ax.set_title(title, fontsize=10)
    ax.locator_params(axis='x', nbins=5)
    ax.set_xlim(XLIM)
    ax.axvline(lfts_median_list[li], color='c', lw=1.5, linestyle='dashed')
    ax.axvline(pops_median_list[li], color='b', lw=1.5, linestyle='dashed')
plt.legend(loc='upper left', prop={'size':'xx-small'})
plt.xlabel('Sparseness')
plt.ylabel('Count')
plt.show()
figure.savefig('lfts_pops.png')

# Measure distance between lifetime and population sparseness
def hellinger(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2

_SQRT2 = np.sqrt(2)

fig = plt.figure()
distances = []
for lfts, pops in zip(lfts_list, pops_list):
    lfts_hist, _ = np.histogram(lfts, BINS, normed=True, range=XLIM)
    pops_hist, _ = np.histogram(pops, BINS, normed=True, range=XLIM)
    dist = hellinger(norm(lfts_hist), norm(pops_hist))
    distances.append(dist)
plt.hist(distances, BINS, normed=True, histtype='bar', color='b', rwidth=1)
figure.canvas.set_window_title('Hellinger distance between lifetime and population sparseness')
plt.title('Hellinger distance between lifetime and population sparseness')
plt.axis([0,1,0,3], fontsize=20)
plt.xlabel('Hellinger distance')
plt.ylabel('Count')
fig.savefig('hellinger.png')
plt.show()

# scatter plot of lifetime sparseness and cell salience correlations
sal_list = names_dict['sal_list']
figure, axarr = plt.subplots(nrows=5, ncols=4, figsize=(9,9))
figure.canvas.set_window_title('Cell salience vs. lifetime sparseness')
plt.title('Cell salience vs. lifetime sparseness')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for li, ax in enumerate(axarr.flatten()):
    lf = zip(*lfts_list_tuples[li])
    sal = np.asarray(sal_list[li])
    idx = np.asarray(lf[0], dtype=np.int64)
    lfts = np.asarray(lf[1], dtype=np.float32)
    ax.scatter(sal[idx], lfts)
    title = names_dict['meta_list'][li]['HVA'] + ' ' + names_dict['meta_list'][li]['depth'] + \
        ' ' + names_dict['meta_list'][li]['Cre']
    ax.set_title(title, fontsize=10)
    # ax.locator_params(axis='x', nbins=5)
plt.legend(loc='upper left', prop={'size':'xx-small'})
plt.xlabel('Salience')
plt.ylabel('Lifetime sparseness')
plt.show()
figure.savefig('sal_lfts.png')
