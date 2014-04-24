#coding:latin_1
import math
import timeit
import matplotlib.pyplot as plt
from mpltools import style

from Functions import *
from Classes import *

style.use('ggplot')

#--------------------------------------------------------------------------------------------------------------#
#---------------------------------------------   1. CHARGING DATA   -------------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

data = np.load("data_cond1.npy")
times = np.load("times_cond1.npy")

# Choosing the range of sensors.
first = 0
last = 203
y = data[first:last, :].T

# - Signal before the experiment.
y_early = data[first:last, :180].T

# - Signal during the experiment.
y_exp = data[first:last, 180:180+256].T

# - Pre-experiment variance.
std = np.zeros(y_early.shape[1])
for i in xrange(y_early.shape[1]):
    std[i] = math.sqrt(np.var(y_early[:, i]))
"""
#--------------------------------------------------------------------------------------------------------------#
#-----------------------------------------   2. DICTIONARY PARAMETERS   ---------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

wave_name = 'db6'
wave_level = None
PhiT = DictT(level=wave_level, name=wave_name)
Phi = Dict(sizes=PhiT.sizes, name=PhiT.name)

BasisT = PhiT.dot(np.identity(y_exp.shape[0]))
BasisT /= np.sqrt(np.sum(BasisT ** 2, axis=0))

#--------------------------------------------------------------------------------------------------------------#
#-------------------------------------------   3. MULTI-CHANNEL MP   ------------------------------------------#
#--------------------------------------------------------------------------------------------------------------#
start = timeit.default_timer()

x, z, err, n = multi_channel_mp(BasisT.T, BasisT, y_exp, std)

stop = timeit.default_timer()
print str(stop - start)

x_1, z_1, err_1, n_1 = multi_channel_omp(BasisT.T, BasisT, y_exp, std)

fig = plt.figure()

sen1 = 0
sen2 = 2
sen3 = 4
sen4 = 6

ax1 = fig.add_subplot(221, xlabel='Time (ms)', ylabel='MEG')
ax1.plot(y_exp[:, sen1], '#585858', label='Noisy signal - sensor ' + str(first+sen1))
ax1.plot(z[:, sen1], '#045FB4', lw=2, label='Denoised: MP :' + str(len(n)) + " atoms.")
plt.legend(loc='upper right', fontsize=11).get_frame().set_alpha(0)
x1, x2, y1, y2 = plt.axis()
plt.axis((0, 256, y1, y2))

ax1 = fig.add_subplot(222, xlabel='Time (ms)', ylabel='MEG')
ax1.plot(y_exp[:, sen2], '#585858', label='Noisy signal - sensor ' + str(first+sen2))
ax1.plot(z[:, sen2], '#0174DF', lw=2, label='Denoised: MP :' + str(len(n)) + " atoms.")
plt.legend(loc='upper right', fontsize=11).get_frame().set_alpha(0)
x1, x2, y1, y2 = plt.axis()
plt.axis((0, 256, y1, y2))

ax1 = fig.add_subplot(223, xlabel='Time (ms)', ylabel='MEG')
ax1.plot(y_exp[:, sen3], '#585858', label='Noisy signal - sensor ' + str(first+sen3))
ax1.plot(z[:, sen3], '#0080FF', lw=2, label='Denoised: MP :' + str(len(n)) + " atoms.")
plt.legend(loc='upper right', fontsize=11).get_frame().set_alpha(0)
x1, x2, y1, y2 = plt.axis()
plt.axis((0, 256, y1, y2))

ax1 = fig.add_subplot(224, xlabel='Time (ms)', ylabel='MEG')
ax1.plot(y_exp[:, sen4], '#585858', label='Noisy signal - sensor ' + str(first+sen4))
ax1.plot(z[:, sen4], '#2E9AFE', lw=2, label='Denoised: MP :' + str(len(n)) + " atoms.")
plt.legend(loc='upper right', fontsize=11).get_frame().set_alpha(0)
x1, x2, y1, y2 = plt.axis()
plt.axis((0, 256, y1, y2))

plt.tight_layout()
"""
#--------------------------------------------------------------------------------------------------------------#
#-----------------------------------------   4. DICTIONARY SELECTION   ----------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

cross_val, cross_indexes = cross_validation(y_exp, std, 'db')
bic_val, bic_indexes = bic_criterion(y_exp, std, 'db', ortho='yes')

fig = plt.figure()
ax1 = fig.add_subplot(211, ylabel='Cross-validation results')
ax2 = fig.add_subplot(212, ylabel='BIC results', xlabel='Daubechies wavelets')

ax1.plot(cross_indexes, cross_val, '-o', c='#DF013A', lw=2)
ax1.fill_between(cross_indexes, cross_val,
                 where=cross_val >= min(cross_val), interpolate=True, color="#F7819F")
ax1.set_xticks(cross_indexes)
ax1.axis((cross_indexes[0], cross_indexes[-1], 0.95*min(cross_val), 1.05*max(cross_val)))

ax2.plot(bic_indexes, bic_val, '-o', c='#084B8A', lw=2)
ax2.fill_between(bic_indexes, bic_val,
                 where=bic_val >= min(bic_val), interpolate=True, color="#5882FA")
ax2.set_xticks(bic_indexes)
ax2.axis((bic_indexes[0], bic_indexes[-1], 0.95*min(bic_val), 1.05*max(bic_val)))

plt.tight_layout()
#plt.savefig("Prints/Cross_validation.pdf", bbox_inches='tight')

plt.show()