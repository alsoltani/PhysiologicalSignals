#coding:latin_1

import math
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

#--------------------------------------------------------------------------------------------------------------#
#-----------------------------------------   2. DICTIONARY PARAMETERS   ---------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

wave_name = 'db17'
wave_level = None
PhiT = DictT(level=wave_level, name=wave_name)
Phi = Dict(sizes=PhiT.sizes, name=PhiT.name)

BasisT = PhiT.dot(np.identity(y_exp.shape[0]))
BasisT /= np.sqrt(np.sum(BasisT ** 2, axis=0))

#--------------------------------------------------------------------------------------------------------------#
#-------------------------------------------   3. MULTI-CHANNEL MP   ------------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

x, z, err, n = multi_channel_mp(BasisT.T, BasisT, y_exp, std)

x_1, z_1, err_1, n_1 = multi_channel_omp(BasisT.T, BasisT, y_exp, std)

fig = plt.figure()

sen1 = 54
sen2 = 55
sen3 = 56
sen4 = 57

ax1 = fig.add_subplot(221, xlabel='Time (ms)', ylabel='MEG (T)')
ax1.plot(y_exp[:, sen1], '#FA5882', label='Noisy - sensor ' + str(first+sen1))
ax1.plot(z[:, sen1], '#045FB4', lw=2, label='Denoised: ' + str(len(n)) + " atoms.")
plt.legend(loc='lower right', fontsize=10).get_frame().set_alpha(0)
x1, x2, y1, y2 = plt.axis()
plt.axis((0, 256, y1, y2))

ax1 = fig.add_subplot(222, xlabel='Time (ms)', ylabel='MEG (T)')
ax1.plot(y_exp[:, sen2], '#FA5882', label='Noisy - sensor ' + str(first+sen2))
ax1.plot(z[:, sen2], '#045FB4', lw=2, label='Denoised: ' + str(len(n)) + " atoms.")
plt.legend(loc='lower right', fontsize=10).get_frame().set_alpha(0)
x1, x2, y1, y2 = plt.axis()
plt.axis((0, 256, y1, y2))

ax1 = fig.add_subplot(223, xlabel='Time (ms)', ylabel='MEG (T)')
ax1.plot(y_exp[:, sen3], '#FA5882', label='Noisy - sensor ' + str(first+sen3))
ax1.plot(z[:, sen3], '#045FB4', lw=2, label='Denoised: ' + str(len(n)) + " atoms.")
plt.legend(loc='lower right', fontsize=10).get_frame().set_alpha(0)
x1, x2, y1, y2 = plt.axis()
plt.axis((0, 256, y1, y2))

ax1 = fig.add_subplot(224, xlabel='Time (ms)', ylabel='MEG (T)')
ax1.plot(y_exp[:, sen4], '#FA5882', label='Noisy - sensor ' + str(first+sen4))
ax1.plot(z[:, sen4], '#045FB4', lw=2, label='Denoised: ' + str(len(n)) + " atoms.")
plt.legend(loc='lower right', fontsize=10).get_frame().set_alpha(0)
x1, x2, y1, y2 = plt.axis()
plt.axis((0, 256, y1, y2))

plt.tight_layout()

#plt.savefig("Prints/MOMP_2.pdf", bbox_inches='tight')

#--------------------------------------------------------------------------------------------------------------#
#-----------------------------------------   4. DICTIONARY SELECTION   ----------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

cross_val, cross_indexes = cross_validation(y_exp, std, 'db')
aicc_val, aicc_indexes = aicc_criterion(y_exp, std, 'db', ortho='yes')

# As we only seek for the difference in AICC values, we add a constant to plot positive values.
aicc_val_relative = map(lambda t: t+15000, aicc_val)

r_sq, ajus_r_sq, r_sq_indexes = r_square(y_exp, std, 'db', ortho='yes')

fig = plt.figure()
ax = fig.add_subplot(211, ylabel='Cross-validation results', xlabel='Daubechies wavelets')
ax.plot(cross_indexes, cross_val, '-o', c='#DF013A', lw=2)
ax.fill_between(cross_indexes, cross_val,
                where=cross_val >= min(cross_val), interpolate=True, color="#F7819F")
ax.set_xticks(cross_indexes)
ax.axis((cross_indexes[0], cross_indexes[-1], 0.95*min(cross_val), 1.05*max(cross_val)))
plt.tight_layout()

#plt.savefig("Prints/Cross_validation.pdf", bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(212, ylabel='AICC results', xlabel='Daubechies wavelets')
ax.plot(aicc_indexes, aicc_val_relative, '-o', c='#084B8A', lw=2)
ax.fill_between(aicc_indexes, aicc_val_relative,
                where=aicc_val_relative >= min(aicc_val_relative), interpolate=True, color="#5882FA")
ax.set_xticks(aicc_indexes)
ax.axis((aicc_indexes[0], aicc_indexes[-1], min(aicc_val_relative), max(aicc_val_relative)))
plt.tight_layout()

#plt.savefig("Prints/AICC_results.pdf", bbox_inches='tight')

fig = plt.figure()
ax1 = fig.add_subplot(211, ylabel='R2')
ax1.bar(r_sq_indexes, r_sq, color='#FA5858')
ax1.set_xticks(r_sq_indexes)
ax1.axis((r_sq_indexes[0], r_sq_indexes[-1], 0.995*min(r_sq), 1.005*max(r_sq)))

ax2 = fig.add_subplot(212, ylabel='Ajusted R2', xlabel='Daubechies wavelets')
ax2.bar(r_sq_indexes, ajus_r_sq)
ax2.set_xticks(r_sq_indexes)
ax2.axis((r_sq_indexes[0], r_sq_indexes[-1], 0.995*min(ajus_r_sq), 1.005*max(ajus_r_sq)))
plt.tight_layout()

#plt.savefig("Prints/R2_results.pdf", bbox_inches='tight')


#--------------------------------------------------------------------------------------------------------------#
#----------------------------------------------   5. K-SVD METHOD   -------------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

n_iter = 8
sensor = 100

updated_basis, updated_sparse, updated_err, updated_atoms = k_svd(BasisT.T, y_exp, std, multi_channel_omp, n_iter)

fig = plt.figure()
ax2 = fig.add_subplot(212, xlabel='Atoms included', ylabel='Log-Error')
for i in xrange(n_iter):
    ax2.plot(map(math.log, updated_err[i]), lw=2,
             label='Iter. '+str(i+1) + ' : ' + str(len(updated_atoms[i]))+' atoms.')
plt.legend(loc='upper right', fontsize=11, ncol=2).get_frame().set_alpha(0)

ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG (T)')
ax1.plot(y_exp[:, sensor], '#A4A4A4', label='Noisy')
ax1.plot(updated_basis.dot(updated_sparse)[:, sensor], '#FAAA18', lw=2,
         label='Denoised, K-SVD : ' + str(len(updated_atoms[-1])) + ' atoms.')
ax1.plot(z_1[:, sensor], '#9508A4', ls='--',  lw=3, label='Denoised,  OMP : ' + str(len(n_1)) + ' atoms.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)

x_a, x_b, y_a, y_b = ax1.axis()
ax1.axis((0, 256, y_a, y_b))

plt.tight_layout()
#plt.savefig("Prints/KSVD.pdf", bbox_inches='tight')

plt.show()