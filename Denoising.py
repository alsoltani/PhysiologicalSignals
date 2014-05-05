#coding:latin_1
import math
import statsmodels.api as sm
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

y = data[100, :]

# Signal before the experiment.
y_early = y[:180]

# Signal during the experiment.
y_exp = y[180:180+256]

std = math.sqrt(np.var(y_early))


#--------------------------------------------------------------------------------------------------------------#
#------------------------------------------   2. DICTIONARY PARAMETERS   --------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

wave_name = 'db17'
wave_level = None
wavelet_operator_t = DictT(level=wave_level, name=wave_name)

basis_t = wavelet_operator_t.dot(np.identity(y_exp.size))
basis_t /= np.sqrt(np.sum(basis_t ** 2, axis=0))

#--------------------------------------------------------------------------------------------------------------#
#-------------------------   3. COMPARING MP RESULTS, W/ & W/O STATISTICAL CRITERION   ------------------------#
#--------------------------------------------------------------------------------------------------------------#

x_mp_arb, z_mp_arb, err_mp_arb, n_mp_arb = mp_arbitrary_criterion(basis_t.T, basis_t, y_exp, 15)

x_mp_stat, z_mp_stat, err_mp_stat, n_mp_stat = mp_stat_criterion(basis_t.T, basis_t, y_exp, std)

fig = plt.figure()
ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG (T)')
ax1.plot(y_exp, '#775A5A', label='Noisy signal - during experiment')
ax1.plot(z_mp_stat, '#FD1528', lw=2, label='Denoised: MP, statistical criterion')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])
plt.tight_layout()

#plt.savefig("Prints/1_MP_arbitrary_and_criterion_pt1.pdf", bbox_inches='tight')

fig = plt.figure()
ax2 = fig.add_subplot(223, xlabel='Time (ms)', ylabel='MEG (T)')
ax2.plot(z_mp_stat, '#FD1528', lw=2, label='Denoised: stat. criterion : ' + str(len(n_mp_stat)) + ' atoms.')
ax2.plot(z_mp_arb, '#000000', ls='--', lw=3,
         label='Denoised: no criterion : ' + str(len(n_mp_arb)) + ' atoms.')
plt.legend(loc='upper left', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])
ax2 = fig.add_subplot(224, xlabel='Time (ms)', ylabel='MEG (T)')
ax2.plot(y_exp-z_mp_stat, 'grey', lw=1,
         label='Residual: MP, stat. criterion.')
plt.legend(loc='upper right', fontsize=11).get_frame().set_alpha(0)
plt.axis([64, 192, -5e-12, 5e-12])
plt.tight_layout()

#plt.savefig("Prints/1_MP_arbitrary_and_criterion_pt2.pdf", bbox_inches='tight')

#--------------------------------------------------------------------------------------------------------------#
#---------------------------------------   4. COMPARING MP & OMP RESULTS   ------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

x_omp_stat, z_omp_stat, err_omp_stat, n_omp_stat = omp_stat_criterion(basis_t.T, basis_t, y_exp, std)

fig = plt.figure()
ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG (T)')
ax1.plot(y_exp, '#929292', label='Noisy')
ax1.plot(z_omp_stat, '#009EFF', lw=2, label='Denoised: OMP : ' + str(len(n_omp_stat)) + ' atoms.')
ax1.plot(z_mp_stat, '#FF006F', ls='--', lw=3, label='Denoised: MP : ' + str(len(n_mp_stat)) + ' atoms.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])

ax2 = fig.add_subplot(212, xlabel='Atoms included', ylabel='Log-error')
ax2.plot(map(math.log, err_omp_stat), '#009EFF', lw=2, label='OMP')
ax2.plot(map(math.log, err_mp_stat), '#FF006F', ls='--', lw=3, label='MP')
plt.legend(loc='upper right', fontsize=11).get_frame().set_alpha(0)
plt.tight_layout()

#plt.savefig("Prints/2_MP_and_OMP.pdf", bbox_inches='tight')

#--------------------------------------------------------------------------------------------------------------#
#-----------------------------   5. COMPARING CLASSIC & ORTHONORMALIZED DICTIONARIES  -------------------------#
#--------------------------------------------------------------------------------------------------------------#

ortho_basis_t = np.linalg.qr(wavelet_operator_t.dot(np.identity(y_exp.size)).T)[0].T
ortho_basis_t /= np.sqrt(np.sum(ortho_basis_t ** 2, axis=0))

x_omp_gstat, z_omp_gstat, err_omp_gstat, n_omp_gstat = omp_stat_criterion(ortho_basis_t.T, ortho_basis_t, y_exp, std)

fig = plt.figure()
ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG (T)')
ax1.plot(y_exp, '#FA58AC', label='Noisy signal')
ax1.plot(z_omp_gstat, '#01DF74', linewidth=2,
         label='Denoised: OMP, orthonorm. dic : ' + str(len(n_omp_gstat)) + ' atoms.')
ax1.plot(z_omp_stat, '#0B0B3B', ls='--', lw=3, label='Denoised: OMP, classic dic : ' + str(len(n_omp_stat)) + ' atoms.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])

ax2 = fig.add_subplot(212, xlabel='Atoms included', ylabel='Log-Error')
ax2.plot(map(math.log, err_omp_gstat), '#01DF74', linewidth=2, label='OMP, orthonorm. dic')
ax2.plot(map(math.log, err_omp_stat), '#0B0B3B', ls='--', lw=3, label='OMP, classic dic')
plt.legend(loc='upper right', fontsize=11).get_frame().set_alpha(0)
plt.tight_layout()

#plt.savefig("Prints/3_Normal_and_Orthon_Dict.pdf", bbox_inches='tight')


#--------------------------------------------------------------------------------------------------------------#
#----------------------------------------   6. LOG-DECREMENTS RESULTS  ----------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

# Let us compare wavelet dictionaries within the Daubechies family,
# using classic and orthonormalized dictionaries.

fig = plt.figure()

#-----------------------------------------   A. CLASSIC DB DICTIONARY  ----------------------------------------#

ax1 = fig.add_subplot(211, xlabel='Atoms included', ylabel='Log-Error')
for i in xrange(1, 21):
    wavelet_op_t = DictT(level=None, name='db'+str(i))
    bas_t = wavelet_op_t.dot(np.identity(y_exp.size))
    bas_t /= np.sqrt(np.sum(bas_t ** 2, axis=0))

    X, Z, Error, N = omp_stat_criterion(bas_t.T, bas_t, y_exp, std)

    ax1.plot(map(math.log, Error), lw=2, label='db'+str(i))

plt.title('Log-Error decrement results - w/o Gram-Schmidt.')
plt.legend(loc='upper right', fontsize=11).get_frame().set_alpha(0)


#----------------------------------------- B. ORTHONORMAL DB DICTIONARY ---------------------------------------#

ax2 = fig.add_subplot(212, xlabel='Atoms included', ylabel='Log-Error')
for i in xrange(1, 21):
    wavelet_op_t = DictT(level=None, name='db'+str(i))
    bas_t = np.linalg.qr(wavelet_op_t.dot(np.identity(y_exp.size)).T)[0].T
    bas_t /= np.sqrt(np.sum(bas_t ** 2, axis=0))

    X, Z, Error, N = omp_stat_criterion(bas_t.T, bas_t, y_exp, std)

    ax2.plot(map(math.log, Error), lw=2, label='db'+str(i))

plt.title('Log-Error decrement results - w/ Gram-Schmidt.')
plt.legend(loc='upper right', fontsize=11).get_frame().set_alpha(0)
plt.tight_layout()

#plt.savefig("Prints/4_Log_Decrements.pdf", bbox_inches='tight')

#--------------------------------------------------------------------------------------------------------------#
#-----------------------------------   7. OMP, HARD THRESHOLDING COMPARISON  ----------------------------------#
#--------------------------------------------------------------------------------------------------------------#

# Gabriel Peyre's article states that MP /w can be equiv. to Hard Thresholding,
# when the decomposition basis is a square orthonormal matrix.
# Hard Thresholding has to verify same sparsity constraint as the Matching Pursuit solution.

z_thr, n_thr = hard_thresholding(ortho_basis_t.T, ortho_basis_t, y_exp, len(n_omp_gstat))

fig = plt.figure()
ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG (T)')
ax1.plot(y_exp, '#5882FA', label='Noisy signal')
ax1.plot(z_thr, '#FFBF00', lw=2, label='Denoised : Hard Thresholding : ' + str(len(n_thr)) + ' atoms.')
ax1.plot(z_omp_gstat, '#B40431', ls='--', lw=3,
         label='Denoised: OMP, orthonorm. dict : ' + str(len(n_omp_gstat)) + ' atoms.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])

plt.tight_layout()

#plt.savefig("Prints/5_Hard_Thresholding.pdf", bbox_inches='tight')


#--------------------------------------------------------------------------------------------------------------#
#-------------------------------------------   8. TESTING OUR MODEL  ------------------------------------------#
#--------------------------------------------------------------------------------------------------------------#


# ------------------------------------------   A.  NOISE SIMULATION   -----------------------------------------#

# Our idea is the following : we simulate a white noise, add it to the denoised signal
# and run the algorithms to find a new denoised signal. If the two signals match,
# the algorithm is considered as efficient.

gauss_noise = np.asarray(np.random.normal(0, pow(std, 2), y_exp.shape[0])).T
simu_noisy_signal = z_mp_stat + gauss_noise
x_simu, z_simu, err_simu, n_simu = mp_stat_criterion(basis_t.T, basis_t, simu_noisy_signal, std)

fig = plt.figure()

ax = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG (T)')
ax.plot(y_exp, '#E47777', label='Noisy')
ax.plot(z_simu, '#30B6F5', lw=2,
        label='Re-denoised: ' + str(len(n_simu)) + ' atoms.')
ax.plot(z_mp_stat, '#0E0846', ls='--', lw=3,
        label='Denoised: ' + str(len(n_mp_stat)) + ' atoms.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])

ax = fig.add_subplot(212, xlabel='Time (ms)', ylabel='MEG (T)')
ax.plot(z_simu-z_mp_stat, '#8F7979', lw=2,
        label='Difference between denoised signals.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])

plt.tight_layout()
#plt.savefig("Prints/6_Noise_Simulation.pdf", bbox_inches='tight')


# ------------------------------------------   B.  RESIDUAL TESTING   -----------------------------------------#
# In this section, we test the assumption under which our residuals are gaussian.
# We implement different tests to compare results.

# ----------------------------------------  B.1. Quantile-Quantile Plot  --------------------------------------#

fig = plt.figure()

for i in xrange(1, 4):
    ax = fig.add_subplot(220+i)
    wavelet_op_t = DictT(level=None, name='db'+str(2*i-1))
    bas_t = wavelet_op_t.dot(np.identity(y_exp.size))
    bas_t /= np.sqrt(np.sum(bas_t ** 2, axis=0))
    X, Z, Error, N = omp_stat_criterion(bas_t.T, bas_t, y_exp, std)

    sm.qqplot(y_exp-Z, line='s', ax=ax)

    txt = ax.text(-2.8, ax.get_ylim()[1] * 0.9, 'Daubechies '+str(2*i-1),
                  fontsize=12, color='#B40431', verticalalignment='top')
    txt.set_bbox(dict(facecolor='#F5A9A9', alpha=1))

ax = fig.add_subplot(224)
sm.qqplot(y_early, line='s', ax=ax)

txt = ax.text(-2.8, ax.get_ylim()[1] * 0.9, 'Pre-experiment signal',
              fontsize=12, color='#B40431', verticalalignment='top')
txt.set_bbox(dict(facecolor='#F5A9A9', alpha=1))
plt.tight_layout()

#plt.savefig("Prints/7_QQPlot.pdf", bbox_inches='tight')


# ---------------------------------------  B.2. Kolmogorov-Smirnoff Test  -------------------------------------#

print sm.stats.diagnostic.kstest_normal(y_early)

for i in xrange(1, 7):
    wavelet_op_t = DictT(level=5, name='db'+str(i))
    bas_t = wavelet_op_t.dot(np.identity(y_exp.size))
    bas_t /= np.sqrt(np.sum(bas_t ** 2, axis=0))

    X, Z, Error, N = omp_stat_criterion(bas_t.T, bas_t, y_exp, std)
    print sm.stats.diagnostic.kstest_normal(y_exp-Z)


#--------------------------------------------------------------------------------------------------------------#
#------------------------------------------   9. ENDOGENOUS VARIANCE  -----------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

# We refine our model by considering variance as an endogenous variable, using OLS estimator s².

x_mp_end, z_mp_end, err_mp_end, n_mp_end = mp_stat_endogen_var(basis_t.T, basis_t, y_exp)
x_omp_end, z_omp_end, err_omp_end, n_omp_end = omp_stat_endogen_var(basis_t.T, basis_t, y_exp)

fig = plt.figure()
ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG (T)')
ax1.plot(y_exp, '#585858', label='Noisy')
ax1.plot(z_mp_end, '#FF66CC', lw=2, label='MP, estimated variance : ' + str(len(n_mp_end)) + ' atoms.')
ax1.plot(z_mp_stat, '#660066', ls='--', lw=3, label='MP, known variance : ' + str(len(n_mp_stat)) + ' atoms.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])

plt.tight_layout()
#plt.savefig("Prints/9_Endog_Variance_1.pdf", bbox_inches='tight')

fig = plt.figure()
ax2 = fig.add_subplot(212, xlabel='Time (ms)', ylabel='MEG (T)')
ax2.plot(y_exp, '#585858', label='Noisy')
ax2.plot(z_omp_end, '#FF9966', lw=2, label='OMP, estimated variance : ' + str(len(n_omp_end)) + ' iter.')
ax2.plot(z_omp_stat, '#8F2400', ls='--', lw=3,
         label='OMP, known variance : ' + str(len(n_omp_stat)) + ' iter.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])

plt.tight_layout()
#plt.savefig("Prints/9_Endog_Variance_2.pdf", bbox_inches='tight')

# Associated log-decrement results

fig = plt.figure()
ax1 = fig.add_subplot(221, xlabel='Atoms included', ylabel='Log-Error')
ax1.plot(map(math.log, err_mp_end), '#FF66CC', lw=2, label='MP, estimated variance.')
ax1.plot(map(math.log, err_mp_stat), '#660066', ls='--', lw=3, label='MP, known variance.')
plt.legend(loc='upper right', fontsize=11).get_frame().set_alpha(0)

ax2 = fig.add_subplot(222, xlabel='Atoms included', ylabel='Log-Error')
ax2.plot(map(math.log, err_omp_end), '#FF9966', lw=2, label='OMP, estimated variance.')
ax2.plot(map(math.log, err_omp_stat), '#8F2400', ls='--', lw=3, label='OMP, known variance.')
plt.legend(loc='upper right', fontsize=11).get_frame().set_alpha(0)
plt.tight_layout()

#plt.savefig("Prints/10_Endog_Variance_Log_Results.pdf", bbox_inches='tight')

plt.show()