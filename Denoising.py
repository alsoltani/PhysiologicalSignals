#coding:latin_1
import math
from Functions import *
from Utility import *
import timeit
from Classes import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
from mpltools import style
style.use('ggplot')

#----------------------------------------   1. CHARGING DATA   --------------------------------------#

data = np.load("data_cond1.npy")
times = np.load("times_cond1.npy")

times *= 1e3
timesD = times[180:360]  # We only denoise between 0 and 300 sec, during the experiment
timesD1 = times[:180]

y = data[100, :]
y1 = y[:180]  # Raw signal before the experiment
y2 = y[180:180+256]  # signal during the experiment

#-------------------------------------   2. DICTIONARY PARAMETERS   ---------------------------------#
    
wave_name = 'db6'
wave_level = None
PhiT = DictT(level=wave_level, name=wave_name)
Phi = Dict(sizes=PhiT.sizes, name=PhiT.name)

BasisT = PhiT.dot(np.identity(y2.size))
BasisT /= np.sqrt(np.sum(BasisT ** 2, axis=0))

#--------------------   3. COMPARING MP RESULTS, W/ & W/O STATISTICAL CRITERION   -------------------#

z_mp_arb, err_mp_arb, n_mp_arb = mp_arbitrary_criterion(BasisT.T, BasisT, y2, 15)

z_mp_stat, err_mp_stat, n_mp_stat = mp_stat_criterion(BasisT.T, BasisT, y2, math.sqrt(np.var(y1)))

fig = plt.figure(1)
ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG')
ax1.plot(y2, '#585858', label='Noisy signal - during experiment')
ax1.plot(z_mp_stat, '#FE2E2E', lw=2, label='Denoised: MP, statistical criterion')
#plt.title('Signal denoising - ' + wave_name + ' wavelets, full-level decomposition.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])

ax2 = fig.add_subplot(212, xlabel='Time (ms)', ylabel='MEG')
ax2.plot(z_mp_stat, '#FE2E2E', lw=2, label='Denoised: MP, statistical criterion : ' + str(len(n_mp_stat)) + ' iter.')
ax2.plot(z_mp_arb, '#1C1C1C', ls='--', lw=3, label='Denoised: MP, no criterion : ' + str(len(n_mp_arb)) + ' iter.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])
plt.tight_layout()

#plt.savefig("Image_1.pdf", bbox_inches='tight')

#----------------------------------   4. COMPARING MP & OMP RESULTS   -------------------------------#

z_omp_stat, err_omp_stat, n_omp_stat = omp_stat_criterion(BasisT.T, BasisT, y2, math.sqrt(np.var(y1)))

fig = plt.figure(2)
ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG')
ax1.plot(y2, '#585858', label='Noisy - during experiment')
ax1.plot(z_omp_stat, '#6699FF', lw=2, label='Denoised: OMP : ' + str(len(n_omp_stat)) + ' iter.')
ax1.plot(z_mp_stat, '#B20000', ls='--', lw=3, label='Denoised: MP : ' + str(len(n_mp_stat)) + ' iter.')
plt.title('Signal denoising - ' + wave_name + ' wavelets, full-level decomposition.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])

ax2 = fig.add_subplot(212, xlabel='Atoms included', ylabel='Log-error')
ax2.plot(map(math.log, err_omp_stat), '#6699FF', lw=2, label='OMP')
ax2.plot(map(math.log, err_mp_stat), '#B20000', ls='--', lw=3, label='MP')
plt.title('Log-decrement results')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.tight_layout()


#------------------------   5. COMPARING CLASSIC & ORTHONORMALIZED DICTIONARIES  --------------------#

start = timeit.default_timer()

GBasisT = gram_schmidt(PhiT.dot(np.identity(y2.size)).T).T
GBasisT /= np.sqrt(np.sum(GBasisT ** 2, axis=0))

stop = timeit.default_timer()
print "Orthonormalization time : " + str(stop - start)


z_omp_gstat, err_omp_gstat, n_omp_gstat = omp_stat_criterion(GBasisT.T, GBasisT, y2, math.sqrt(np.var(y1)))

fig = plt.figure()
ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG')
ax1.plot(y2, '#585858', label='Noisy signal - during experiment')
ax1.plot(z_omp_gstat, '#FF6699', linewidth=2,
         label='Denoised: OMP, orthonorm. dic :' + str(len(n_omp_gstat)) + ' iter.')
ax1.plot(z_omp_stat, '#002E8A', ls='--', lw=3, label='Denoised: OMP, classic dic : ' + str(len(n_omp_stat)) + ' iter.')
plt.title('Signal denoising - ' + wave_name + ' wavelets, full-level decomposition.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])

ax2 = fig.add_subplot(212, xlabel='Atoms included', ylabel='Log-Error')
ax2.plot(map(math.log, err_omp_gstat), '#FF6699', linewidth=2, label='OMP, orthonorm. dic')
ax2.plot(map(math.log, err_omp_stat), '#002E8A', ls='--', lw=3, label='OMP, classic dic')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.tight_layout()


#-----------------------------------   6. LOG-DECREMENTS RESULTS  -----------------------------------#

# Let us compare wavelet dictionaries within the Daubechies family,
# using classic and orthonormalized dictionaries.


#--------------- A. CLASSIC DB DICTIONARY ---------------#

ax1 = fig.add_subplot(211, xlabel='Atoms included', ylabel='Log-Error')
for i in xrange(1, 7):
    WaveT = DictT(level=None, name='db'+str(i))
    MatT = WaveT.dot(np.identity(y2.size))
    MatT /= np.sqrt(np.sum(MatT ** 2, axis=0))

    Z, Error, N = omp_stat_criterion(MatT.T, MatT, y2, math.sqrt(np.var(y1)))

    ax1.plot(map(math.log, Error), lw=2, label='db'+str(i))

plt.title('Log-Error decrement results - w/o Gram-Schmidt.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)

#--------------- B. ORTHONORMAL DB DICTIONARY ---------------#

ax2 = fig.add_subplot(212, xlabel='Atoms included', ylabel='Log-Error')
for i in xrange(1, 7):
    WaveT = DictT(level=None, name='db'+str(i))
    MatT = gram_schmidt(WaveT.dot(np.identity(y2.size)).T).T
    MatT /= np.sqrt(np.sum(MatT ** 2, axis=0))

    Z, Error, N = omp_stat_criterion(MatT.T, MatT, y2, math.sqrt(np.var(y1)))

    ax2.plot(map(math.log, Error), lw=2, label='db'+str(i))

plt.title('Log-Error decrement results - w/ Gram-Schmidt.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.tight_layout()

#------------------------------   7. OMP, HARD THRESHOLDING COMPARISON  -----------------------------#

# Gabriel Peyre's article states that MP /w can be equiv. to Hard Thresholding,
# when the decomposition basis is a square orthonormal matrix.
# Hard Thresholding has to verify same sparsity constraint as the Matching Pursuit solution.

z_thr = hard_thresholding(GBasisT.T, GBasisT, y2, len(n_omp_gstat))

fig = plt.figure()
ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG')
ax1.plot(y2, '#585858', label='Noisy signal - during experiment')
ax1.plot(z_thr, '#FF9933', lw=2, label='Denoised: Hard Thresholding.')
plt.title('Signal denoising - ' + wave_name + ' wavelets, full-level decomposition.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])

ax2 = fig.add_subplot(212, xlabel='Time (ms)', ylabel='MEG')
ax2.plot(z_thr, '#FF9933', lw=2, label='Denoised: Hard Thresholding.')
ax2.plot(z_omp_gstat, '#A30052', ls='--', lw=3,
         label='Denoised: OMP, orthonorm. dict: ' + str(len(n_omp_gstat)) + ' iter.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])
plt.tight_layout()


#--------------------------------------   8. TESTING OUR MODEL  -------------------------------------#

# -------------------------------------   A.  NOISE SIMULATION   ------------------------------------#

# Our idea is the following : we simulate a white noise, add it to the denoised signal
# and run the algorithms to find a new denoised signal. If the two signals match,
# the algorithm is considered as efficient.

noise = np.asarray(np.random.normal(0, np.var(y1), y2.shape[0])).T

z_new = z_mp_stat+noise
z_new_stat, err_new_stat, n_new_stat = mp_stat_criterion(BasisT.T, BasisT, z_new, math.sqrt(np.var(y1)))

fig = plt.figure()
ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG')
ax1.plot(y2, '#585858', label='Noisy signal - during experiment')
ax1.plot(z_new_stat, '#33FF33', lw=2, label='Re-denoised: same charact. : ' + str(len(n_new_stat)) + ' iter.')
ax1.plot(z_mp_stat, '#660000', ls='--', lw=3, label='Denoised: MP, classic dict. : ' + str(len(n_mp_stat)) + ' iter.')
plt.title('Signal denoising - ' + wave_name + ' wavelets, full-level decomposition.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])

ax2 = fig.add_subplot(212, xlabel='Time (ms)', ylabel='MEG')
ax2.plot(z_new_stat-z_mp_stat, '#585858', lw=2, label='Difference between the 2 denoised signals.')
plt.xlabel('Time (ms)')
plt.ylabel('MEG')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])
plt.tight_layout()


# -------------------------------------   B.  RESIDUAL TESTING   ------------------------------------#
# In this section, we test the assumption under which our residuals are gaussian.
# We implement different tests to compare results.

# -----------------------------------  B.1. Quantile-Quantile Plot  ---------------------------------#


# MP Outputs

fig = plt.figure()

for i in xrange(1, 5):
    ax = fig.add_subplot(220+i)
    WaveT = DictT(level=None, name='db'+str(2*i-1))
    MatT = WaveT.dot(np.identity(y2.size))
    MatT /= np.sqrt(np.sum(MatT ** 2, axis=0))
    Z, Error, N = omp_stat_criterion(MatT.T, MatT, y2, math.sqrt(np.var(y1)))

    sm.qqplot(y2-Z, line='s', ax=ax)

    txt = ax.text(-2.8, ax.get_ylim()[1] * 0.9, 'Daubechies '+str(2*i-1),
                  fontsize=12, color='#B40431', verticalalignment='top')
    txt.set_bbox(dict(facecolor='#F5A9A9', alpha=1))

plt.tight_layout()
#plt.savefig("QQPlot.pdf", bbox_inches='tight')

# Pre-experiment acquisitions

fig = plt.figure()
sm.qqplot(y1, line='s')

txt = plt.text(-2.8, 0, 'Pre-experiment signal',
              fontsize=12, color='#B40431', verticalalignment='top')
txt.set_bbox(dict(facecolor='#F5A9A9', alpha=1))

plt.savefig("QQPlot-y1.pdf", bbox_inches='tight')


# ----------------------------------  B.2. Kolmogorov-Smirnoff Test  --------------------------------#

for i in xrange(1, 7):
    WaveT = DictT(level=5, name='db'+str(i))
    MatT = WaveT.dot(np.identity(y2.size))
    MatT /= np.sqrt(np.sum(MatT ** 2, axis=0))

    Z, Error, N = omp_stat_criterion(MatT.T, MatT, y2, math.sqrt(np.var(y1)))
    print sm.stats.diagnostic.kstest_normal(y2-Z)

print "\n"
print sm.stats.diagnostic.kstest_normal(y1)


#-------------------------------------   9. ENDOGENOUS VARIANCE  ------------------------------------#

# We refine our model by considering variance as an endogenous variable,
# using OLS estimator s².

z_mp_end, err_mp_end, n_mp_end = mp_stat_endogen_var(BasisT.T, BasisT, y2)
z_omp_end, err_omp_end, n_omp_end = omp_stat_endogen_var(BasisT.T, BasisT, y2)

fig = plt.figure()
ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG')
ax1.plot(y2, '#585858', label='Noisy')
ax1.plot(z_mp_end, '#FF66CC', lw=2, label='MP, endo. variance : ' + str(len(n_mp_end)) + ' iter.')
ax1.plot(z_mp_stat, '#660066', ls='--', lw=3, label='MP, exog. variance : ' + str(len(n_mp_stat)) + ' iter.')
#plt.title('Signal denoising - ' + wave_name + ' wavelets, ' + str(wave_level) + '-level decomposition.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])

ax2 = fig.add_subplot(212, xlabel='Time (ms)', ylabel='MEG')
ax2.plot(y2, '#585858', label='Noisy')
ax2.plot(z_omp_end, '#FF9966', lw=2, label='OMP, endo. variance : ' + str(len(n_omp_end)) + ' iter.')
ax2.plot(z_omp_stat, '#8F2400', ls='--', lw=3,
         label='OMP, exog. variance : ' + str(len(n_omp_stat)) + ' iter.')
plt.legend(loc='lower right', fontsize=11).get_frame().set_alpha(0)
plt.axis([0, 256, -5e-12, 5e-12])

plt.tight_layout()

plt.savefig("Endog_Variance.pdf", bbox_inches='tight')

# Associated log-decrement results

fig = plt.figure()
ax1 = fig.add_subplot(211, xlabel='Atoms included', ylabel='Log-Error')
ax1.plot(map(math.log, err_mp_end), '#FF66CC', lw=2, label='MP, endo. var.')
ax1.plot(map(math.log, err_mp_stat), '#660066', ls='--', lw=3, label='MP, exog. var.')
#plt.title('Log-decrement results')
plt.legend(loc='upper right', fontsize=11).get_frame().set_alpha(0)

ax2 = fig.add_subplot(212, xlabel='Atoms included', ylabel='Log-Error')
ax2.plot(map(math.log, err_omp_end), '#FF9966', lw=2, label='OMP, endo. var.')
ax2.plot(map(math.log, err_omp_stat), '#8F2400', ls='--', lw=3, label='OMP, exog. var.')
plt.legend(loc='upper right', fontsize=11).get_frame().set_alpha(0)
plt.tight_layout()

plt.savefig("Endog_Variance_Log_Results.pdf", bbox_inches='tight')

#------------------------------------   10. DICTIONARY SELECTION  -----------------------------------#

#------------------------------------   A. GOODNESS OF FIT : R2  ------------------------------------#
# We compute classic and ajusted R2 obtained via OMP algorithm, for an entire wavelet
# family. Results are presented in the following histogram.


fig = plt.figure()

r2_seq = []
r2_seq_bis = []
wvlist = pywt.wavelist('db')

ax1 = fig.add_subplot(211, ylabel='R2')
ax2 = fig.add_subplot(212, xlabel='Daubechies wavelets', ylabel='Ajusted R2')

for a in wvlist:
    r2_seq.append(classic_r2(level=None, name_1=a, algorithm=omp_stat_criterion,
                  y=y2, sigma=math.sqrt(np.var(y1)), orth='no')[0])
    r2_seq_bis.append(ajusted_r2(level=None, name_1=a, algorithm=omp_stat_criterion,
                      y=y2, sigma=math.sqrt(np.var(y1)), orth='no')[0])

r2_dict = dict(zip(wvlist, r2_seq))
r2_dict_bis = dict(zip(wvlist, r2_seq_bis))

ax1.bar(range(len(r2_dict)), r2_dict.values())
ax1.set_xticks(range(len(r2_dict)))
L = []
for a in r2_dict.keys():
    L.append(a[2:])
ax1.set_xticklabels(L)
x1, x2, y1, y2 = ax1.axis()
ax1.axis((x1, x2, 0.84, 0.96))

ax2.bar(range(len(r2_dict_bis)), r2_dict_bis.values(), color="#FA5858")
ax2.set_xticks(range(len(r2_dict)))

L = []
for a in r2_dict_bis.keys():
    L.append(a[2:])
ax2.set_xticklabels(L)
x1, x2, y1, y2 = ax2.axis()
ax2.axis((x1, x2, 0.84, 0.96))

plt.tight_layout()

plt.show()


#---------------------------------   B. MODEL COMPARISON : F-TEST  ----------------------------------#

print "\n-----------------TESTING MODELS------------------------\n"


name_1 = 'db'+str(6)
name_2 = 'db'+str(14)
print "Testing " + name_1 + " and " + name_2 + "..."
dictionary_testing(level=5, name_1=name_1, name_2=name_2, algorithm=omp_stat_criterion,
                   y=y2, sigma=math.sqrt(np.var(y1)), orth="no")

print "\n"


plt.show()
