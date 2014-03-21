#coding:latin_1
import math
from Functions import *
from Utility import *
from Classes import *
import matplotlib.pyplot as plt
#from mpltools import style
#style.use('ggplot')

######### 1) CHARGING DATA #########

data = np.load("data_cond1.npy")
times = np.load("times_cond1.npy")

times *= 1e3
timesD = times[180:360]  # We only denoise between 0 and 300 sec, during the experiment
timesD1 = times[:180]

y = data[100, :]
y1 = y[:180]  # Raw signal before the experiment
y2 = y[180:180+256]  # signal during the experiment

######### 2) SETTING DICTIONARY PARAMETERS #########
    
wave_name = 'db6'
wave_level = 5
PhiT = DictT(level=wave_level, name=wave_name)
Phi = Dict(sizes=PhiT.sizes, name=PhiT.name)

BasisT = PhiT.dot(np.identity(y2.size))
BasisT /= np.sqrt(np.sum(BasisT ** 2, axis=0))

######### 3) COMPARING MP RESULTS, W/ & W/O STATISTICAL CRITERION #########

z_mp_arb, err_mp_arb, n_mp_arb = mp_arbitrary_criterion(BasisT.T, BasisT, y2, 15)

#plt.figure()
#plt.matshow(BasisT.dot(BasisT.T))
#plt.colorbar()
#plt.show()

z_mp_stat, err_mp_stat, n_mp_stat = mp_stat_criterion(BasisT.T, BasisT, y2, math.sqrt(np.var(y1)))

fig = plt.figure(1)
ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG')
ax1.plot(y2, '#585858', label='Noisy signal - during experiment')
ax1.plot(z_mp_stat, '#FE2E2E', lw=2, label='Denoised: MP, statistical criterion')
plt.title('Signal denoising - ' + wave_name + ' wavelets, ' + str(wave_level) + '-level decomposition.')
plt.legend(loc='lower right', fontsize=11)
plt.axis([0, 256, -5e-12, 5e-12])

ax2 = fig.add_subplot(212, xlabel='Time (ms)', ylabel='MEG')
ax2.plot(z_mp_stat, '#FE2E2E', lw=2, label='Denoised: MP, statistical criterion : ' + str(n_mp_stat) + ' iter.')
ax2.plot(z_mp_arb, '#1C1C1C', ls='--', lw=3, label='Denoised: MP, no criterion : ' + str(n_mp_arb) + ' iter.')
plt.legend(loc='lower right', fontsize=11)
plt.axis([0, 256, -5e-12, 5e-12])
plt.tight_layout()

######### 4) COMPARING MP & OMP RESULTS #########

z_omp_stat, err_omp_stat, n_omp_stat = omp_stat_criterion(BasisT.T, BasisT, y2, math.sqrt(np.var(y1)))

fig = plt.figure(2)
ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG')
ax1.plot(y2, '#585858', label='Noisy - during experiment')
ax1.plot(z_omp_stat, '#6699FF', lw=2, label='Denoised: OMP : ' + str(n_omp_stat) + ' iter.')
ax1.plot(z_mp_stat, '#B20000', ls='--', lw=3, label='Denoised: MP : ' + str(n_mp_stat) + ' iter.')
plt.title('Signal denoising - ' + wave_name + ' wavelets, ' + str(wave_level) + '-level decomposition.')
plt.legend(loc='lower right', fontsize=11)
plt.axis([0, 256, -5e-12, 5e-12])

ax2 = fig.add_subplot(212, xlabel='Atoms included', ylabel='Log-error')
ax2.plot(map(math.log, err_omp_stat), '#6699FF', lw=2, label='OMP')
ax2.plot(map(math.log, err_mp_stat), '#B20000', ls='--', lw=3, label='MP')
plt.title('Log-decrement results')
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()

######### 5) COMPARING CLASSIC & ORTHONORMALIZED DICTIONARIES #########
    
GBasisT = gram_schmidt(PhiT.dot(np.identity(y2.size)).T).T
GBasisT /= np.sqrt(np.sum(GBasisT ** 2, axis=0))
#GBasisT = np.linalg.qr(PhiT.dot(np.identity(y2.size)))[0]
#print GBasisT.dot(GBasisT.T)

z_omp_gstat, err_omp_gstat, n_omp_gstat = omp_stat_criterion(GBasisT.T, GBasisT, y2, math.sqrt(np.var(y1)))

fig = plt.figure()
ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG')
ax1.plot(y2, '#585858', label='Noisy signal - during experiment')
ax1.plot(z_omp_gstat, '#FF6699', linewidth=2, label='Denoised: OMP, orthonorm. dic :' + str(n_omp_gstat) + ' iter.')
ax1.plot(z_omp_stat, '#002E8A', ls='--', lw=3, label='Denoised: OMP, classic dic : ' + str(n_omp_stat) + ' iter.')
plt.title('Signal denoising - ' + wave_name + ' wavelets, ' + str(wave_level) + '-level decomposition.')
plt.legend(loc='lower right', fontsize=11)
plt.axis([0, 256, -5e-12, 5e-12])

ax2 = fig.add_subplot(212, xlabel='Atoms included', ylabel='Log-Error')
ax2.plot(map(math.log, err_omp_gstat), '#FF6699', linewidth=2, label='OMP, orthonorm. dic')
ax2.plot(map(math.log, err_omp_stat), '#002E8A', ls='--', lw=3, label='OMP, classic dic')
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()

######### 6) LOG-DECREMENT RESULTS #########
    
#Let us compare wavelet dictionaries within the Daubechies family,
#using classic and orthonormalized dictionaries.

fig = plt.figure()

# - CLASSIC DB DICTIONARY

ax1 = fig.add_subplot(211, xlabel='Atoms included', ylabel='Log-Error')
for i in xrange(1, 7):
    WaveT = DictT(level=5, name='db'+str(i))
    MatT = WaveT.dot(np.identity(y2.size))
    MatT /= np.sqrt(np.sum(MatT ** 2, axis=0))

    Error = omp_stat_criterion(MatT.T, MatT, y2, math.sqrt(np.var(y1)))[1]
    ax1.plot(map(math.log, Error), lw=2, label='db'+str(i))

plt.title('Log-Error decrement results - w/o Gram-Schmidt.')
plt.legend(loc='lower right', fontsize=11)

# - ORTHONORMAL DB DICTIONARY

ax2 = fig.add_subplot(212, xlabel='Atoms included', ylabel='Log-Error')
for i in xrange(1, 7):
    WaveT = DictT(level=5, name='db'+str(i))
    MatT = gram_schmidt(WaveT.dot(np.identity(y2.size)).T).T
    MatT /= np.sqrt(np.sum(MatT ** 2, axis=0))

    Error = omp_stat_criterion(MatT.T, MatT, y2, math.sqrt(np.var(y1)))[1]
    ax2.plot(map(math.log, Error), lw=2, label='db'+str(i))

plt.title('Log-Error decrement results - w/ Gram-Schmidt.')
plt.legend(loc='lower right', fontsize=11)

plt.tight_layout()

######### 7) OMP, HARD THRESHOLDING COMPARISON #########

#Gabriel Peyre's article states that MP /w can be equiv. to Hard Thresholding,
#when the decomposition basis is a square orthonormal matrix.
#Hard Thresholding has to verify same sparsity constraint as the Matching Pursuit solution.

z_thr = hard_thresholding(GBasisT.T, GBasisT, y2, n_omp_gstat)

fig = plt.figure()
ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG')
ax1.plot(y2, '#585858', label='Noisy signal - during experiment')
ax1.plot(z_thr, '#FF9933', lw=2, label='Denoised: Hard Thresholding.')
plt.title('Signal denoising - ' + wave_name + ' wavelets, ' + str(wave_level) + '-level decomposition.')
plt.legend(loc='lower right', fontsize=11)
plt.axis([0, 256, -5e-12, 5e-12])

ax2 = fig.add_subplot(212, xlabel='Time (ms)', ylabel='MEG')
ax2.plot(z_thr, '#FF9933', lw=2, label='Denoised: Hard Thresholding.')
ax2.plot(z_omp_gstat, '#A30052', ls='--', lw=3, label='Denoised: OMP, orthonorm. dict: ' + str(n_omp_gstat) + ' iter.')
plt.legend(loc='lower right', fontsize=11)
plt.axis([0, 256, -5e-12, 5e-12])
plt.tight_layout()

######### 8) TESTING OUR MODEL #########

#TESTING OUR MODEL : NOISE SIMULATION
#Our idea is the following : we simulate a white noise, add it to the denoised signal
#and run the algorithms to find a new denoised signal. If the two signals match,
#the algorithm is considered as efficient.

noise = np.asarray(np.random.normal(0, np.var(y1), y2.shape[0])).T

z_new = z_mp_stat+noise
z_new_stat, err_new_stat, n_new_stat = mp_stat_criterion(BasisT.T, BasisT, z_new, math.sqrt(np.var(y1)))

fig = plt.figure()
ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG')
ax1.plot(y2, '#585858', label='Noisy signal - during experiment')
ax1.plot(z_new_stat, '#33FF33', lw=2, label='Re-denoised: same charact. : ' + str(n_new_stat) + ' iter.')
ax1.plot(z_mp_stat, '#660000', ls='--', lw=3, label='Denoised: MP, classic dict. : ' + str(n_mp_stat) + ' iter.')
plt.title('Signal denoising - ' + wave_name + ' wavelets, ' + str(wave_level) + '-level decomposition.')
plt.legend(loc='lower right', fontsize=11)
plt.axis([0, 256, -5e-12, 5e-12])

ax2 = fig.add_subplot(212, xlabel='Time (ms)', ylabel='MEG')
ax2.plot(z_new_stat-z_mp_stat, '#585858', lw=2, label='Difference between the 2 denoised signals.')
plt.xlabel('Time (ms)')
plt.ylabel('MEG')
plt.legend(loc='lower right', fontsize=11)
plt.axis([0, 256, -5e-12, 5e-12])
plt.tight_layout()

######### 9) ENDOGENOUS VARIANCE #########

#We refine our model by considering variance as an endogenous variable,
#using OLS estimator s².

z_mp_end, err_mp_end, n_mp_end = mp_stat_endogen_var(BasisT.T, BasisT, y2)
z_omp_end, err_omp_end, n_omp_end = omp_stat_endogen_var(BasisT.T, BasisT, y2)

fig = plt.figure()
ax1 = fig.add_subplot(211, xlabel='Time (ms)', ylabel='MEG')
ax1.plot(y2, '#585858', label='Noisy - during experiment')
ax1.plot(z_mp_end, '#FF66CC', lw=2, label='Denoised: MP, endo. variance : ' + str(n_mp_end) + ' iter.')
ax1.plot(z_mp_stat, '#660066', ls='--', lw=3, label='Denoised: MP, exog. variance : ' + str(n_mp_stat) + ' iter.')
plt.title('Signal denoising - ' + wave_name + ' wavelets, ' + str(wave_level) + '-level decomposition.')
plt.legend(loc='lower right', fontsize=11)
plt.axis([0, 256, -5e-12, 5e-12])

ax2 = fig.add_subplot(212, xlabel='Time (ms)', ylabel='MEG')
ax2.plot(y2, '#585858', label='Noisy - during experiment')
ax2.plot(z_omp_end, '#FF9966', lw=2, label='Denoised: OMP, endo. variance : ' + str(n_omp_end) + ' iter.')
ax2.plot(z_omp_stat, '#8F2400', ls='--', lw=3, label='Denoised: OMP, exog. variance : ' + str(n_omp_stat) + ' iter.')
plt.legend(loc='lower right', fontsize=11)
plt.axis([0, 256, -5e-12, 5e-12])

plt.tight_layout()

#Associated log-decrement results

fig = plt.figure()
ax1 = fig.add_subplot(211, xlabel='Atoms included', ylabel='Log-Error')
ax1.plot(map(math.log, err_mp_end), '#FF66CC', lw=2, label='MP, endo. var.')
ax1.plot(map(math.log, err_mp_stat), '#660066', ls='--', lw=3, label='MP, exog. var.')
plt.title('Log-decrement results')
plt.legend(loc='lower right', fontsize=11)

ax2 = fig.add_subplot(212, xlabel='Atoms included', ylabel='Log-Error')
ax2.plot(map(math.log, err_omp_end), '#FF9966', lw=2, label='OMP stat, endo. var.')
ax2.plot(map(math.log, err_omp_stat), '#8F2400', ls='--', lw=3, label='OMP stat, exog. var.')
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()

plt.show()