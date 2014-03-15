#coding:latin_1
import math
from Functions import *
from Utility import *
from Classes import *

if __name__ == '__main__':
    import matplotlib.pyplot as plt

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

    z_mp_arb, err_mp_arb, n_mp_arb = mp_arbitrary_criterion(BasisT.T, BasisT, y2, 20)

    #plt.figure()
    #plt.matshow(BasisT.dot(BasisT.T))
    #plt.colorbar()
    #plt.show()

    z_mp_stat, err_mp_stat, n_mp_stat = mp_stat_criterion(BasisT.T, BasisT, y2, math.sqrt(np.var(y1)))

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(y2, '#3D3D29', label='Noisy signal - during experiment')
    ax1.plot(z_mp_stat, 'r', label='Denoised signal : MP, stat criterion')
    plt.xlabel('Time (ms)')
    plt.ylabel('MEG')
    plt.title('Signal denoising - ' + wave_name + ' wavelets, ' + str(wave_level) + '-level decomposition.')
    plt.legend(loc='lower right')
    ax2 = fig.add_subplot(212)
    ax2.plot(z_mp_arb, 'b', label='Denoised signal: MP, no criterion : ' + str(n_mp_arb) + ' iter.')
    ax2.plot(z_mp_stat, 'r', label='Denoised signal : MP, stat criterion : ' + str(n_mp_stat) + ' iter.')
    plt.xlabel('Time (ms)')
    plt.ylabel('MEG')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    
    ######### 4) COMPARING MP & OMP RESULTS #########
    
    z_omp_stat, err_omp_stat, n_omp_stat = omp_stat_criterion(BasisT.T, BasisT, y2, math.sqrt(np.var(y1)))

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(y2, '#3D3D29', label='Noisy signal - during experiment')
    ax1.plot(z_omp_stat, '#009933', label='Denoised signal: OMP, stat criterion')
    plt.xlabel('Time (ms)')
    plt.ylabel('MEG')
    plt.title('Signal denoising - ' + wave_name + ' wavelets, ' + str(wave_level) + '-level decomposition.')
    plt.legend(loc='lower right')
    ax2 = fig.add_subplot(212)
    ax2.plot(z_mp_stat, 'r', label='Denoised signal : MP, stat criterion : ' + str(n_mp_stat) + ' iter.')
    ax2.plot(z_omp_stat, '#009933', label='Denoised signal : OMP, stat criterion : ' + str(n_omp_stat) + ' iter.')
    plt.xlabel('Time (ms)')
    plt.ylabel('MEG')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    
    ######### 4) COMPARING CLASSIC & ORTHONORMALIZED DICTIONARIES #########
    
    GBasisT = gram_schmidt(PhiT.dot(np.identity(y2.size)).T).T
    GBasisT /= np.sqrt(np.sum(GBasisT ** 2, axis=0))
    
    z_omp_gstat, err_omp_gstat, n_omp_gstat = omp_stat_criterion(GBasisT.T, GBasisT, y2, math.sqrt(np.var(y1)))

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(y2, '#3D3D29', label='Noisy signal - during experiment')
    ax1.plot(z_omp_gstat, '#A319A3', label='Denoised signal: OMP stat, Gram-Schmidt.')
    plt.xlabel('Time (ms)')
    plt.ylabel('MEG')
    plt.title('Signal denoising - ' + wave_name + ' wavelets, ' + str(wave_level) + '-level decomposition.')
    plt.legend(loc='lower right')
    ax2 = fig.add_subplot(212)
    ax2.plot(z_omp_gstat, '#A319A3', label='Denoised signal : OMP stat, Gram-Schmidt : ' + str(n_omp_gstat) + ' iter.')
    ax2.plot(z_omp_stat, '#009933', label='Denoised signal : OMP stat, w/o Gram-Sch. : ' + str(n_omp_stat) + ' iter.')
    plt.xlabel('Time (ms)')
    plt.ylabel('MEG')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    ######### 6) LOG-DECREMENT RESULTS #########
    
    #6.1) Insights : Which algorithm, dictionary is the best ?
    #Our criterion here is the log-decrement of the error term.

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(map(math.log, err_mp_stat), 'r', label='MP stat, w/o Gram-Schmidt.')
    ax1.plot(map(math.log, err_omp_stat), '#009933', label='OMP stat, w/o Gram-Schmidt.')
    plt.xlabel('Atoms included')
    plt.ylabel('Log-Error')
    plt.title('Selecting the right algorithm : MP, OMP.')
    plt.legend(loc='lower right')
    ax2 = fig.add_subplot(212)
    ax2.plot(map(math.log, err_omp_gstat), '#A319A3', label='OMP stat, Gram-Schmidt.')
    ax2.plot(map(math.log, err_omp_stat), '#009933', label='OMP stat, w/o Gram-Sch.')
    plt.xlabel('Atoms included')
    plt.ylabel('Log-Error')
    plt.title('Selecting the right dictionary : orthogonalization results.')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    #6.2) Let us compare wavelets with the Daubechies family,
    # using classic and orthonormalized dictionaries.

    fig = plt.figure()

    # - CLASSIC DB DICTIONARY

    ax1 = fig.add_subplot(211)
    for i in xrange(1, 7):
        WaveT = DictT(level=5, name='db'+str(i))
        Wave = Dict(sizes=PhiT.sizes, name=PhiT.name)
        MatT = WaveT.dot(np.identity(y2.size))
        MatT /= np.sqrt(np.sum(MatT ** 2, axis=0))

        Error = omp_stat_criterion(MatT.T, MatT, y2, math.sqrt(np.var(y1)))[1]
        ax1.plot(map(math.log, Error), label='db'+str(i))

    plt.xlabel('Atoms included')
    plt.ylabel('Log-Error')
    plt.title('Log-Error decrement results - w/o Gram-Schmidt.')
    plt.legend(loc='lower right')

    # - ORTHONORMAL DB DICTIONARY

    ax2 = fig.add_subplot(212)
    for i in xrange(1, 7):
        WaveT = DictT(level=5, name='db'+str(i))
        Wave = Dict(sizes=PhiT.sizes, name=PhiT.name)
        MatT = gram_schmidt(WaveT.dot(np.identity(y2.size)).T).T
        MatT /= np.sqrt(np.sum(MatT ** 2, axis=0))

        Error = omp_stat_criterion(MatT.T, MatT, y2, math.sqrt(np.var(y1)))[1]
        ax2.plot(map(math.log, Error), label='db'+str(i))

    plt.xlabel('Atoms included')
    plt.ylabel('Log-Error')
    plt.title('Log-Error decrement results - w/ Gram-Schmidt.')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

    ######### 7) OMP, HARD THRESHOLDING COMPARISON #########

    #Gabriel Peyré's article states that MP /w Orthogonal Basis is equiv. to Hard Thresholding.

    z_thr = hard_thresholding_operator(GBasisT.T, GBasisT, y2, 2e-12)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(y2, '#3D3D29', label='Noisy signal - during experiment')
    ax1.plot(z_thr, 'y', label='Denoised signal: Hard Thresholding. - MAD Threshold.')
    plt.xlabel('Time (ms)')
    plt.ylabel('MEG')
    plt.title('Signal denoising - ' + wave_name + ' wavelets, ' + str(wave_level) + '-level decomposition.')
    plt.legend(loc='lower right')
    ax2 = fig.add_subplot(212)
    ax2.plot(z_omp_gstat, 'r', label='Denoised signal : OMP stat, orthonorm. dict. : ' + str(n_omp_gstat) + ' iter.')
    ax2.plot(z_thr, 'y', label='Denoised signal: Hard Thresholding. - MAD Threshold.')
    plt.xlabel('Time (ms)')
    plt.ylabel('MEG')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
