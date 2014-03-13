#coding:latin_1
import numpy as np
import pywt
import math
from Functions import MP_Arbitrary_Criterion, MP_Stat_Criterion, OMP_Stat_Criterion
from Utility import Gram_Schmidt
from Classes import DictT, Dict

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ######### CHARGING DATA ######### 

    data = np.load("data_cond1.npy")
    times = np.load("times_cond1.npy")

    times = 1e3 * times
    timesD = times[180:360] # We only denoise between 0 and 300 sec, during the experiment
    timesD1 = times[:180]

    y = data[100, :]
    y1 = y[:180] # Raw signal before the experiment
    y2 = y[180:180+256] # signal during the experiment

    ######### SETTING DICTIONARY PARAMETERS ######### 
    
    wave_name='db6'
    wave_level=5
    PhiT=DictT(level=wave_level, name=wave_name)
    Phi = Dict(sizes= PhiT.sizes, name=PhiT.name)

    BasisT=PhiT.dot(np.identity(y2.size))
    BasisT/= np.sqrt(np.sum(BasisT ** 2, axis=0))
    
    ######### COMPARING MP RESULTS, W/ & W/O STATISTICAL CRITERION ######### 

    z_mp_arb, err_mp_arb, n_mp_arb =MP_Arbitrary_Criterion(BasisT.T,BasisT,y2,20)

    #plt.figure()
    #plt.matshow(BasisT.dot(BasisT.T))
    #plt.colorbar()
    #plt.show()

    z_mp_stat,err_mp_stat, n_mp_stat =MP_Stat_Criterion(BasisT.T, BasisT,y2,math.sqrt(np.var(y1)))
    
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
    
     ######### COMPARING MP & OMP RESULTS ######### 
    
    z_omp_stat,err_omp_stat, n_omp_stat=OMP_Stat_Criterion(BasisT.T, BasisT,y2,math.sqrt(np.var(y1)))
    
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
    
    ######### COMPARING CLASSIC & ORTHONORMALIZED DICTIONARIES ######### 
    
    GBasisT=Gram_Schmidt(PhiT.dot(np.identity(y2.size)).T).T
    GBasisT/= np.sqrt(np.sum(GBasisT ** 2, axis=0))
    
    z_omp_gstat,err_omp_gstat, n_omp_gstat =OMP_Stat_Criterion(GBasisT.T, GBasisT,y2,math.sqrt(np.var(y1)))
    
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
    ax2.plot(z_omp_stat, '#009933', label='Denoised signal : OMP stat, w/o Gram-Schmidt : ' + str(n_omp_stat) + ' iter.')
    plt.xlabel('Time (ms)')
    plt.ylabel('MEG')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    
    ######### LOG-DECREMENT RESULTS ######### 
    
    #Which dictionary is the best ?
    #Our criterion here is the log-decrement of the error term.
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(map(math.log,err_mp_stat), 'r', label='MP stat, w/o Gram-Schmidt.')
    ax1.plot(map(math.log,err_omp_stat), '#009933', label='OMP stat, w/o Gram-Schmidt.')
    plt.xlabel('Atoms included')
    plt.ylabel('Log-Error')
    plt.title('Selecting the right algorithm : MP, OMP.')
    plt.legend(loc='lower right')
    ax2 = fig.add_subplot(212)
    ax2.plot(map(math.log,err_omp_gstat), '#A319A3', label='OMP stat, Gram-Schmidt.')
    ax2.plot(map(math.log,err_omp_stat), '#009933', label='OMP stat, w/o Gram-Schmidt.')
    plt.xlabel('Atoms included')
    plt.ylabel('Log-Error')
    plt.title('Selecting the right dictionary : orthogonalization results.')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
