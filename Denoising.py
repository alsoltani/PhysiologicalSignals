#coding:latin_1
import numpy as np
import matplotlib.pyplot as plt 
import pywt
from Functions import mp, omp, f, ompwavelet, oenergie, ompwavelet2, g

#CHARGING DATA

data = np.load("data_cond1.npy")
data = 1000000000*np.array(data)
times = np.load("times_cond1.npy")

times = 1e3 * times
timesD=times[181:360] #We only denoise between 0 and 300 sec, during the experiment
timesD1=times[1:180]

y = data[100, :]
y1=y[1:180] #Raw signal before the experiment
y2=y[181:360] #signal during the experiment
#~ y3=y[361:540] #signal after the experiment

#ORTHOGONAL MATCHING PURSUIT ORTHOGONAL : DAUBECHIES WAVELETS

z_denoise=ompwavelet2(y1,y2,20,'db1')
z_denoise_2=ompwavelet2(y1,y2,20,'db6')

fig=plt.figure()

ax1 = fig.add_subplot(211)
ax1.plot(y2, 'black', label='Signal during experiment')
ax1.plot(z_denoise, 'r', label='Denoised signal using db1=haar')
plt.xlabel('Time (ms)')
plt.ylabel('MEG')
plt.title('Signal denoising')
plt.legend()

ax2 = fig.add_subplot(212)
ax2.plot(y2, 'black', label='Signal during experiment')
ax2.plot(z_denoise_2, 'b', label='Denoised signal using db6')
plt.xlabel('Time (ms)')
plt.ylabel('MEG')
plt.legend()

plt.show()

#TESTING OUR MODEL : NOISE SIMULATION
#Our idea is the following : we simulate a white noise, add it to the denoised signal
#and run the algorithms to find a new denoised signal. If the two signals match, 
#the algorithm is considered as efficient.

print z_denoise_2.size

