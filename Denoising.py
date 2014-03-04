#coding:latin_1
import numpy as np
import matplotlib.pyplot as plt 
import pywt
from mp_2 import mp, omp, f, ompwavelet, oenergie, ompwavelet2, g

#CHARGEMENT DES DONNEES

data = np.load("data_cond1.npy")
data = 1000000000*np.array(data)
times = np.load("times_cond1.npy")

times = 1e3 * times
timesD=times[181:360] #On ne débruite qu'entre 0 et 300 sec, pendant l'expérience
timesD1=times[1:180]

y = data[100, :]
y1=y[1:180] #signal avant l'expérience
y2=y[181:360] #signal pendant l'expérience
#~ y3=y[361:540] #signal après l'expérience

#MATCHING PURSUIT ORTHOGONAL : ONDELETTES

z_denoise=ompwavelet2(y1,y2,20,'db1')
z_denoise_2=ompwavelet2(y1,y2,20,'db6')

fig=plt.figure()

ax1 = fig.add_subplot(211)
ax1.plot(y2, 'black', label='Signal pendant experience')
ax1.plot(z_denoise, 'r', label='Signal debruite avec db1=haar')
plt.xlabel('Time (ms)')
plt.ylabel('MEG')
plt.title('Debruitage de signaux')
plt.legend()

ax2 = fig.add_subplot(212)
ax2.plot(y2, 'black', label='Signal pendant experience')
ax2.plot(z_denoise_2, 'b', label='Signal debruite avec db6')
plt.xlabel('Time (ms)')
plt.ylabel('MEG')
plt.legend()

plt.show()

