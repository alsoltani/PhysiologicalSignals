#coding:latin_1
import math
from Functions import *
from Utility import *
from Classes import *

import matplotlib.pyplot as plt
from mpltools import style
style.use('ggplot')

#----------------------------------------   1. CHARGING DATA   ----------------------------------------#

data = np.load("data_cond1.npy")
times = np.load("times_cond1.npy")

times *= 1e3
timesD = times[180:360]
timesD1 = times[:180]


#----------------------------- A. Let us use 2 different acquisitions first. --------------------------#

first = 90
last = 100
y = data[first:last, :].T

# - Signal before the experiment.
y_early = data[first:last, :180].T

# - Signal during the experiment.
y_exp = data[first:last, 180:180+256].T


# - Pre-experiment variance.
std = np.zeros(y_early.shape[1])
for i in xrange(y_early.shape[1]):
    std[i] = math.sqrt(np.var(y_early[:, i]))

#------------------------------------   2. DICTIONARY PARAMETERS   ------------------------------------#

wave_name = 'db6'
wave_level = None
PhiT = DictT(level=wave_level, name=wave_name)
Phi = Dict(sizes=PhiT.sizes, name=PhiT.name)

BasisT = PhiT.dot(np.identity(y_exp.shape[0]))
BasisT /= np.sqrt(np.sum(BasisT ** 2, axis=0))

#---------------------------------------   3. MULTI-SENSOR MP   ---------------------------------------#

z, err, n = multi_channel_mp(BasisT.T, BasisT, y_exp, std)

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


plt.savefig("Multichannel_MP_2.pdf", bbox_inches='tight')

plt.show()