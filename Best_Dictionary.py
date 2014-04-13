#coding:latin_1
import math
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpltools import style
import timeit
from Functions import *
from Classes import *

style.use('ggplot')

#--------------------------------------------------------------------------------------------------------------#
#---------------------------------------------   1. CHARGING DATA   -------------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

data = np.load("data_cond1.npy")
times = np.load("times_cond1.npy")

times *= 1e3

# We only denoise between 0 and 300 sec, during the experiment.
timesD = times[180:360]
timesD1 = times[:180]

y = data[100, :]

# Signal before the experiment.
y1 = y[:180]

# Signal during the experiment.
y2 = y[180:180+256]

#--------------------------------------------------------------------------------------------------------------#
#------------------------------------------   2. DICTIONARY PARAMETERS   --------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

wave_name = 'db6'
wave_level = None
PhiT = DictT(level=wave_level, name=wave_name)
Phi = Dict(sizes=PhiT.sizes, name=PhiT.name)

BasisT = PhiT.dot(np.identity(y2.size))
BasisT /= np.sqrt(np.sum(BasisT ** 2, axis=0))


#--------------------------------------------------------------------------------------------------------------#
#------------------------------------------   SELECTING BEST DICTIONARY   -------------------------------------#
#--------------------------------------------------------------------------------------------------------------#

# Number of samples.
N_iter = 50

# Length of wavelist.
N_list = 20

Wavelist = np.arange(1, N_list+1, 1)
Entropy = np.zeros((N_list, N_iter))

for k in xrange(N_iter):
    print "-------ITERATION " + str(k+1) + " -------"

    for i in Wavelist:
        #print "Stage " + str(i) + "..."
        WaveT = DictT(level=None, name='db'+str(i))
        MatT = WaveT.dot(np.identity(y2.size))
        MatT /= np.sqrt(np.sum(MatT ** 2, axis=0))

        x = omp_stat_criterion(MatT.T, MatT, y2, math.sqrt(np.var(y1)))[0]
        x_unit = np.abs(x) / np.linalg.norm(x, 1)

        Entropy[i-1, k] = shannon(x_unit)

fig = plt.figure()
Mean_Entropy = np.mean(Entropy, axis=1)
List_Entropy = list(np.array(Mean_Entropy).reshape(-1,))

plt.plot(Wavelist, List_Entropy, '-', c='#8A0829', lw=2)
plt.ylabel("Cost Function : Shannon Entropy")
plt.xlabel("Daubechies Wavelets")
plt.xticks(np.arange(min(Wavelist), max(Wavelist)+1, 1))
x1, x2, y1, y2 = plt.axis()
plt.axis([x1, 21, y1, y2])
plt.fill_between(Wavelist, List_Entropy, where=List_Entropy >= min(List_Entropy), interpolate=True, color="#FA5858")
plt.show()
