#coding:latin_1
import numpy as np
import pywt
from math import sqrt, copysign

######### MATCHING PURSUIT, NO CRITERION ######### 

def MP_Arbitrary_Criterion(Phi, PhiT, y, n_iter):
    x = np.zeros(PhiT.dot(y).shape[0])
    err = [np.linalg.norm(y)]

    for k in xrange(n_iter):
        c = PhiT.dot(y - Phi.dot(x))
        abs_c = np.abs(c)
        i_0 = np.argmax(abs_c)
        x[i_0] += c[i_0]

        err.append(np.linalg.norm(y - Phi.dot(x)))

    z = Phi.dot(x)
    return z, err, n_iter

######### MATCHING PURSUIT, STATISTICAL CRITERION ######### 

def MP_Stat_Criterion(Phi, PhiT, y, sigma):
	p=PhiT.dot(y).shape[0]
	Lambda = np.zeros(p)
	err = [np.linalg.norm(y)]
	
	#1st MP STEP
	C = PhiT.dot(y)
	abs_C = np.abs(C)
	i_0 = np.argmax(abs_C)
	
	Dic=[Phi.dot(np.identity(p)[:,i_0])] #Temporary dictionary containing evaluated atoms
	Dic_Arr=np.asarray(Dic).T

	TT, Q=(2,1)
	
	while TT>Q: 
	
		Lambda[i_0] += C[i_0]
		err.append(np.linalg.norm(y - Phi.dot(Lambda)))

		#MP
		C = np.dot(PhiT, y - Phi.dot(Lambda))
		abs_C = np.abs(C)
		i_0 = np.argmax(abs_C)
		
		x=Phi.dot(np.identity(p)[:,i_0])
		H=Dic_Arr.dot(np.linalg.inv((Dic_Arr.T).dot(Dic_Arr))).dot(Dic_Arr.T)
		TT=np.abs(x.T.dot(y))/(sigma*sqrt(np.linalg.norm(x)-(x.T).dot(H).dot(x)))
		
		S=np.random.standard_t(int(y.shape[0]-Dic_Arr.shape[1])-1,50)
		Q=np.percentile(S,95)	
		
		Dic.append(x)
		Dic_Arr=np.asarray(Dic).T
		#print Dic_Arr.shape, i_0, C[i_0], err[-1]

	return Phi.dot(Lambda), err, Dic_Arr.shape[1]

######### ORTHOGONAL MATCHING PURSUIT, STAT CRITERION ######### 

def OMP_Stat_Criterion(Phi, PhiT, y, sigma):	
	p=PhiT.dot(y).shape[0]
	Lambda = np.zeros(p)
	err = [np.linalg.norm(y)]
	
	C = PhiT.dot(y)
	abs_C = np.abs(C)
	i_0 = np.argmax(abs_C)
	
	Dic=[Phi.dot(np.identity(p)[:,i_0])]
	Dic_Arr=np.asarray(Dic).T

	TT, Q=(2,1)
	
	while TT>Q: 
	
		Lambda[i_0] += C[i_0]
		II = np.where(Lambda)[0] 
		Lambda[II] = np.linalg.pinv(Phi.dot(np.identity(p)[:,II])).dot(y) #OMP projection
		err.append(np.linalg.norm(y - Phi.dot(Lambda)))

		#MP
		C = np.dot(PhiT, y - Phi.dot(Lambda))
		abs_C = np.abs(C)
		i_0 = np.argmax(abs_C)
		
		x=Phi.dot(np.identity(p)[:,i_0])
		H=Dic_Arr.dot(np.linalg.inv((Dic_Arr.T).dot(Dic_Arr))).dot(Dic_Arr.T)
		TT=np.abs(x.T.dot(y))/(sigma*sqrt(np.linalg.norm(x)-(x.T).dot(H).dot(x)))
		
		S=np.random.standard_t(int(y.shape[0]-Dic_Arr.shape[1])-1,50)
		Q=np.percentile(S,95)	
		
		Dic.append(x)
		Dic_Arr=np.asarray(Dic).T

	return Phi.dot(Lambda), err, Dic_Arr.shape[1]
	
	
######### MEDIAN ABSOLUTE DEVIATION ######### 

#consistent estimator of the std deviation
#where hat(Sigma) a consistent estimator of the std dev. of the finest level detail coefficients.
    
def MAD(data):

    return np.ma.median(np.abs(data - np.ma.median(data)))



######### HARD THRESHOLDING USING MAD ######### 

#NB : equiv. to OMP when using orthogonal dictionary.
#We use the following threshold :
#Thresh =   hat(Sigma) * sqrt(2 log n),

def Hard_Thresholding_MAD(y,name,lvl):


    c = pywt.wavedec(y, name, level=lvl,mode='per')
    sigma = mad(c[-1])
    thresh=sigma*np.sqrt(2*np.log(len(y)))

    denoised = c[:]
    denoised[1:] = (pywt.thresholding.hard(i, value=thresh) for i in denoised[1:])
    return pywt.waverec(denoised,name,mode='per')


######### COMPARING RESIDUALS ######### 

#Outputs a list containing residuals energy, obtained iteratively by adding significant atoms.

def OEnergy(y2, family, r):
    L=[]
    for i in range(len(family)):
	c = pywt.wavedec(y2, family[i],level=r)
	c=[list(x) for x in c]
	c=sum(c,[])
	c=[x*x for x in c if x<>0] #x=0 happens with null probability
	c.sort(reverse=True)
	c=[np.log(sum(c[n:len(c)])) for n in range(len(c))]
	L+=[c]
    return L
    
    
