#coding:latin_1
import numpy as np
import pywt

#MATCHING PURSUIT

def mp(Phi, y, n_iter=20):
    N = Phi.shape[1]
    x = np.zeros(N)

    err = [np.linalg.norm(y)]

    for k in xrange(n_iter):
        c = np.dot(Phi.T, y - np.dot(Phi, x))
        abs_c = np.abs(c)
        i_0 = np.argmax(abs_c)
        x[i_0] += c[i_0]

        err.append(np.linalg.norm(y - np.dot(Phi, x)))

    return x, err

#MATCHING PURSUIT ORTHOGONAL

def omp(Phi, y, seuil,nmax):	
    N = Phi.shape[1]
    x = np.zeros(N)
    err = [np.linalg.norm(y)]
    c = np.dot(Phi.T, y)
    i_0 = np.argmax(np.abs(c))
    n=0

    while np.abs(c)[i_0]>seuil and n<nmax :
        x[i_0] +=c[i_0] 
        II = np.where(x)[0]
        x[II] = np.dot(np.linalg.pinv(Phi[:, II]), y)
        err.append(np.linalg.norm(y - np.dot(Phi, x)))
	c = np.dot(Phi.T, y - np.dot(Phi,x))
	i_0 = np.argmax(np.abs(c))
	n+=1
    return x, err
    
#SEUILLAGE DUR - PREMIER CRITERE (INVALIDE).
#Ici on réalise un seuillage dur sur le signal pendant l'expérience, avec pour critère : seuls les coefficients > seuil sont conservés.
#(Nous avons vu que le seuil choisi (max) n'était pas adapté.)
	
#Indicatrice{|x|>seuil}
def f(x,seuil):
	y=0
	if abs(x)>seuil:
		y=x
	return y

def ompwavelet(y2, seuil, r):	
    c = pywt.wavedec(y2, 'coif2',level=r)
    c=[[f(x,seuil) for x in z] for z in c] #dans le cas de l'omp, le critere d'arret concerne uniquement le produit scalaire entre le signal et l'atome
    y=pywt.waverec(c, 'coif2')
    return y

#SEUILLAGE DUR - SECOND CRITERE
#Ici on réalise un seuillage dur sur le signal pendant l'expérience, avec un nouveau seuil :
#seuls les carrés des coefficients > variance avant l'expérience sont conservés.

#Indicatrice{x²>seuil}
def g(x,seuil):
	y=0
	if x*x>seuil:
		y=x
	return y   

def ompwavelet2(y1,y2,r,famille):
    v=np.var(y1)
    c = pywt.wavedec(y2, famille,level=r)
    c=[[g(x,v) for x in z] for z in c]
    y=pywt.waverec(c, famille)
    return y

#COMPARAISON DES VARIANCES
#Comparer la variance du residu courant,avec la variance pré-expérience, c'est comparer :
#- la somme des carrés des coordonnées danss la base d'ondelettes (pré-expérience) 
#- la somme des carrés des coordonnées "restantes"
       
#Renvoie une liste contenant l'énergie des résidus, en ajoutant à chaque étape des atomes significatifs.
def oenergie(y2, famille, r):
    L=[]
    for i in range(len(famille)):
	c = pywt.wavedec(y2, famille[i],level=r)
	c=[list(x) for x in c]
	c=sum(c,[])
	c=[x*x for x in c if x<>0] #x=0 survient avec une probabilité nulle
	c.sort(reverse=True)
	c=[np.log(sum(c[n:len(c)])) for n in range(len(c))]
	L+=[c]
    return L
    
    
