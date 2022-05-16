import numpy as np
import matplotlib.pyplot as plt
import cv2
delta = [0]

Nrand = 0
Arand = 0
GaussAdd = 0
GaussFac = 0

def InitGauss(seed:"seed value for random number generator"):
    global Nrand,Arand,GaussAdd,GaussFac
    Nrand = 4
    Arand = 2**31-1
    GaussAdd = (3*Nrand)**0.5
    GaussFac = 2*GaussAdd/(Nrand*Arand)
    np.random.seed(seed)

def Gauss()->"random number":
    sum = 0
    for i in range(Nrand):
        sum += np.random.random()
#    return (GaussFac*sum-GaussAdd)
    return np.random.random()*3
def MidPointBM(maxlevel,sigma,seed):
    InitGauss(seed)
    for i in range(maxlevel):
        delta.append(sigma*0.5**((i+1)/2))
    N = 2**maxlevel
    X = np.zeros((N+1,1))
    X[0] = 0; X[N] = sigma*Gauss()
    MidPointRecursion(X,0,N,0,maxlevel)
    return X

def MidPointRecursion(X,index0,index2,level,maxlevel):
    index1 = (index0 + index2) >>1
    X[index1] = 0.5*(X[index0] + X[index2]) + delta[level]*Gauss()
    if level<maxlevel:
        MidPointRecursion(X,index0,index1,level+1,maxlevel)
        MidPointRecursion(X,index1,index2,level+1,maxlevel)
    return X

def MidPointFM1D(maxlevel,sigma,H,seed):
    global delta
    InitGauss(seed)
    delta = [0]
    for i in range(maxlevel+5):
        delta.append(sigma*0.5**(i*H)*(1-2**(2*H-2)**0.5))
    N = 2**maxlevel
    X = np.zeros((N+1,1))
    X[0] = 0; X[N] = sigma*Gauss()
    MidPointRecursion(X,0,N,0,maxlevel)
    return X

def AdditionsFM1D(maxlevel,sigma,H,seed):
    global delta
    delta = [0]
    InitGauss(seed)
    for i in range(maxlevel+5):
        delta.append(sigma*0.5**(i*H)*(0.5**0.5)*(1-2**(2*H-2))**0.5)
    N = 2**maxlevel
    X = np.zeros((N+1,1))
    X[0]=0;X[N]=sigma*Gauss()
    D = N
    d = D >> 1
    level = 1
    while level <= maxlevel:
        for i in range(d,N-d,D):
            X[i] = 0.5*(X[i-d] + X[i+d])
        for i in range(0,N,d):
            X[i] = X[i] + delta[level]*Gauss()
        D = D >> 1
        d = d >> 1
        level = level + 1
    return X

def f3(delta,x0,x1,x2):
    return (x0+x1+x2)/3 + delta*Gauss()

def f4(delta,x0,x1,x2,x3):
    return  (x0+x1+x2+x3)/4 + delta*Gauss()

def MidPointFM2D(maxlevel,sigma,H,addition,seed):
    InitGauss(seed)
    N = 2**maxlevel

    delta = sigma
    X = np.zeros((N+1,N+1))
    X[0, 0] = delta * Gauss()
    X[0, N] = delta * Gauss()
    X[N, 0] = delta * Gauss()
    X[N, N] = delta * Gauss()
    D = N
    d = D >> 1
    for stage in range(1,maxlevel+1):
        delta = delta * 0.5**(0.5*H)
        for x in range(d,N-d+1,D):
            for y in range(d,N-d+1,D):
                X[x,y] = f4(delta,X[x+d,y+d],X[x+d,y-d],X[x-d,y+d],X[x-d,y-d])

        if addition:
            for x in range(0,N+1,D):
                for y in range(0,N+1,D):
                    X[x,y] = X[x,y] + delta*Gauss()
        delta = delta * 0.5**(0.5*H)
        for x in range(d,N-d+1,D):
            X[x, 0] = f3(delta, X[x + d, 0], X[x - d, 0], X[x, d])
            X[x, N] = f3(delta, X[x + d, N], X[x - d, N], X[x, N - d])
            X[0, x] = f3(delta, X[0, x+d], X[0, x-d], X[d, x])
            X[N, x] = f3(delta, X[N, x+d], X[N, x-d], X[N-d, x])

        for x in range(d,N-d+1,D):
            for y in range(D,N-d+1,D):
                X[x,y] = f4(delta,X[x,y+d],X[x,y-d],X[x+d,y],X[x-d,y])

        for x in range(D,N-d+1,D):
            for y in range(d,N-d+1,D):
                X[x,y] = f4(delta,X[x,y+d],X[x,y-d],X[x+d,y],X[x-d,y])

        if addition:
            for x in range(0,N+1,D):
                for y in range(0,N+1,D):
                    X[x,y] += delta*Gauss()

        D = D >> 1
        d = d >> 1
    return X

def gray_save(x):
    rows,cols = x.shape
    m = 255
    img = np.zeros((rows,cols))
    x = m * (x - np.min(x))/(np.max(x)-np.min(x))
    img = np.int32(x)
    cv2.imwrite("test.bmp",img)



if __name__ == "__main__":
    x = MidPointFM2D(13,1.2,0.05,True,54646)
    gray_save(x)