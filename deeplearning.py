import numpy as np
import time

def sigmoid(Z):
    sigm=1./(1+np.exp(-Z))
    return sigm

def ReLU(Z):
    r=np.maximum(0,Z)
    return r

def dReLU(A):
    n,m=A.shape
    r=np.zeros((n,m))
    for i in range(0,n):
        for j in range(0,m):
            if A[i,j]<0:
                r[i,j]=0
            else:
                r[i,j]=1
    return r


layers=np.array([2,4,3,2,1])
iterations=1000
alpha=0.001

Y=np.array([3,5,7,9,11,13])
m=len(Y)    #numer training examples
Y=Y.reshape(1,m)
X=np.array([1,2,3,4,5,6])
X=X.reshape(1,m)                 # [x(1) x(2) x(3) ,..]

Prediction=10
inputnr,traininglength=X.shape

#initialization w,b
w={}; b={}; A={}; Z={}; dZ={};
tic=time.time()
for i in range(1,len(layers)+1):
    if i==1:
        w['w{0}'.format(i)]=np.random.rand(inputnr,layers[i-1])
    if i==len(layers)+1:
        w['w{0}'.format(i)]=np.random.rand(layers[i-1],1)
    else:
        w['w{0}'.format(i)]=np.random.rand(layers[i-2],layers[i-1])
    b['b{0}'.format(i)]=np.zeros((layers[i-1],1))


A['A{0}'.format(0)]=X
for j in range(0,iterations):

    for i in range(1,len(layers)+1):
        Z2=np.dot(np.transpose(w['w'+str(i)]),A['A'+str(i-1)])+b['b'+str(i)]
        A2=ReLU(Z2)
        A['A{0}'.format(i)]=A2
        Z['Z{0}'.format(i)]=Z2

    J=np.sum((A['A'+str(i)]-Y)**2/2/m)

    for l in range(1,len(layers)+1):
        index=len(layers)+1-(l)   #start with last layers
        if index==len(layers):   #last layers
            dZold=A['A'+str(index)]-Y
            dZ=A['A'+str(index)]-Y

        else:
            dZ=((w['w'+str(index+1)])@dZold)*dReLU(Z['Z'+str(index)])
        db=np.sum(dZ,axis=1,keepdims=True)/m
        dW=A['A'+str(index-1)]@np.transpose(dZ)/m

        w['w{0}'.format(index)]=w['w'+str(index)]-alpha*dW
        b['b{0}'.format(index)]=b['b'+str(index)]-alpha*db

        dZold=[];
        dZold=dZ
        dZ=[]

    print('Error',J)
toc=time.time()
print('time',toc-tic)
