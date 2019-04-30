import numpy as np

## Training Data
#traininglength=10
neurons=2
layers=1
#inputnr=3
iterations=100
alpha=0.12
# Y=np.random.rand(1,traininglength) #output
# X=np.random.rand(inputnr,traininglength) #inputs
Y=np.array([3,5,7,9,11,13])
m=len(Y)    #numer training examples
Y=Y.reshape(1,m)
X=np.array([1,2,3,4,5,6])
X=X.reshape(1,m)                 # [x(1) x(2) x(3) ,..]
Prediction=10
print(X.shape)
inputnr,traininglength=X.shape

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
            if Z[i,j]<0:
                r[i,j]=0
            else:
                r[i,j]=1
    return r


#initialize parameters
w=[]; Z=[]; A=[];
w=np.random.rand(inputnr,neurons)*0.01   #=nr of neurons in layer
w=np.transpose(w,axes=None)
b1=np.zeros((1,1))

for j in range(0,iterations):


    #for i in range(0,layers):
## forward propagation
    #Z1=w1@X+b1
    Z1=np.dot(w,X)+b1
        #A=np.tanh(Z)
    A1=ReLU(Z1)
    J=np.sum((A1-Y)**2)/2/m
    # Z2=w2@A1+b2
    # A2=ReLU(Z2)
    # Z3=w3@A2+b3
    # A3=ReLU(Z3)
    #print('A',A1)
    # print('Y',Y)

## backward propagation
    dZ1=A1-Y

    dW=1/m*dZ1@np.transpose(X)
    db=1/m*np.sum(dZ1,axis=1,keepdims=True)
    # dZ3=dReLU(A3)
    # dW3=1/m
    #print(dZ)
    #dZ=1-(np.tanh(A))**2
    # dW=1/m*dZ@np.transpose(A)
    # db=1/m*np.sum(dZ,axis=1,keepdims=True)
    #print(db-dZ)
    #print(dW,dW.shape)
    #print(db,db.shape)
    w1=w1-alpha*dW
    b1=b1-alpha*db
    print('Error',J)
print('w,b',w1,b1)
print('result',ReLU(w1*Prediction+b1))
