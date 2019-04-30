import numpy as np


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
iterations=1
alpha=0.01
# Y=np.random.rand(1,traininglength) #output
# X=np.random.rand(inputnr,traininglength) #inputs
Y=np.array([3,5,7,9,11,13])
m=len(Y)    #numer training examples
Y=Y.reshape(1,m)
X=np.array([1,2,3,4,5,6])
X=X.reshape(1,m)                 # [x(1) x(2) x(3) ,..]
Prediction=10

inputnr,traininglength=X.shape

w={}; b={}; A={}; Z={}; dZ={}; #dW={}; db={};
#initialization w,b
for i in range(1,len(layers)+1):
    if i==1:
        w['w{0}'.format(i)]=np.random.rand(inputnr,layers[i-1])
    if i==len(layers)+1:
        w['w{0}'.format(i)]=np.random.rand(layers[i-1],1)
    else:
        w['w{0}'.format(i)]=np.random.rand(layers[i-2],layers[i-1])
    b['b{0}'.format(i)]=np.zeros((layers[i-1],1))


def forward(w,b,A):
    Z=np.dot(np.transpose(w),A)+b
    A1=ReLU(Z)
    return A1,Z

A['A{0}'.format(0)]=X
for j in range(0,iterations):

    for i in range(1,len(layers)+1):
        A2,Z2=forward(w['w'+str(i)],b['b'+str(i)],A['A'+str(i-1)])

        A['A{0}'.format(i)]=A2
        Z['Z{0}'.format(i)]=Z2

    for l in range(1,len(layers)+1):
        index=len(layers)+1-(l)   #start with last layers
        print('1',(w['w'+str(index)]).shape)#print(index)
        if index==len(layers):   #last layers
            dZ['dZ{0}'.format(index)]=A['A'+str(index)]-Y
        else:
            #print(np.transpose(w['w'+str(index+1)]).shape)
            # print(dZ['dZ'+str(index+1)].shape)
            # print(dReLU(Z['Z'+str(index)]).shape)
            dZ['dZ{0}'.format(index)]=((w['w'+str(index+1)])@dZ['dZ'+str(index+1)])*dReLU(Z['Z'+str(index)])
        #dW['dW{0}'.format(index)]=dZ['dZ'+str(index)]@np.transpose(A['A'+str(index-1)])/m
        #db['db{0}'.format(index)]=np.sum(dZ['dZ'+str(index)],axis=1,keepdims=True)/m
        #print('dZ',dZ['dZ'+str(index)].shape)
        #print('A',A['A'+str(index-1)].shape)
        db=np.sum(dZ['dZ'+str(index)],axis=1,keepdims=True)/m
        dW=A['A'+str(index-1)]@np.transpose(dZ['dZ'+str(index)])/m
        #print('dW',dW.shape)
        w['w{0}'.format(index)]=w['w'+str(index)]-alpha*dW
        b['b{0}'.format(index)]=b['b'+str(index)]-alpha*db
        print('2',(w['w'+str(index)]).shape)

#dZ2=np.array(A2-Y)
#     dW2=1/m*dZ2@np.transpose(A1)
#     db2=1/m*np.sum(dZ2,axis=1,keepdims=True)
#     dZ1=(np.transpose(w2)*dZ2)*ReLU(Z1)
#     dW1=1/m*dZ1@np.transpose(X)
#     db1=1/m*np.sum(dZ1,axis=1,keepdims=True)
#
#     # print('w1',w1,'w2',w2)
#     # print('b1',b1,'b2',b2)
#     #for i in range(0,layers):
# ## forward propagation
#     #Z1=w1@X+b1
#     Z1=np.dot(w1,X)+b1
#         #A=np.tanh(Z)
#     A1=ReLU(Z1)
#     Z2=np.dot(w2,A1)+b2
#     A2=ReLU(Z2)
#     J=np.sum((A2-Y)**2)/2/m
#
# ## backward propagation
#     dZ2=np.array(A2-Y)
#     dW2=1/m*dZ2@np.transpose(A1)
#     db2=1/m*np.sum(dZ2,axis=1,keepdims=True)
#     dZ1=(np.transpose(w2)*dZ2)*ReLU(Z1)
#     dW1=1/m*dZ1@np.transpose(X)
#     db1=1/m*np.sum(dZ1,axis=1,keepdims=True)
#     # print('dZ2',dZ2,dZ2.shape)
#     # print('dW2',dW2,dW2.shape)
#     # print('db2',db2,db2.shape)
#     # print('dRelu(Z1)',dReLU(Z1))
#     # print('dZ1',dZ1)
#     # print('dW1',dW1)
#     # print('db1',db1)
#
#
#     w1=w1-alpha*dW1
#     b1=b1-alpha*db1
#     w2=w2-alpha*dW2
#     b2=b2-alpha*db2
#     print('Error',J)
# #print('w,b',w,b1)
# print('result',ReLU(w2@ReLU(w1*Prediction+b1)+b2))
