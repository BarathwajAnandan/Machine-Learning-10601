import numpy as np

def sigmoid(a):
    return 1/(1+np.exp(-1*a))
def softmax(b):
    return np.exp(b)/np.sum(np.exp(b))
def loss(y_target,y):
    return -1* np.sum( np.multiply(y_target,np.log(y)))
    
x = np.asarray([1,1,1,0,0,1,1])
w1 = np.asarray([[1,1,2,-3,0,1,-3],[1,3,1,2,1,0,2],[1,2,2,2,2,2,1],[1,1,0,2,1,-2,2]])
w2 = np.asarray([[1,1,2,-2,1],[1,1,-1,1,2],[1,3,1,-1,1]])
w1_ = w1[:,1:]
w2_ = w2[:,1:]
y_target = np.asarray([0,1,0])
a = np.dot(w1,x)

z_  = sigmoid(a)
z = np.hstack((1,z_))
b = np.dot(w2,z)

y = softmax(b) 
l = loss(y_target,y)

#backprop!


dl_db= np.reshape(dl_db,(3,1))
z = np.reshape(z,(1,5))

dl_dW2 = np.dot(dl_db,z)


dl_dz = np.dot(dl_db.T,w2_).T

z_ = (z_ * (1- z_)).reshape(4,1)dl_db =  y - y_target
print(dl_dz.shape,z_.shape)
dl_da = np.multiply(dl_dz,z_)

print(dl_da.shape,x.shape)
dl_dW1 = np.dot(dl_da,x.reshape(1,7))










    
    
    

      
    
    
    