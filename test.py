import numpy as np
import random
import matplotlib.pyplot as plt


# a = np.full((3,2),0)
# print(a)

# b = list(a)
# print(b)
# print(np.transpose(b))

# b=np.full((len(b),len(b[0])),0.5)+b
# print(b)

# c = np.random.random(1)
# print(c)

# g = [1, 5, 3 ,5 ,6, 4, 2]
# d = np.linspace(-5,5,6)
# print(d)
# dd = d.reshape(2,3)
# print(dd)

# rand_w1 = np.random.rand(4)
# print(rand_w1,"\n")
# arr_w1 = rand_w1.reshape(2,2)
# print(arr_w1,"\n")
# W1 = np.asmatrix(arr_w1, dtype=float)
# print(W1,"\n")

# hidden_layer_1 = np.matrix([[1.0,2.0]])
# print("1:", hidden_layer_1)
# hidden_layer_1 = hidden_layer_1 * 0.5
# print("2:", hidden_layer_1)
# print(np.shape(hidden_layer_1))

# print("\n",hidden_layer_1[0][0])



# ran_w1 = np.random.rand(6)  #list
# arr_w1 = ran_w1.reshape(3,2)    #array
# W1 = np.asmatrix(arr_w1, dtype=float)   #matrix
# print("W1:",W1,'\n', np.shape(W1))
# print("\n",W1[2,1])

# Error_d_W3 = np.matrix([[3.4,6.5]])
# print(Error_d_W3[0,0])

# H1 = np.matrix([[5.0],[3.0],[2.0]])
# print(H1[2])



# Z = np.matmul(W1, X)
# print("Z:", Z, "\n", "Z[2]:",Z[2],"\n")

# C = Z
# print(C, np.shape(C))
# print(C.type())


# x = np.random.rand()
# print(x)
# print(np.shape(x))

# X = np.matrix([[np.random.rand()],[np.random.rand()]])
# print("x:", X, '\n')
# print(np.shape(X))

# ran_w1 = np.random.rand(6)  #list
# print(ran_w1,'\n')
# arr_w1 = ran_w1.reshape(3,2)    #array
# print(arr_w1)
# print(type(arr_w1))
# W1 = np.asmatrix(arr_w1, dtype=float)   #matrix

# print("\n",W1)
# print(type(W1))

#============================================

# ran_X = np.random.rand(20)  #list
# arr_X = ran_X.reshape(10,2)    #array
# X = np.asmatrix(arr_X, dtype=float)  
# # print("x:", X, '\n')  
# # print(len(X)) 

# GT = [X[i,0]+ X[i,1] for i in range(len(X))]  #(1,10)
# print(GT)
# print(len(GT))
# Y = np.random.rand(10)
# print(Y)
# print(len(Y))

# Error_d_W1 = np.zeros((len(Y),3,2))

# Error = [(1/2)*((GT-Y[i])**2) for i in range(len(Y))]  #cost => minimize

# Error_d = [(GT-Y) for i in range(len(Y))]


# # for i in range(3): #3
# #     for j in range(2): #2  len(W1[0]) -> 1 ??
# #         Error_d_W1[i,j] = Error_d*5

# # print(Error_d_W1)

# for k in range(len(Y)):
#     for i in range(3): #3
#         for j in range(2): #2  len(W1[0]) -> 1 ??
#             Error_d_W1[k,i,j] = Error_d * X[i,j]

# print(Error_d_W1)



# W = np.zeros((3,2)) + 3.5
# print(W)   
# ran_X = np.random.rand(6)  #list
# arr_X = ran_X.reshape(2,3)    #array
# X = np.asmatrix(arr_X, dtype=float)  
# print("x:", X, '\n')  

# print(W*X)
# print(np.matmul(W,X))

# print(W*2)


# W = np.zeros((2,4,3)) + 3.5
# W[1] = W[1] + 4.5
# print(W)
# print("----------------------")
# print(W.mean(axis=0))



# ran_X = np.random.rand(20)  #list
# arr_X = ran_X.reshape(10,2)    #array
# train_X = np.asmatrix(arr_X, dtype=float)  
# print(len(train_X))


# E = np.zeros((1,10))
# print(E[1])

# K = np.zeros((10))
# print(K[7])

# ran_X = np.random.rand(2000)  #list
# arr_X = ran_X.reshape(1000,2)
# print(arr_X)
# train_X = np.asmatrix(arr_X, dtype=float) 
# print("-------------------------------------")
# print(train_X)
# train_GT = np.matrix(np.zeros((len(train_X),1))) 
# for i in range(len(train_X)):
#     train_GT[i,0] = train_X[i,0]+ train_X[i,1] 

# print("GT:", train_GT)


# X_GT = np.zeros(1000,3)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def relu(x):
    return np.maximum(x,0)

def leaky_relu(x):
    a = 0.1
    return np.maximum(a*x, x)

# def ELU(x):
#     a = 0.1
#     if x.all() > 0:
#         return x 
#     else:
#         return a*(np.exp(x) -1)

def swish(x):
    return x* sigmoid(x)

def elu(z,alpha):
	return z if z >= 0 else alpha*(e^z -1)

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = tanh(x)
y3 = relu(x)
y4 = leaky_relu(x)
# y5 = ELU(x)
y6 = swish(x)


# plt.subplot(2,2,1)
# plt.plot(x, y1)
# plt.title('sigmoid')
# plt.ylim(-0.1, 1.1)
# plt.grid(True)


# plt.subplot(2,2,2)
# plt.plot(x, y2 )
# plt.title('tanh')
# plt.ylim(-1.1, 1.1)
# plt.grid(True)


# plt.subplot(2,2,3)
# plt.plot(x, y3)
# plt.ylim(-0.1, 5.0)
# plt.title('ReLU')
# plt.grid(True)


# plt.subplot(2,2,4)
# plt.plot(x, y4)
# plt.title('Leaky ReLU')
# plt.ylim(-1.1, 5.0)
# plt.grid(True)
# plt.show()


# plt.plot(x, y5)
# plt.title('ELU')
# plt.ylim(-1.1, 5.0)
# plt.grid(True)
# plt.show()

plt.plot(x, y6)
plt.title('Swish')
plt.ylim(-1.1, 5.0)
plt.grid(True)
plt.show()


def elu(x):
    a=1
    if x>0:
        return x
    else:
        return a*(np.exp(x)-1)

x_range = np.arange(-3., 3., 0.1)
y_range = np.array([elu(x) for x in x_range])

plt.plot(x_range, y_range, label='ELU')

plt.ylim([-1.0, 3.0])
plt.xlim([-3.0, 3.0])
plt.grid(which='major')
plt.title('ELU')
plt.show()






# plt.title('Leaky ReLU Function')
# plt.show()

#     plt.subplot(2,1,1)
#     plt.plot(loss, label='Loss')
#     plt.legend()
#     plt.subplot(2,1,2)
#     plt.plot(loss_d, label='Loss\' Gradient')
#     plt.legend()
#     plt.tight_layout()    
#     plt.show()