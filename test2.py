import numpy as np
import random


def AF_tanh(H):   #activation function tanh
    H_out = np.matrix([[0.0],[0.0],[0.0]]) #(3,1)      # after Activation fuction
    H_out_d = np.matrix([[0.0],[0.0],[0.0]]) #(3,1)    # differential
    for i in range(len(H)):
        output = (np.exp(H[i]) - np.exp(-H[i])) / (np.exp(H[i]) + np.exp(-H[i]))
        H_out[i] = output
        H_out_d[i] = 1-output**2  #if it doesn't work, seperate H_out and H_out_d into two def
    return H_out, H_out_d

def AF_sigmoid(H):   #activation function tanh
    H_out = np.matrix([[0.0],[0.0],[0.0]]) #(3,1)      # after Activation fuction
    H_out_d = np.matrix([[0.0],[0.0],[0.0]]) #(3,1)    # differential
    for i in range(len(H)):
        output = 1/(1+np.exp(-H[i])) 
        H_out[i] = output
        H_out_d[i] = output*(1-output) 
    return H_out, H_out_d





ran_w1 = np.random.rand(6)  #list
arr_w1 = ran_w1.reshape(3,2)    #array
W1 = np.asmatrix(arr_w1, dtype=float)   #matrix

ran_w2 = np.random.rand(6)  #list
arr_w2 = ran_w2.reshape(2,3)    #array
W2 = np.asmatrix(arr_w2, dtype=float)   #matrix

ran_w3 = np.random.rand(2)  #list
arr_w3 = ran_w3.reshape(1,2)    #array
W3 = np.asmatrix(arr_w3, dtype=float)   #matrix

X = np.matrix([[random.randrange(0,101)],[random.randrange(0,101)]])
print("x:", X)

GT = X[0] + X[1] #Ground Truth
print("GT:", GT)

#feed forward
H1 = np.matrix([[0.0],[0.0],[0.0]]) # hidden_layer_1 (3,1)
H2 = np.matrix([[0.0],[0.0]])       # hidden_layer_2 (2,1)
        
H1 = np.matmul(W1, X) #(3,2)x(2,1) = (3,1)
print("H1: ", H1)


print("\n\n------------------------------")
print("\ntanh")
H1_out, H1_out_d = AF_tanh(H1)
print("\nH1_out: ", H1_out)
# print("H1_out_d: ", H1_out_d)

H2 = np.matmul(W2, H1_out) #(2,3)x(3,1) = (2,1)
print("H2: ", H2)
Y = np.matmul(W3, H2) # (1,2)x(2,1) = 1

print("\nY:", Y)

Error = (1/2)*((GT-Y)**2)
Error_d = -(GT-Y)

print("\nError:", Error)
print("Error_d:", Error_d)

#=========================================================

print("------------------------------")
print("\nsigmoid")

H1_out, H1_out_d = AF_sigmoid(H1)
print("\nH1_out: ", H1_out)
# print("H1_out_d: ", H1_out_d)

H2 = np.matmul(W2, H1_out) #(2,3)x(3,1) = (2,1)
print("H2: ", H2)
Y = np.matmul(W3, H2) # (1,2)x(2,1) = 1

print("\nY:", Y)


Error = (1/2)*((GT-Y)**2)
Error_d = -(GT-Y)

print("\nError:", Error)
print("Error_d:", Error_d)
