"""
x1,x2 0~100 아무숫자
back -> forward
"""

import random
import numpy as np
from math import exp
import matplotlib.pyplot as plt
import itertools


global Learning_rate
Learning_rate=0.5

def A_F_tanh (H):  #activation function tanh
    H_out = np.full((3,1),0)
    H_out_d = np.full((3,1),0)
    tmp =[]
    H_test = np.full((3,1),0)

    for i in range(len(H)):
        # output = (exp(H[i]) - exp(-H[i]))/(exp(H[i]) + exp(-H[i]))
        # H_out[i] = output 
        # # print(output)
        # H_out_d[i] = 1-output**2
        output = (np.power(np.e, H[i]) - np.power(np.e, -H[i]))/(np.power(np.e, H[i]) + np.power(np.e, -H[i]))
        tmp.append(output)
        H_out[i] = output
        H_out_d[i] = 1 - H_out[i]**2
        print("herehrehrhehrehrehrhe")
        print(H_out[i],H_out_d[i], output, output*output)
    return H_out, H_out_d
    # return output, 1 - output**2

def A_F_sigmoid (H):  #activation function sigmoid
    H_out = np.full((3,1),0)
    H_out_d = np.full((3,1),0)
    for i in range(len(H)):
        # print(H[i])
        output = 1/(1 + exp(-H[i]))   #1/(1+np.power(np.e, -x))    
        H_out[i] = output
        # print(output)
        H_out_d[i] = output*(1-output)
    return H_out, H_out_d

def Error (y,GT):
    # Error = np.round((1/2)*((GT-y)**2), 3)
    # print(Error)
    Error = (1/2)*((GT-y)**2)
    Error_d = (GT-y)
    return Error, Error_d

def backward_pass(x1,x2,w1,w2,w3,b1,b2,b3,H1,H2,y):
    #initialize
    Error_d_w1 = np.full((len(w1),len(w1[0])),0) #(3,2)
    Error_d_w2 = np.full((len(w2),len(w2[0])),0) #(2,3)
    Error_d_w3 = np.full((len(w3),len(w3[0])),0) #(2,1)  
    Error_d_b1 = np.full((len(b1),len(b1[0])),0) #(3,1)
    Error_d_b2 = np.full((len(b2),len(b2[0])),0) #(2,1)
    Error_d_b3 = 0 #1

    Error_y, Error_y_d = Error(y)  #return Error, Error_d
    
    x= [x1,x2]
    H1_out, H1_out_d = A_F_tanh(H1)
    # H1_out, H1_out_d = A_F_sigmoid(H1)

    print("================================check in back propagation==========================")

    #calculate error differntial for each w,b   
    for i in range(len(w1)): #3
        for j in range(len(w1[0])): #2
            Error_d_w1[i][j] = Error_y_d*(w3[0][0]*w2[0][i]*H1_out_d[i]*x[j] \
                                        + w3[0][1]*w2[1][i]*H1_out_d[i]*x[j] )

            print('Error_d_w1[{}][{}]:'.format(i,j), Error_d_w1[i][j])
            print('w31:', w3[0][0],'w32:', w3[0][1])
            print('w2[0][{}]:'.format(i), w2[0][i])
            print('w2[1][{}]:'.format(i), w2[1][i])
            print('H1_out_d[{}]:'.format(i), H1_out_d[i])
            print('x[{}]:'.format(j), x[j], '\n')

    for i in range(len(w2)): #2
        for j in range(len(w2[0])): #3
            Error_d_w2[i][j] = Error_y_d * w3[0][i] * H1_out[j]

    for i in range(len(w3)):
            Error_d_w3[i] = Error_y_d * H2[i] 

    for i in range(len(b1)): 
        Error_d_b1[i] = Error_y_d*(w3[0][0]*w2[0][i]*H1_out_d[i]*1 \
                                        + w3[0][1]*w2[1][i]*H1_out_d[i]*1 )
                                        
    for i in range(len(b2)): #2
        Error_d_b2[i] = Error_y_d* w3[0][i] * 1                                      

    Error_d_b3 = Error_y_d * 1 
    
    print('\n each Error check\n ---------------------------------------------------------------------')
    print("Error_d_w1 \n", Error_d_w1)
    print("Error_d_w2 \n", Error_d_w2)
    print("Error_d_w3 \n", Error_d_w3)
    print("Error_d_b1 \n", Error_d_b1)
    print("Error_d_b2 \n", Error_d_b2)
    print("Error_d_b3 \n", Error_d_b3, '\n')

    print("=============================================check==========================")

    #update
    w1 = w1 + np.full((len(w1),len(w1[0])),Learning_rate)*Error_d_w1  
    w2 = w2 + np.full((len(w2),len(w2[0])),Learning_rate)*Error_d_w2
    w3 = w3 + np.full((len(w3),len(w3[0])),Learning_rate)*Error_d_w3
    b1 = b1 + np.full((len(b1),len(b1[0])),Learning_rate)*Error_d_b1
    b2 = b2 + np.full((len(b2),len(b2[0])),Learning_rate)*Error_d_b2
    b3 = b3 + Learning_rate*Error_d_b3
    
    # w1 = np.squeeze(w1)
    # w2 = np.squeeze(w2)
    # w3 = np.squeeze(w3)
    # b1 = np.squeeze(b1)
    # b2 = np.squeeze(b2)
    # b3 = np.squeeze(b3)

    return w1,w2,w3,b1,b2,b3, Error_y, Error_y_d


#initalize
# x1 = random.randrange(0,101)
# x2 = random.randrange(0,101)

x1 = np.round(np.random.random(1), 3)
x2 = np.round(np.random.random(1), 3)
x = np.matrix([x1, x2])   #(1,2) => need to transpose
GT = x1 + x2 #Ground Truth

w1 = np.round(np.random.random((3,2)), 3)
w2 = np.round(np.random.random((2,3)), 3)
w3 = np.round(np.random.random((1,2)), 3)

b1 = np.round(np.random.random((3,1)), 3)/2
b2 = np.round(np.random.random((2,1)), 3)/2
b3 = np.round(np.random.random(1), 3)/2


print("initializing check")
print("--------------------------------------")
print("x:", x)
print('w1:',w1,'\nw2:',w2,'\nw3:',w3,'\nb1:',b1,'\nb2:',b2,'\nb3:',b3)

H1 = w1.dot(x) + b1 #(3,1)
print("H1: ", H1)
H1_out, H1_out_d = A_F_tanh(H1)
# H1_out, H1_out_d = A_F_sigmoid(H1)
print("H1_out: ", H1_out, '\n', "H1_out_d: ", H1_out_d)
H2 = w2.dot(H1_out) + b2
y = w3.dot(H2) +b3
Error_y, Error_y_d = Error(y)

print('\nGT:',GT,'/ y:',y)
print('Error:',Error_y)
print('Error_d:', Error_y_d,'\n')

w1,w2,w3,b1,b2,b3,Error_y,Error_y_d = backward_pass(x1,x2,w1,w2,w3,b1,b2,b3,H1,H2,y)

print("parameters check after backpropagation")
print("--------------------------------------")
print('w1:',w1,'\nw2:',w2,'\nw3:',w3,'\nb1:',b1,'\nb2:',b2,'\nb3:',b3, '\n')


check_Error = []
check_Error_d = []

for i in range(10):
    H1 = w1.dot(x) + b1
    H1_out, H1_out_d = A_F_tanh(H1)
    # H1_out, H1_out_d = A_F_sigmoid(H1)
    H2 = w2.dot(H1_out) + b2
    y = w3.dot(H2) +b3
    Error_y = Error(y)[0]

    w1,w2,w3,b1,b2,b3, Error_y, Error_y_d = backward_pass(x1,x2,w1,w2,w3,b1,b2,b3,H1,H2,y)
    check_Error.append(Error_y.tolist())
    check_Error_d.append(Error_y_d.tolist())

    if i%100 == 0:
        print("====================================")
        print("no.{}".format(i+1),'GT:',GT,'\ty:',y)
        print("no.{}".format(i+1),'Error:',Error_y)

print("\n\n Last\n")
print('GT:',GT,'\ty:',y)
print('Error:',Error_y,"")



check_Error = list(itertools.chain.from_iterable(check_Error))
check_Error = list(itertools.chain.from_iterable(check_Error))
check_Error_d = list(itertools.chain.from_iterable(check_Error_d))
check_Error_d = list(itertools.chain.from_iterable(check_Error_d))


plt.subplot(2,1,1)
plt.plot(check_Error, label='Loss')
plt.legend()
plt.subplot(2,1,2)
plt.plot(check_Error_d, label='Loss\' Gradient')
plt.legend()
plt.tight_layout()    
plt.show()
