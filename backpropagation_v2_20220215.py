"""
x1,x2 0~100 아무숫자
back -> forward
"""

import random
import numpy as np
from math import exp

global Learning_rate
Learning_rate=0.5

def A_F_tanh (H):  #activation function tanh
    H_out = np.full((3,1),0)
    H_out_d = np.full((3,1),0)
    print("11111111111111")
    print(H, '\n', H_out)
    for i in range(len(H)):
        # print(H[i])
        # output = (exp(H[i]) - exp(-H[i]))/(exp(H[i]) - exp(-H[i]))  
        output = (np.power(np.e, H[i]) - np.power(np.e, -H[i]))/(np.power(np.e, H[i]) + np.power(np.e, -H[i]))
        H_out[i] = output
        # print(output)
        H_out_d[i] = 1 - np.power(np.e, -H[i])**2
    return H_out, H_out_d

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

def Error (y):
    Error = (1/2)*((GT-y)**2)
    Error_d = -(GT-y)
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

    #calculate error differntial for each w,b   
    for i in range(len(w1)): #3
        for j in range(len(w1[0])): #2
            Error_d_w1[i][j] = Error_y_d*(w3[0][0]*w2[0][i]*H1_out_d[i]*x[j] \
                                        + w3[0][1]*w2[1][i]*H1_out_d[i]*x[j] )
    
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

    return w1,w2,w3,b1,b2,b3









#initalize
# x1 = random.randrange(0,101)
# x2 = random.randrange(0,101)
# x = np.array([x1, x2])   #(1,2) => need to transpose
# GT = x1 + x2 #Ground Truth
x = np.random.choice(101,(2,1))
GT = x[0] + x[1] #Ground Truth

w1 = np.round(np.random.normal(0, 0.5, (3,2)), 3)
w2 = np.round(np.random.normal(0, 0.5, (2,3)), 3)
w3 = np.round(np.random.normal(0, 0.5, (1,2)), 3)

b1 = np.round(np.random.normal(0, 0.5, (3,1)), 3)
b2 = np.round(np.random.normal(0, 0.5, (2,1)), 3)
b3 = np.round(np.random.normal(0, 0.5, 1), 3)


# w1 = np.round(np.random.random((3,2)) + 1, 3)
# w2 = np.round(np.random.random((2,3)) + 1, 3)
# w3 = np.round(np.random.random((1,2)) + 1, 3)

# b1 = np.round(np.random.random((3,1)) + 1, 3)
# b2 = np.round(np.random.random((2,1)) + 1, 3)
# b3 = np.round(np.random.random(1) + 1, 3)

print("initializing check")
print('w1:',w1,'\nw2:',w2,'\nw3:',w3,'\nb1:',b1,'\nb2:',b2,'\nb3:',b3)

# H1 = w1.dot(np.transpose(x)) + b1 #(3,1)
H1 = w1.dot(x) + b1 #(3,1)
print(H1)
H1_out = A_F_tanh(H1)[0]  
# H1_out = A_F_sigmoid(H1)[0]
H2 = w2.dot(H1_out) + b2
y = w3.dot(H2) +b3
Error_y = Error(y)[0]

print('\nGT:',GT,'/ y:',y)
print('Error:',Error_y,"\n")

w1,w2,w3,b1,b2,b3 = backward_pass(x[0],x[1],w1,w2,w3,b1,b2,b3,H1,H2,y)

print("parameters check after backpropagation")
print('w1:',w1,'\nw2:',w2,'\nw3:',w3,'\nb1:',b1,'\nb2:',b2,'\nb3:',b3)

for i in range(1000):
    H1 = w1.dot(np.transpose(x)) + b1
    H1_out = A_F_tanh(H1)[0]
    # H1_out = A_F_sigmoid(H1)[0]
    H2 = w2.dot(H1_out) + b2
    y = w3.dot(H2) +b3
    Error_y = Error(y)[0]

    w1,w2,w3,b1,b2,b3 = backward_pass(x1,x2,w1,w2,w3,b1,b2,b3,H1,H2,y)

    if i%100 == 0:
        print("====================================")
        print("no.{}".format(i+1),'GT:',GT,'\ty:',y)
        print("no.{}".format(i+1),'Error:',Error_y)

print("\n\n Last\n")
print('GT:',GT,'\ty:',y)
print('Error:',Error_y,"")


# while Error ==0:
#     H1 = w1.dot(np.transpose(x)) + b1 #(3,1)
#     H1_out = A_F_tanh(H1)[0]
#     # H1_out = A_F_sigmoid(H1)[0]
#     H2 = w2.dot(H1_out) + b2    
#     y = w3.dot(H2) +b3
#     Error_y = Error(y)[0]
#     print(Error_y)
#     w1,w2,w3,b1,b2,b3 = backward_pass(x1,x2,w1,w2,w3,b1,b2,b3,H1,H2,y)
#     print(GT,y)
    
# =========================================================================


