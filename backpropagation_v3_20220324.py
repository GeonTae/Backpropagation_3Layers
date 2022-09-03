"""
x1,x2 0~100 아무숫자
back -> forward
"""

import random
import numpy as np
from math import exp

global Learning_rate
Learning_rate=0.5


class deeplearning:

    def AF_tanh(H):   #activation function tanh
        H_out = np.matrix([[0.0],[0.0],[0.0]]) #(3,1)      # after Activation fuction
        H_out_d = np.matrix([[0.0],[0.0],[0.0]]) #(3,1)    # differential
        for i in range(len(H)): #len(H)=3
            output = (np.exp(H[i]) - np.exp(-H[i])) / (np.exp(H[i]) + np.exp(-H[i]))
            H_out[i] = output
            H_out_d[i] = 1-output**2  #if it doesn't work, seperate H_out and H_out_d into two def
        return H_out, H_out_d


    def AF_sigmoid(H):
        output = 1/(1+np.exp(-H)) 
        # H_out[i] = output
        # H_out_d[i] = output*(1-output)
        return output



    def Error (Y):
        Error = (1/2)*((GT-Y)**2)
        Error_d = -(GT-Y)
        return Error, Error_d


    def backpropagation(W1, W2, W3, X, GT): # W1:(3,2), W2:(2,3), W3:(1,2), X:(2,1), GT:1
        H1 = np.matrix([[0.0],[0.0],[0.0]]) # hidden_layer_1 (3,1)
        H2 = np.matrix([[0.0],[0.0]])       # hidden_layer_2 (2,1)
        
        #feed forward
        print("Feed forward")
        print("----------------------------------")
        H1 = np.matmul(W1, X) #(3,2)x(2,1) = (3,1)
        H1_out, H1_out_d = deeplearning.AF_tanh(H1)

        H2 = np.matmul(W2, H1_out) #(2,3)x(3,1) = (2,1)

        Y = np.matmul(W3, H2) # (1,2)x(2,1) = 1

        Error_Y, Error_Y_d = deeplearning.Error(Y)
        print('Error_Y:', Error_Y)

        Error_d_W1 = np.matrix([[0.0,0.0], [0.0,0.0], [0.0,0.0]]) #(3,2)
        Error_d_W2 = np.matrix([[0.0,0.0,0.0], [0.0,0.0,0.0]]) #(2,3)
        Error_d_W3 = np.matrix([[0.0,0.0]]) #(1,2)


        print("\n\nError_d_W1")
        print("----------------------------------")
        for i in range(len(W1)): #3
            for j in range(np.size(W1[0])): #2  len(W1[0]) -> 1 ??
                Error_d_W1[i,j] = Error_Y_d*(W3[0,0]*W2[0,i]*H1_out_d[i]*X[j] \
                                           + W3[0,1]*W2[1,i]*H1_out_d[i]*X[j] )

                # print('Error_d_W1[{},{}]:'.format(i,j), Error_d_W1[i,j])
                # print('W31:', W3[0,0],'W32:', W3[0,1])
                # print('W2[0,{}]:'.format(i), W2[0,i])
                # print('W2[1,{}]:'.format(i), W2[1,i])
                # print('H1_out_d[{}]:'.format(i), H1_out_d[i])
                # print('X[{}]:'.format(j), X[j], '\n')

        print('Error_d_w1:', Error_d_W1)



        print("\n\nError_d_W2")
        print("----------------------------------")
        for i in range(len(W2)): #2
            for j in range(np.size(W2[0])): #3  len(W2[0]) -> 1 ??
                Error_d_W2[i,j] = Error_Y_d * W3[0,i] * H1_out[j]
            # print('Error_d_W2[{},{}]:'.format(i,j), Error_d_W2[i,j])
            # print('W3[0,{}]:'.format(i), W3[0,i])
            # print('H1_out_d[{}]:'.format(j), H1_out_d[j], "\n")

        print('Error_d_w2:', Error_d_W2, "\n\n")

        print("Error_d_W3")
        print("----------------------------------")
        for i in range(np.size(W3)): #2
            Error_d_W3[0,i] = Error_Y_d * H2[i] 
        print('Error_d_w3:', Error_d_W3)

        # W value update
        W1 = W1 + Error_d_W1 * Learning_rate
        W2 = W2 + Error_d_W2 * Learning_rate
        W3 = W3 + Error_d_W3 * Learning_rate

        return W1, W2, W3, Error_Y






if __name__ == '__main__':
    deep = deeplearning

#initialize   
    #W1
    ran_w1 = np.random.rand(6)  #list
    arr_w1 = ran_w1.reshape(3,2)    #array
    W1 = np.asmatrix(arr_w1, dtype=float)   #matrix
    print("W1:",W1,'\n')
    
    #W2
    ran_w2 = np.random.rand(6)  #list
    arr_w2 = ran_w2.reshape(2,3)    #array
    W2 = np.asmatrix(arr_w2, dtype=float)   #matrix
    print("W2:",W2,'\n')
   
    #W3
    ran_w3 = np.random.rand(2)  #list
    arr_w3 = ran_w3.reshape(1,2)    #array
    W3 = np.asmatrix(arr_w3, dtype=float)   #matrix
    print("W3:",W3,'\n')
    print("w3 shape", np.shape(W3), '\n')
    
    #X
    X = np.matrix([[random.randrange(0,101)],[random.randrange(0,101)]])
    print("x:", X, '\n')

    GT = X[0] + X[1] #Ground Truth
    print("GT:", GT)

    #================================================================


    # back propagation starts
    print("\n\n backpropagation starts \n\n")
    print("===================================================")
    for i in range(100):
        W1, W2, W3, Error = deep.backpropagation(W1, W2, W3, X, GT)
        if i%1 == 0:
            print("\nno.{} W Value:".format(i+1) ,"\nW1:", W1, "\nW2:", W2, "\nW3:", W3, "\n")
            print("Error: ", Error, "\n\n" )



    #feed forward
    H1 = np.matrix([[0.0],[0.0],[0.0]]) # hidden_layer_1 (3,1)
    H2 = np.matrix([[0.0],[0.0]])       # hidden_layer_2 (2,1)
        
    H1 = np.matmul(W1, X) #(3,2)x(2,1) = (3,1)
    H1_out = deep.AF_tanh(H1)[0]

    H2 = np.matmul(W2, H1_out) #(2,3)x(3,1) = (2,1)

    Y = np.matmul(W3, H2) # (1,2)x(2,1) = 1

    print("final Y:", Y)