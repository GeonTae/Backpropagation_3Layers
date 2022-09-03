"""
x1,x2 0~100 아무숫자
back -> forward
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import itertools


global Learning_rate
Learning_rate=0.5


class deeplearning:

    def AF_tanh(H):   #activation function tanh
        H_out = np.matrix([[0.0],[0.0],[0.0]]) #(3,1)      # after Activation fuction
        H_out_d = np.matrix([[0.0],[0.0],[0.0]]) #(3,1)    # differential
        for i in range(len(H)):
            output = (np.exp(H[i]) - np.exp(-H[i])) / (np.exp(H[i]) + np.exp(-H[i]))
            H_out[i] = output
            H_out_d[i] = 1-output**2 
        return H_out, H_out_d


    def AF_sigmoid(H):
        output = 1/(1+np.exp(-H)) 
        # H_out[i] = output
        # H_out_d[i] = output*(1-output)
        return output



    def Error (Y, GT):
        Error = (1/2)*((GT-Y)**2)  #cost => minimize
        # if GT > Y:
        #     Error_d = -(GT-Y)
        # else:
        #     Error_d = (GT-Y)
        # Error_d = -(GT-Y) #here????????????????????
        Error_d = (GT-Y) # Error_d=0  <= minimum area that I need to approach
        return Error, Error_d


    def backpropagation(W1, W2, W3, X, GT): # W1:(3,2), W2:(2,3), W3:(1,2), X:(2,1), GT:1
        H1 = np.matrix([[0.0],[0.0],[0.0]]) # hidden_layer_1 (3,1)
        H2 = np.matrix([[0.0],[0.0]])       # hidden_layer_2 (2,1)
        
        #feed forward
        H1 = np.matmul(W1, X) #(3,2)x(2,1) = (3,1)
        # print("\nH1:", H1)
        H1_out, H1_out_d = deeplearning.AF_tanh(H1)
        # print("H1_out:", H1_out, "\nH1_out_d:", H1_out_d)
        H2 = np.matmul(W2, H1_out) #(2,3)x(3,1) = (2,1)
        # print("H2:", H2)
        Y = np.matmul(W3, H2) # (1,2)x(2,1) = 1
        # print("\nY:", Y)
        Error_Y, Error_Y_d = deeplearning.Error(Y, GT)
        # print('Error_Y:', Error_Y,'Error_Y_d:', Error_Y_d)

        Error_d_W1 = np.matrix([[0.0,0.0], [0.0,0.0], [0.0,0.0]]) #(3,2)
        Error_d_W2 = np.matrix([[0.0,0.0,0.0], [0.0,0.0,0.0]]) #(2,3)
        Error_d_W3 = np.matrix([[0.0,0.0]]) #(1,2)

        # print("\n\nEach Error, value check")
        # print("\nError_d_W1")
        # print("----------------------------------")
        for i in range(len(W1)): #3
            for j in range(np.size(W1[0])): #2  len(W1[0]) -> 1 ??
                Error_d_W1[i,j] = Error_Y_d*(W3[0,0]*W2[0,i]*H1_out_d[i]*X[j] \
                                           + W3[0,1]*W2[1,i]*H1_out_d[i]*X[j] )

                # print('W31:', W3[0,0],'\nW32:', W3[0,1])
                # print('W2[0,{}]:'.format(i), W2[0,i])
                # print('W2[1,{}]:'.format(i), W2[1,i])
                # print('H1_out_d[{}]:'.format(i), H1_out_d[i])
                # print('X[{}]:'.format(j), X[j], '\n')
                # print('Error_d_W1[{},{}]:'.format(i,j), Error_d_W1[i,j])
        # print('\n=> Error_d_w1:\n', Error_d_W1)



        # print("\n\nError_d_W2")
        # print("----------------------------------")
        for i in range(len(W2)): #2
            for j in range(np.size(W2[0])): #3  len(W2[0]) -> 1 ??
                Error_d_W2[i,j] = Error_Y_d * W3[0,i] * H1_out[j]
            
            # print('W3[0,{}]:'.format(i), W3[0,i])
            # print('H1_out_d[{}]:'.format(j), H1_out_d[j], "\n")
            # print('Error_d_W2[{},{}]:'.format(i,j), Error_d_W2[i,j])
        # print('Error_d_w2:', Error_d_W2)


        # print("\n\nError_d_W3")
        # print("----------------------------------")
        for i in range(np.size(W3)): #2
            Error_d_W3[0,i] = Error_Y_d * H2[i]
            
            # print('H2[{}]:'.format(i), H2[i], "\n")
            # print('Error_d_W3[{},{}]:'.format(0,i), Error_d_W3[0,i])
        # print('Error_d_w3:', Error_d_W3)

        # W value update
        W1 = W1 + Error_d_W1 * Learning_rate
        W2 = W2 + Error_d_W2 * Learning_rate
        W3 = W3 + Error_d_W3 * Learning_rate

        return W1, W2, W3, Y, Error_Y, Error_Y_d 





if __name__ == '__main__':

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
    # print("w3 shape", np.shape(W3), '\n')
    
    #X
    X = np.matrix([[np.random.rand()],[np.random.rand()]])
    # X = np.matrix([[random.randrange(0,101)],[random.randrange(0,101)]])
    print("x:", X, '\n')


    GT = X[0] + X[1] #Ground Truth

    # if X[0] > X[1]:
    #     GT = X[0] - X[1]   #Ground Truth
    # else:
    #     GT = X[1] - X[0] 

    print("GT:", GT)


    loss = []
    loss_d = []

    # back propagation starts
    print("\n\n**************backpropagation starts*********************\n")
    # print("==============================================================")
    for i in range(1000):
        # print("step {}".format(i+1))
        W1, W2, W3, Y, Error, Error_d = deeplearning.backpropagation(W1, W2, W3, X, GT) # return W1, W2, W3, Y, Error_Y, Error_Y_d 
        loss.append(Error.tolist())
        loss_d.append(Error_d.tolist())
        if i%9 == 0:
        # print("\nstep {} Back propagation is done\n".format(i+1))   
            print("step {}".format(i+1))
            print("\nY:", Y)
            print("Error:", Error)
        # print("Updated Value" ,"\nW1:", W1, "\nW2:", W2, "\nW3:", W3)
            print("---------------------------------------------------------------\n")


    #feed forward
    print("\n\n\n\n\nLast feed forward")
    print("============================================")
    H1 = np.matrix([[0.0],[0.0],[0.0]]) # hidden_layer_1 (3,1)
    H2 = np.matrix([[0.0],[0.0]])       # hidden_layer_2 (2,1)
    
    H1 = np.matmul(W1, X) #(3,2)x(2,1) = (3,1)
    print("final H1: ", H1)
    H1_out = deeplearning.AF_tanh(H1)[0]
    print("final H1_out: ", H1_out)
    H2 = np.matmul(W2, H1_out) #(2,3)x(3,1) = (2,1)
    print("final H2: ", H2)

    print("\nfinal step {} Error: ".format(i+1), Error)

    Y = np.matmul(W3, H2) # (1,2)x(2,1) = 1
    print("\nfinal Y:", Y)
    print("GroundTruth:", GT)


    # check_Error = sum(check_Error, [])
    # check_Error = sum(check_Error, [])
    # check_Error_d = sum(check_Error_d, [])
    # check_Error_d = sum(check_Error_d, [])
 
    loss = list(itertools.chain.from_iterable(loss))
    loss = list(itertools.chain.from_iterable(loss))
    loss_d = list(itertools.chain.from_iterable(loss_d))
    loss_d = list(itertools.chain.from_iterable(loss_d))

    plt.subplot(2,1,1)
    plt.plot(loss, label='Loss')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(loss_d, label='Loss\' Gradient')
    plt.legend()
    plt.tight_layout()    
    plt.show()
