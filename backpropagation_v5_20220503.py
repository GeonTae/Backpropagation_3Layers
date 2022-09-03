"""
x1,x2 0~100 any numbers
back -> forward
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain


global Learning_rate
Learning_rate=0.05



class deeplearning:

    def AF_tanh(H,X):   #activation function tanh
        H_out = np.matrix(np.zeros((3,len(X)))) #(3,the number of x datasets)      # after Activation fuction
        H_out_d = np.matrix(np.zeros((3,len(X)))) #(3,the number of x datasets)    # differential
        for i in range(len(H)):
            for j in range(len(X)):    
                output = (np.exp(H[i,j]) - np.exp(-H[i,j])) / (np.exp(H[i,j]) + np.exp(-H[i,j]))
                H_out[i,j] = output
                H_out_d[i,j] = 1-output**2 
        return H_out, H_out_d

    def test_AF_tanh(H):   #activation function tanh
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



    def Error (Y,GT):

        # if GT > Y:
        #     Error_d = -(GT-Y)
        # else:
        #     Error_d = (GT-Y)

        Error = [(1/2)*((GT[i]-Y[0,i])**2) for i in range(np.size(Y))]  
        Error_d = [(GT[i]-Y[0,i]) for i in range(np.size(Y))] # Error_d=0  <= minimum area that I need to approach

        return Error, Error_d


    def backpropagation(W1, W2, W3, X, GT): # W1:(3,2), W2:(2,3), W3:(1,2), X:(the number of x data, 2), GT: the number of x data
        H1 = np.matrix(np.zeros((3,len(X))))       # hidden_layer_1 (3,1000) if the number of x data = 1000
        H2 = np.matrix(np.zeros((2,len(X))))       # hidden_layer_2 (2,1000)
        
        #feed forward
        H1 = np.matmul(W1, np.transpose(X)) #(3,2)x(2,1000) = (3,1000)
        # print("\nH1:", H1)
        H1_out, H1_out_d = deeplearning.AF_tanh(H1,X) # X for len(X)
        # print("H1_out:", H1_out, "\nH1_out_d:", H1_out_d)
        H2 = np.matmul(W2, H1_out) #(2,3)x(3,1000) = (2,1000)
        # print("H2:", H2)
        Y = np.matmul(W3, H2) # (1,2)x(2,1000) = (1,1000)
        # print("\nY:", Y)
        Error_Y, Error_Y_d = deeplearning.Error(Y,GT)
        # print('Error_Y:', Error_Y,'Error_Y_d:', Error_Y_d)

        Error_d_W1 = np.zeros((len(X),3,2)) #(3,2) x 1000
        Error_d_W2 = np.zeros((len(X),2,3)) #(2,3) x 1000
        Error_d_W3 = np.zeros((len(X),1,2)) #(1,2) x 1000

        # Error_d_W1
        for k in range(len(X)): 
            for i in range(len(W1)): #3
                for j in range(np.size(W1[0])): #2  len(W1[0]) -> 1 ??
                    Error_d_W1[k,i,j] = Error_Y_d[k]*(W3[0,0]*W2[0,i]*H1_out_d[i,k]*X[k,j] \
                                           + W3[0,1]*W2[1,i]*H1_out_d[i,k]*X[k,j] )
                # print('W31:', W3[0,0],'\nW32:', W3[0,1])
                # print('W2[0,{}]:'.format(i), W2[0,i])
                # print('W2[1,{}]:'.format(i), W2[1,i])
                # print('H1_out_d[{}]:'.format(i), H1_out_d[i,k])
                # print('X[{}]:'.format(j), X[j], '\n')
                # print('Error_d_W1[{},{}]:'.format(i,j), Error_d_W1[k,i,j])
        # print('\n=> Error_d_w1:\n', Error_d_W1)



        # Error_d_W2
        for k in range(len(X)):
            for i in range(len(W2)): #2
                for j in range(np.size(W2[0])): #3  len(W2[0]) -> 1 ??
                    Error_d_W2[k,i,j] = Error_Y_d[k] * W3[0,i] * H1_out[j,k]
            
            # print('W3[0,{}]:'.format(i), W3[0,i])
            # print('H1_out_d[{}]:'.format(j), H1_out_d[j,k], "\n")
            # print('Error_d_W2[{},{}]:'.format(i,j), Error_d_W2[i,j])
        # print('Error_d_w2:', Error_d_W2)


        # Error_d_W3
        for k in range(len(X)):
            for i in range(np.size(W3)): #2
                Error_d_W3[k,0,i] = Error_Y_d[k] * H2[i,k]
            
            # print('H2[{}]:'.format(i), H2[i], "\n")
            # print('Error_d_W3[{},{}]:'.format(0,i), Error_d_W3[0,i])
        # print('Error_d_w3:', Error_d_W3)

        # W value update
        W1 = W1 + Error_d_W1.mean(axis=0) * Learning_rate  
        W2 = W2 + Error_d_W2.mean(axis=0) * Learning_rate
        W3 = W3 + Error_d_W3.mean(axis=0) * Learning_rate

        return W1, W2, W3, Y, Error_Y, Error_Y_d 





if __name__ == '__main__':

#initialize
    print("Initial Value")
    #W1
    ran_w1 = np.random.rand(6)  #list
    arr_w1 = ran_w1.reshape(3,2)    #array
    W1 = np.asmatrix(arr_w1, dtype=float)   #matrix
    print("W1:",W1[0])
    
    #W2
    ran_w2 = np.random.rand(6)  #list
    arr_w2 = ran_w2.reshape(2,3)    #array
    W2 = np.asmatrix(arr_w2, dtype=float)   #matrix
    print("W2:",W2[0])
   
    #W3
    ran_w3 = np.random.rand(2)  #list
    arr_w3 = ran_w3.reshape(1,2)    #array
    W3 = np.asmatrix(arr_w3, dtype=float)   #matrix
    print("W3:",W3[0])
    # print("w3 shape", np.shape(W3), '\n')
    
    #X 
    ran_X = np.random.rand(2000)   # => the number of data = 1000
    arr_X = ran_X.reshape(1000,2)  
    train_X = np.asmatrix(arr_X, dtype=float)  
    print("\ntrain_X:", train_X[0], '\n')

    # train_GT = [train_X[i,0]+ train_X[i,1] for i in range(len(train_X))]
    # print("GT:", train_GT[0])

    train_GT = np.matrix(np.zeros((len(train_X),1))) 
    for i in range(len(train_X)):
        train_GT[i,0] = train_X[i,0]+ train_X[i,1] 

    #for checking loss curve plot
    loss = []
    loss_d = []

    #epoch batch 
    epoch = 2000 # train 2000 times
    batch_size = 100 # The number of data used for one iteration
    iteration = int(len(train_X)/batch_size)  #10

    # back propagation starts
    print("\n\n**************backpropagation starts*********************\n")
    # print("==============================================================")
    for i in range(epoch):

        for j in range (iteration):
            X = train_X[batch_size*j: batch_size*(j+1)]
            GT = train_GT[batch_size*j: batch_size*(j+1)]
            W1, W2, W3, Y, Error, Error_d = deeplearning.backpropagation(W1, W2, W3, X, GT) # return W1, W2, W3, Y, Error_Y, Error_Y_d 

            if j == 0: # for checking loss curve in graph
                loss.append(np.array(Error[0]))
                loss_d.append(np.array(Error_d[0]))
                Y0 = Y[0] #for print
                

        if i%100 == 0:
            # print("\nstep {} Back propagation is done\n".format(i+1))   
            print("step {}".format(i+1))
            print("\nGT:", train_GT[0])
            print("Y:", Y0[0,0])
            print("Error:", loss[i])
            # print("Updated Value" ,"\nW1:", W1, "\nW2:", W2, "\nW3:", W3)
            print("---------------------------------------------------------------\n")



    # Test
    ran_X = np.random.rand(20)   # => the number of data = 1000
    arr_X = ran_X.reshape(10,2)  
    test_X = np.asmatrix(arr_X, dtype=float)  
    print("x:", test_X)

    test_GT = np.matrix(np.zeros((len(test_X),1))) 
    for i in range(len(test_X)):
        test_GT[i,0] = test_X[i,0]+ test_X[i,1]

    H1 = np.matmul(W1, np.transpose(test_X)) #(3,2)x(2,10) = (3,10)
    H1_out = deeplearning.AF_tanh(H1,test_X)[0] # X for len(X)
    H2 = np.matmul(W2, H1_out) #(2,3)x(3,10) = (2,10)
    Y = np.matmul(W3, H2) # (1,2)x(2,10) = (1,10)
    

    test_Error = deeplearning.Error(Y,test_GT)[0]
    print("\nfinal Y:\n", np.transpose(Y)) 
    print("\nGroundTruth:\n", test_GT)
    print("\nError", sum(test_Error)/len(test_Error), "\n\n\n")



    loss = list(chain(*loss))   # 3d -> 2d
    loss_d = list(chain(*loss_d)) 


    #plot
    plt.subplot(2,1,1)
    plt.plot(loss, label='Loss')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(loss_d, label='Loss\' Gradient')
    plt.legend()
    plt.tight_layout()    
    plt.show()

