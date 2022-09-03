# Backpropagation_3Layers[Uploading Derivation_Backpropagation_2022-05-11.docx…]()
![image](https://user-images.githubusercontent.com/49116137/188257039-b0eca5a0-819c-4f86-920e-aef8152cad55.png)


Summary of Equation for Backpropagation

	Consider W1,1,1
∂Error/(∂w_1,1,1 )   =   ∂Error/∂y  ∂y/(∂w_1,1,1 )  
=  ∂Error/∂y  ∂(H_2,1 w_3,1+ H_2,2 w_3,2+b_3 )/(∂w_1,1,1 )
=  ∂Error/∂y ((〖∂H〗_2,1 w_3,1)/(∂w_1,1,1 )+(〖∂H〗_2,2 w_3,2)/(∂w_1,1,1 )+(∂b_3)/(∂w_1,1,1 ))
=  ∂Error/∂y (w_3,1  〖∂H〗_2,1/(∂w_1,1,1 )+w_3,2  〖∂H〗_2,2/(∂w_1,1,1 ))  
=∂Error/∂y (w_3,1  〖∂H〗_2,1/(∂outH_1,1 )  (∂outH_1,1)/(∂H_1,1 )  (∂H_1,1)/〖∂w〗_1,1,1 +w_3,2  〖∂H〗_2,2/(∂outH_1,1 )  (∂outH_1,1)/(∂H_1,1 )  (∂H_1,1)/〖∂w〗_1,1,1 )

∂Error/(∂w_1,1,1 )  =  ∂Error/∂y (w_3,1  〖∂H〗_2,1/(∂outH_1,1 )  (∂outH_1,1)/(∂H_1,1 )  (∂H_1,1)/〖∂w〗_1,1,1 +w_3,2  〖∂H〗_2,2/(∂outH_1,1 )  (∂outH_1,1)/(∂H_1,1 )  (∂H_1,1)/〖∂w〗_1,1,1 )  
