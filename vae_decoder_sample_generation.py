import tensorflow as tf
import numpy as np

##first of all we load the trained decoder parameters
DW_4=np.loadtxt('./temporary/Decoder_W4.dat').astype(np.float32)
#DW_4=tf.reshape(DW_4,[1,1024])
Db_4=np.loadtxt('./temporary/Decoder_b4.dat').astype(np.float32)
DW_5=np.loadtxt('./temporary/Decoder_W5.dat').astype(np.float32)
Db_5=np.loadtxt('./temporary/Decoder_b5.dat').astype(np.float32)
DW_6=np.loadtxt('./temporary/Decoder_W6.dat').astype(np.float32)
Db_6=np.loadtxt('./temporary/Decoder_b6.dat').astype(np.float32)


#this part of the code does the decoder operation on a given latent sample
def decoder_operation(z,data_dim):
    hd_1=tf.nn.relu(tf.matmul(z, DW_4)+Db_4)
    hd_2=tf.nn.relu(tf.matmul(hd_1, DW_5)+Db_5)
    Decoder_output=tf.matmul(hd_2, DW_6)+Db_6
        
    mean_vector= Decoder_output[:,:data_dim]
    mean_vector=tf.reshape(mean_vector, [data_dim,])
    mean_de=mean_vector.numpy()
        
    std_de = tf.exp(Decoder_output[:, data_dim:])
    std_de=tf.reshape(std_de,[data_dim,])
    sigma_de=np.square(np.diag(std_de))

    return mean_de, sigma_de


# #this function generates samples in the input space using just the trained decoder with two dimensional Gaussian as input
# # #n_samples refers to the number of samples we want to generate
# # #data_dim stands for the dimension of each sample
## n_dim refers to the dimension of the latent space

def decoder_samples(n_samples,data_dim, n_dim):
    x_hat_samples=np.zeros((n_samples,data_dim), dtype=float)
 
    for i in range(n_samples):
        if (i+1) % 10000 == 0:
            print('number of samples generated so far:', i+1)
        z=tf.random.normal((1,n_dim))
        mean_de,sigma_de=decoder_operation(z,data_dim)
        x_hat_samples[i,:]=np.random.multivariate_normal(mean_de, sigma_de)

    x_hat_samples=tf.reshape(x_hat_samples,[n_samples,data_dim])
        
    return x_hat_samples


##here we execute our code for given set of inputs and save the generated samples in the desired directory
deco_data=decoder_samples(5, 10, 2)
np.savetxt('./temporary/decoder_data.dat'.format(), deco_data)

