from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import vae_layers as vae_layers
import vae_models as vae_models
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.models import Model
import data_processing as data_processing


def main(args):

    # creating the model
    def create_model():
        input=Input((args.data_dim,))
        struc=vae_models.VAE('vae',args.data_dim,args.n_dim,args.n_width,args.batch_size)
        output=struc(input)
        model = Model(input, output)
        return model

    model=create_model()

    #call the model to compute the vae loss function
    def vae_loss(z):
        y,s,a= model(z)
        return y,s,a


    #these two lines of codes contain information about the optimization technique that we are going to use
    loss_metric = tf.keras.metrics.Mean()
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)
    
    #defining a single training step that includes gradient finding and parameter update
    @tf.function
    def vae_train_step(inputs, vars):
        yt = inputs
        with tf.GradientTape() as tape:
            cost, KL_diverge, log_pd= vae_loss(yt)
            
        grads = tape.gradient(cost, vars)
        optimizer.apply_gradients(zip(grads, vars))
        
        return vars, KL_diverge, log_pd
    

    #loading the data and preparing the training dataset
    load=np.loadtxt('./temporary/input_data.dat').astype(np.float32)
    y=tf.reshape(load, [args.data_size,args.data_dim])
    y=y[:args.training_data_size, :]
    frame=data_processing.dataflow(y,args.training_data_size,args.batch_size)
      
    #the second line here is a vector that records the parts of the cost function after every epoch
    number_of_batches=int(args.training_data_size/args.batch_size)
    vae_cost_function_vector=np.zeros((args.n_epoch,2), dtype=float)


    #training starts here. the first for loop counts for the number of epochs
    for j in range(args.n_epoch):
        print('EPOCH NUMBER:', j+1)
        #for every epoch we get a suffled dataset
        training_dataset=frame.get_shuffled_batched_dataset()
        
        #This second for loop counts for the number of iterations (i.e. number of batches) for a given epoch
        for step, train_batch in enumerate(training_dataset):
            current_vars = model.trainable_weights
            updated_vars,KL_diver, log_PD= vae_train_step(train_batch, current_vars)
        vae_cost_function_vector[j,:]=[KL_diver.numpy(), log_PD.numpy()]
        print('updated_ELBO',-KL_diver.numpy()+log_PD.numpy())
        
    vae_cost_function_vector=tf.reshape(vae_cost_function_vector, [args.n_epoch,2])

    #saving the trained values of the generator parameters
    EW_1,Eb_1,EW_2,Eb_2,EW_3,Eb_3,DW_4,Db_4,DW_5,Db_5,DW_6,Db_6 = updated_vars
   
    np.savetxt('./temporary/Encoder_W1.dat'.format(), EW_1)
    np.savetxt('./temporary/Encoder_b1.dat'.format(), Eb_1)
    np.savetxt('./temporary/Encoder_W2.dat'.format(), EW_2)
    np.savetxt('./temporary/Encoder_b2.dat'.format(), Eb_2)
    np.savetxt('./temporary/Encoder_W3.dat'.format(), EW_3)
    np.savetxt('./temporary/Encoder_b3.dat'.format(), Eb_3)
    np.savetxt('./temporary/Decoder_W4.dat'.format(), DW_4)
    np.savetxt('./temporary/Decoder_b4.dat'.format(), Db_4)
    np.savetxt('./temporary/Decoder_W5.dat'.format(), DW_5)
    np.savetxt('./temporary/Decoder_b5.dat'.format(), Db_5)
    np.savetxt('./temporary/Decoder_W6.dat'.format(), DW_6)
    np.savetxt('./temporary/Decoder_b6.dat'.format(), Db_6)

    #saving the cost function vector
    #each element in the vector respresents the value of the updated cost function at the end of the corresponding epoch
    np.savetxt('./temporary/cost_function_vae.dat'.format(), vae_cost_function_vector)





if __name__ == '__main__':
    from configargparse import ArgParser
    p = ArgParser()
    
    # save parameters
    p.add("--n_dim", type=int, default=2, help='The number of random dimensions for latent space.')
    p.add("--batch_size", type=int, default=5, help='The number of data points in a batch.')
    p.add("--data_size", type=int, default=100, help='The number of data points in the dataset')
    p.add("--training_data_size", type=int, default=80, help='The number of data points to be used for training.')
    p.add("--n_epoch", type=int, default=2, help='The number of training epochs for the program')
    p.add('--n_width', type=int, default=128, help='The number of neurons for the hidden layers in VAE encoder.')
    p.add('--data_dim', type=int, default=10, help='The number of dimensions in the data')

    #optimization hyperparams:
    p.add("--lr", type=float, default=0.001, help='Base learning rate.')
    
    args = p.parse_args()
    main(args)
