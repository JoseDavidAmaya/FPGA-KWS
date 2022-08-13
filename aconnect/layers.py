import tensorflow as tf
import numpy as np
import math
############ This layer was made using the template provided by Keras. For more info, go to the official site.
"""
Fully Connected layer with A-Connect
INPUT ARGUMENTS:
-output_size: output_size is the number of neurons of the layer
-Wstd: Wstd standard deviation of the weights(number between 0-1. By default is 0)
-Bstd: Bstd standard deviation of the bias(number between 0-1. By default is 0)
-isBin: if the layer will binarize the weights(String yes or no. By default is no)
-pool: Number of error that you want to use
-Slice: If you want to slice the batch in order to reduce the memory usage. Only supports slicing the batch into 2,4 or 8 parts. Any other number will not slice the batch.
-d_type: Data type of the weights and other variables. Default is fp32. Please see tf.dtypes.Dtype
-weights_regularizer: Weights regularizer. Default is None
-bias_regularizer: Bias regularizer. Default is None
"""

class FC_AConnect(tf.keras.layers.Layer):
        def __init__(self,
                output_size,
                Wstd=0,
                Bstd=0,
                errDistr="normal",
                isQuant=["no","no"],
                bw=[1,1],
                pool=None,
                Slice=1,
                d_type=tf.dtypes.float32,
                weights_regularizer=None,
                bias_regularizer=None,
                **kwargs): #__init__ method is the first method used for an object in python to initialize the ...

                super(FC_AConnect, self).__init__()                                                             #...object attributes
                self.output_size = output_size                                                                  #output_size is the number of neurons of the layer
                self.Wstd = Wstd                                                                                                #Wstd standard deviation of the weights(number between 0-1. By default is 0)
                self.Bstd = Bstd                                                                                                #Bstd standard deviation of the bias(number between 0-1. By default is 0)
                self.errDistr = errDistr                                         #Distribution followed by the error matrices
                self.isQuant = isQuant                                           #if the layer will binarize the weights, bias or both (list [weights_quat (yes or no) , bias_quant (yes or no)]. By default is ["no","no"])
                self.bw = bw                                                     #Number of bits of weights and bias quantization (List [bw_weights, bw_bias]. By default is [1,1])
                self.pool = pool                                                #Number of error that you want to use
                self.Slice = Slice                                              #If you want to slice the batch in order to reduce the memory usage
                self.d_type = d_type                                            #Data type of the weights and other variables. Default is fp32. Please see tf.dtypes.Dtype
                self.weights_regularizer = tf.keras.regularizers.get(weights_regularizer)                  #Weights regularizer. Default is None
                self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)                        #Bias regularizer. Default is None
                self.validate_init()
        def build(self,input_shape):                                                             #This method is used for initialize the layer variables that depend on input_shape
                                                                                                    #input_shape is automatically computed by tensorflow
                self.W = self.add_weight("W",
                                                                                shape = [int(input_shape[-1]),self.output_size], #Weights matrix
                                                                                initializer = "glorot_uniform",
                                        dtype = self.d_type,
                                        regularizer = self.weights_regularizer,
                                                                                trainable=True)

                self.bias = self.add_weight("bias",
                                                                                shape = [self.output_size,],                                    #Bias vector
                                                                                initializer = "zeros",
                                        dtype = self.d_type,
                                        regularizer = self.bias_regularizer,
                                                                                trainable=True)
                if(self.Wstd != 0 or self.Bstd != 0): #If the layer will take into account the standard deviation of the weights or the std of the bias or both
                        if(self.Bstd != 0):
                                self.infBerr = Merr_distr([self.output_size],self.Bstd,self.d_type,self.errDistr)
                                self.infBerr = self.infBerr.numpy()  #It is necessary to convert the tensor to a numpy array, because tensors are constant and therefore cannot be changed
                                                                                                         #This was necessary to change the error matrix/array when Monte Carlo was running.


                        else:
                                self.Berr = tf.constant(1,dtype=self.d_type)
                        if(self.Wstd != 0):
                                self.infWerr = Merr_distr([int(input_shape[-1]),self.output_size],self.Wstd,self.d_type,self.errDistr)
                                self.infWerr = self.infWerr.numpy()


                        else:
                                self.Werr = tf.constant(1,dtype=self.d_type)
                else:
                        self.Werr = tf.constant(1,dtype=self.d_type) #We need to define the number 1 as a float32.
                        self.Berr = tf.constant(1,dtype=self.d_type)
                super(FC_AConnect, self).build(input_shape)

        def call(self, X, training=None): #With call we can define all the operations that the layer do in the forward propagation.
                self.X = tf.cast(X, dtype=self.d_type)
                row = tf.shape(self.X)[-1]
                self.batch_size = tf.shape(self.X)[0] #Numpy arrays and tensors have the number of array/tensor in the first dimension.
                                                                                          #i.e. a tensor with this shape [1000,784,128] are 1000 matrix of [784,128].
                                                                                          #Then the batch_size of the input data also is the first dimension.

                #This code will train the network. For inference, please go to the else part
                if(training):
                        if(self.Wstd != 0 or self.Bstd != 0):
                                if(self.isQuant==["yes","yes"]):
                                        weights = self.LWQuant(self.W)                     #Quantize the weights and multiply them element wise with Werr mask
                                        bias = self.LBQuant(self.bias)                     #Quantize the bias and multiply them element wise with Werr mask
                                elif(self.isQuant==["yes","no"]):
                                        weights = self.LWQuant(self.W)
                                        bias = self.bias
                                elif(self.isQuant==["no","yes"]):
                                        weights = self.W
                                        bias = self.LBQuant(self.bias)
                                else:
                                    weights = self.W
                                    bias = self.bias
                                if(self.pool is None):
                                    if(self.Slice == 2): #Slice the batch into 2 minibatches of size batch/2
                                        miniBatch = tf.cast(self.batch_size/2,dtype=tf.int32)
                                        Z1 = self.slice_batch(weights,miniBatch,0,row) #Takes a portion from 0:minibatch
                                        Z2 = self.slice_batch(weights,miniBatch,1,row) #Takes a portion from minibatch:2*minibatch
                                        Z = tf.concat([Z1,Z2],axis=0)
                                    elif(self.Slice == 4):
                                        miniBatch = tf.cast(self.batch_size/4,dtype=tf.int32) #Slice the batch into 4 minibatches of size batch/4
                                        Z = self.slice_batch(weights,miniBatch,0,row) #Takes a portion from 0:minibatch
                                        for i in range(3):
                                            Z1 = self.slice_batch(weights,miniBatch,i+1,row) #Takes a portion from minibatch:2*minibatch
                                            Z = tf.concat([Z,Z1],axis=0)
                                    elif(self.Slice == 8):
                                        miniBatch = tf.cast(self.batch_size/8,dtype=tf.int32) #Slice the batch into 8 minibatches of size batch/8
                                        Z = self.slice_batch(miniBatch,0,row) #Takes a portion from 0:minibatch
                                        for i in range(7):
                                            Z1 = self.slice_batch(miniBatch,i+1,row) #Takes a portion from minibatch:2*minibatch
                                            Z = tf.concat([Z,Z1],axis=0)
                                    else:
                                        if(self.Wstd !=0):
                                            #Werr = tf.gather(self.Werr,[loc_id])               #Finally, this line will take only N matrices from the "Pool" of error matrices. Where N is the batch size.
                                            Werr = Merr_distr([self.batch_size,tf.cast(row,tf.int32),self.output_size],self.Wstd,self.d_type,self.errDistr)
                                                                        #That means, with a weights shape of [784,128] and a batch size of 256. Werr should be a tensor with shape
                                                                        #[256,784,128], but gather return us a tensor with shape [1,256,784,128], so we remove that 1 with squeeze.
                                        else:
                                            Werr = self.Werr
                                        memW = tf.multiply(weights,Werr)                                        #Finally we multiply element-wise the error matrix with the weights.

                                        if(self.Bstd !=0):                                                              #For the bias is exactly the same situation
                                            #Berr = tf.gather(self.Berr, [loc_id])
                                            Berr = Merr_distr([self.batch_size,self.output_size],self.Bstd,self.d_type,self.errDistr)
                                        else:
                                            Berr = self.Berr
                                        membias = tf.multiply(Berr,self.bias)

                                        Xaux = tf.reshape(self.X, [self.batch_size,1,tf.shape(self.X)[-1]]) #We need this reshape, beacuse the input data is a column vector with
                                                                                            # 2 dimension, e.g. in the first layer using MNIST we will have a vector with
                                                                                            #shape [batchsize,784], and we need to do a matrix multiplication.
                                                                                            #Which means the last dimension of the first matrix and the first dimension of the
                                                                                            #second matrix must be equal. In this case, with a FC layer with 128 neurons we will have this dimensions
                                                                                            #[batchsize,784] for the input and for the weights mask [batchsize,784,128]
                                                                                            #And the function tf.matmul will not recognize the first dimension of X as the batchsize, so the multiplication will return a wrong result.
                                                                                            #Thats why we add an extra dimension, and transpose the vector. At the end we will have a vector with shape [batchsize,1,784].
                                                                                            #And the multiplication result will be correct.

                                        Z = tf.matmul(Xaux, memW)       #Matrix multiplication between input and mask. With output shape [batchsize,1,128]
                                        Z = tf.reshape(Z,[self.batch_size,tf.shape(Z)[-1]]) #We need to reshape again because we are working with column vectors. The output shape must be[batchsize,128]
                                        Z = tf.add(Z,membias) #FInally, we add the bias error mask
                                        #Z = self.forward(self.W,self.bias,self.Xaux)
                                else: #if we have pool attribute the layer will train with a pool of error matrices created during the forward propagation.
                                    if(self.Wstd !=0):
                                        Werr = Merr_distr([self.pool,tf.cast(row,tf.int32),self.output_size],self.Wstd,self.d_type,self.errDistr)
                                                                        #That means, with a weights shape of [784,128] and a batch size of 256. Werr should be a tensor with shape
                                                                        #[256,784,128], but gather return us a tensor with shape [1,256,784,128], so we remove that 1 with squeeze.
                                    else:
                                        Werr = self.Werr

                                    if(self.Bstd !=0):                                                          #For the bias is exactly the same situation
                                        Berr = Merr_distr([self.pool,self.output_size],self.Bstd,self.d_type,self.errDistr)
                                    else:
                                        Berr = self.Berr

                                    newBatch = tf.cast(tf.floor(tf.cast(self.batch_size/self.pool,dtype=tf.float16)),dtype=tf.int32)
                                    Z = tf.matmul(self.X[0:newBatch], weights*Werr[0])  #Matrix multiplication between input and mask. With output shape [batchsize,1,128]
                                    Z = tf.reshape(Z,[newBatch,tf.shape(Z)[-1]]) #We need to reshape again because we are working with column vectors. The output shape must be[batchsize,128]
                                    Z = tf.add(Z,self.bias*Berr[0]) #FInally, we add the bias error mask
                                    for i in range(self.pool-1):
                                        Z1 = tf.matmul(self.X[(i+1)*newBatch:(i+2)*newBatch], weights*Werr[i+1])
                                        Z1 = tf.add(Z1,self.bias*Berr[i+1])
                                        Z = tf.concat([Z,Z1],axis=0)

                        else:
                                if(self.isQuant==['yes','yes']):
                                        self.memW = self.LWQuant(self.W)*self.Werr
                                        self.membias = self.LBQuant(self.bias)*self.Berr
                                elif(self.isQuant==['yes','no']):
                                        self.memW = self.LWQuant(self.W)*self.Werr
                                        self.membias = self.bias*self.Berr
                                elif(self.isQuant==['no','yes']):
                                        self.memW = self.W*self.Werr
                                        self.membias = self.LBQuant(self.bias)*self.Berr
                                else:
                                        self.memW = self.W*self.Werr
                                        self.membias = self.bias*self.Berr
                                Z = tf.add(tf.matmul(self.X,self.memW),self.membias) #Custom FC layer operation when we don't have Wstd or Bstd.

                else:
                    #This part of the code will be executed during the inference
                        if(self.Wstd != 0 or self.Bstd !=0):
                                if(self.Wstd !=0):
                                        Werr = self.infWerr
                                else:
                                        Werr = self.Werr
                                if(self.Bstd != 0):
                                        Berr = self.infBerr
                                else:
                                        Berr = self.Berr
                        else:
                                Werr = self.Werr
                                Berr = self.Berr
                        if(self.isQuant==['yes','yes']):
                                weights =self.LWQuant(self.W)*Werr
                                bias =self.LBQuant(self.bias)*Berr
                        elif(self.isQuant==['yes','no']):
                                weights =self.LWQuant(self.W)*Werr
                                bias =self.bias*Berr
                        elif(self.isQuant==['no','yes']):
                                weights =self.W*Werr
                                bias =self.LBQuant(self.bias)*Berr
                        else:
                                weights = self.W*Werr
                                bias = self.bias*Berr
                        Z = tf.add(tf.matmul(self.X, weights), bias)

                return Z
        def slice_batch(self,miniBatch,N,row):
                if(self.Wstd != 0):
                        Werr = Merr_distr([miniBatch,tf.cast(row,tf.int32),self.output_size],self.Wstd,self.d_type,self.errDistr)
                else:
                        Werr = self.Werr
                if(self.Bstd != 0):
                        Berr = Merr_distr([miniBatch,self.output_size],self.Bstd,self.d_type,self.errDistr)
                else:
                        Berr = self.Berr
                memW = weights*Werr
                membias = self.bias*Berr
                Xaux = tf.reshape(self.X[N*miniBatch:(N+1)*miniBatch], [miniBatch,1,row])

                Z = tf.matmul(Xaux, memW)       #Matrix multiplication between input and mask. With output shape [batchsize,1,128]
                Z = tf.reshape(Z,[miniBatch,tf.shape(Z)[-1]]) #We need to reshape again because we are working with column vectors. The output shape must be[batchsize,128]
                Z = tf.add(Z,membias) #FInally, we add the bias error mask
                #Z = self.forward(self.W,self.bias,Xaux)
                #tf.print('Z dims: ',tf.shape(Z))
                return Z

        #THis is only for saving purposes. Does not affect the layer performance.
        def get_config(self):
                config = super(FC_AConnect, self).get_config()
                config.update({
                        'output_size': self.output_size,
                        'Wstd': self.Wstd,
                        'Bstd': self.Bstd,
                        'isBin': self.isQuant,
                        'bw': self.bw,
            'pool' : self.pool,
            'Slice': self.Slice,
            'd_type': self.d_type,
            'weights_regularizer': self.weights_regularizer,
            'bias_regularizer' : self.bias_regularizer})
                return config

        def validate_init(self):
                if self.output_size <= 0:
                    raise ValueError('Unable to build a Dense layer with 0 or negative dimension. ' 'Output size: %d' %(self.output_size,))
                if self.Wstd > 1 or self.Wstd < 0:
                    raise ValueError('Wstd must be a number between 0 and 1. \n' 'Found %d' %(self.Wstd,))
                if self.Bstd > 1 or self.Bstd < 0:
                    raise ValueError('Bstd must be a number between 0 and 1. \n' 'Found %d' %(self.Bstd,))
                if not isinstance(self.errDistr, str):
                    raise TypeError('errDistr must be a string. Only two distributions supported: "normal", "lognormal"'
                            'Found %s' %(type(self.errDistr),))
                if not isinstance(self.isQuant, list):
                    raise TypeError('isQuant must be a list, ["yes","yes"] , ["yes","no"], ["no","yes"] or ["no","no"]. ' 'Found %s' %(type(self.isQuant),))
                if self.pool is not None and not isinstance(self.pool, int):
                    raise TypeError('pool must be a integer. ' 'Found %s' %(type(self.pool),))

        @tf.custom_gradient
        def LWQuant(self,x):      # Gradient function for weights quantization
            if (self.bw[0]==1):
                y = tf.math.sign(x)
                def grad(dy):
                        dydx = tf.divide(dy,abs(x)+1e-5)
                        return dydx
            else:
                limit = math.sqrt(6/((x.get_shape()[0])+(x.get_shape()[1])))
                y = (tf.clip_by_value(tf.floor((x/limit)*(2**(self.bw[0]-1))+1),-(2**(self.bw[0]-1)-1), 2**(self.bw[0]-1)) -0.5)*(2/(2**self.bw[0]-1))*limit
                def grad(dy):
                        dydx = tf.multiply(dy,tf.divide((tf.clip_by_value(tf.floor((x/limit)*(2**(self.bw[0]-1))	+1),-(2**(self.bw[0]-1)-1),2**(self.bw[0]-1)) -0.5)*(2/(2**self.bw[0]-1))*limit,x+1e-5))
                        return dydx
            return y, grad

        @tf.custom_gradient
        def LBQuant(self,x):      # Gradient function for bias quantization
            if (self.bw[1]==1):
                y = tf.math.sign(x)
                def grad(dy):
                        dydx = tf.divide(dy,abs(x)+1e-5)
                        return dydx
            else:
                limit = (2**self.bw[1])/2 #bias quantization limits
                y = (tf.clip_by_value(tf.floor((x/limit)*(2**(self.bw[1]-1))+1),-(2**(self.bw[1]-1)-1), 2**(self.bw[1]-1)) -0.5)*(2/(2**self.bw[1]-1))*limit
                def grad(dy):
                        dydx = tf.multiply(dy,tf.divide((tf.clip_by_value(tf.floor((x/limit)*(2**(self.bw[1]-1))	+1),-(2**(self.bw[1]-1)-1),2**(self.bw[1]-1)) -0.5)*(2/(2**self.bw[1]-1))*limit,x+1e-5))
                        return dydx
            return y, grad

###HOW TO IMPLEMENT MANUALLY THE BACKPROPAGATION###
"""
        @tf.custom_gradient
        def forward(self,W,bias,X):
                if(self.Wstd != 0):
                        mWerr = Merr_distr([self.miniBatch,tf.cast(self.row,tf.int32),self.output_size],self.Wstd,self.d_type,self.errDistr)
                else:
                        mWerr = self.Werr
                if(self.Bstd != 0):
                        Berr = Merr_distr([self.miniBatch,self.output_size],self.Bstd,self.d_type,self.errDistr)
                else:
                        Berr = self.Berr
                if(self.isBin=='yes'):
                        weights = tf.math.sign(W)                       #Binarize the weights
                else:
                        weights = W
                weights = tf.expand_dims(weights, axis=0)
                loc_W = weights*mWerr                           #Get the weights with the error matrix included. Also takes the binarization error when isBin=yes
                bias = tf.expand_dims(bias, axis=0)
                loc_bias = bias*Berr
                Z = tf.matmul(X,loc_W)
                Z = tf.reshape(Z, [self.miniBatch,tf.shape(Z)[-1]]) #Reshape Z to column vector
                Z = tf.add(Z, loc_bias) # Add the bias error mask
                def grad(dy):
                        if(self.isBin=="yes"):
                                layerW = tf.expand_dims(W, axis=0)
                                Werr = loc_W/layerW             #If the layer is binary we use Werr as W*/layer.W as algorithm 3 describes.
                        else:
                                Werr = mWerr  #If not, Werr will be the same matrices that we multiplied before
                        dy = tf.reshape(dy, [self.miniBatch,1,tf.shape(dy)[-1]]) #Reshape dy to [batch,1,outputsize]
                        dX = tf.matmul(dy,loc_W, transpose_b=True) #Activations gradient
                        dX = tf.reshape(dX, [self.miniBatch, tf.shape(dX)[-1]])
                        dWerr = tf.matmul(X,dy,transpose_a=True) #Gradient for the error mask of weights
                        dBerr = tf.reshape(dy, [self.miniBatch,tf.shape(dy)[-1] ]) #Get the gradient of the error mask of bias with property shape
                        dW = dWerr*Werr #Get the property weights gradient
                        dW = tf.reduce_sum(dW, axis=0)
                        dB = dBerr*Berr #Get the property bias gradient
                        dB = tf.reduce_sum(dB, axis=0)
                        return dW,dB,dX
                return Z, grad """
###########################################################################################################3
"""
Convolutional layer with A-Connect
INPUT ARGUMENTS:
-filters: Number of filter that you want to use during the convolution.(Also known as output channels)
-kernel_size: List with the dimension of the filter. e.g. [3,3]. It must be less than the input data size
-Wstd and Bstd: Weights and bias standard deviation for training
-pool: Number of error matrices that you want to use.
-isBin: string yes or no, whenever you want binary weights
-strides: Number of strides (or steps) that the filter moves during the convolution
-padding: "SAME" or "VALID". If you want to keep the same size or reduce it.
-Op: 1 or 2. Which way to do the convolution you want to use. The first option is slower but has less memory cosumption and the second one is faster
but consumes a lot of memory.
-Slice: Optional parameter. Used to divide the batch into 2,4 or 8 minibatches of size batch/N.
-d_type: Type of the parameters that the layers will create. Supports fp16, fp32 and fp64
"""
class Conv_AConnect(tf.keras.layers.Layer):
        def __init__(self,
                filters,
                kernel_size,
                strides=1,
                padding="VALID",
                Wstd=0,
                Bstd=0,
                errDistr="normal",
                pool=None,
                isQuant=['no','no'],
                bw=[1,1],
                Op=1,
                Slice=1,
                d_type=tf.dtypes.float32,
                weights_regularizer=None,
                bias_regularizer=None,
                **kwargs):

                super(Conv_AConnect, self).__init__()
                self.filters = filters
                self.kernel_size = kernel_size
                self.Wstd = Wstd
                self.Bstd = Bstd
                self.errDistr = errDistr
                self.pool = pool
                self.isQuant = isQuant
                self.bw = bw
                self.strides = strides
                self.padding = padding
                self.Op = Op
                self.Slice = Slice
                self.d_type = d_type
                self.weights_regularizer = tf.keras.regularizers.get(weights_regularizer)                  #Weights regularizer. Default is None
                self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)                        #Bias regularizer. Default is None
                self.validate_init()
        def build(self,input_shape):
                self.shape = list(self.kernel_size) + list((int(input_shape[-1]),self.filters)) ### Compute the shape of the weights. Input shape could be [batchSize,H,W,Ch] RGB

                self.W = self.add_weight('kernel',
                                                                  shape = self.shape,
                                                                  initializer = "glorot_uniform",
                                  dtype=self.d_type,
                                  regularizer = self.weights_regularizer,
                                                                  trainable=True)
                self.bias = self.add_weight('bias',
                                                                        shape=(self.filters,),
                                                                        initializer = 'zeros',
                                    dtype=self.d_type,
                                    regularizer = self.bias_regularizer,
                                                                        trainable=True)
                if(self.Wstd != 0 or self.Bstd != 0): #If the layer will take into account the standard deviation of the weights or the std of the bias or both
                        if(self.Bstd != 0):
                                self.infBerr = Merr_distr([self.filters,],self.Bstd,self.d_type,self.errDistr)
                                self.infBerr = self.infBerr.numpy()  #It is necessary to convert the tensor to a numpy array, because tensors are constant and therefore cannot be changed
                                                                                                         #This was necessary to change the error matrix/array when Monte Carlo was running.

                        else:
                                self.Berr = tf.constant(1,dtype=self.d_type)
                        if(self.Wstd !=0):
                                self.infWerr = Merr_distr(self.shape,self.Wstd,self.d_type,self.errDistr)
                                self.infWerr = self.infWerr.numpy()

                        else:
                                self.Werr = tf.constant(1,dtype=self.d_type)
                else:
                        self.Werr = tf.constant(1,dtype=self.d_type) #We need to define the number 1 as a float32.
                        self.Berr = tf.constant(1,dtype=self.d_type)
                super(Conv_AConnect, self).build(input_shape)
        def call(self,X,training):
                self.X = tf.cast(X, dtype=self.d_type)
                self.batch_size = tf.shape(self.X)[0]
                if(training):
                        if(self.Wstd != 0 or self.Bstd != 0):
                                if(self.isQuant==['yes','yes']):
                                    weights = self.LWQuant(self.W)
                                    bias = self.LBQuant(self.bias)
                                elif(self.isQuant==['yes','no']):
                                    weights = self.LWQuant(self.W)
                                    bias = self.bias
                                elif(self.isQuant==['no','yes']):
                                    weights = self.W
                                    bias = self.LBQuant(self.bias)
                                else:
                                    weights=self.W
                                    bias=self.bias
                                if(self.pool is None):
                                    if(self.Op == 1):
                                        if(self.Slice == 2): #Slice the batch into 2 minibatches of size batch/2
                                            miniBatch = tf.cast(self.batch_size/2,dtype=tf.int32)
                                            Z1 = self.slice_batch(weights,miniBatch,0,self.strides) #Takes a portion from 0:minibatch
                                            Z2 = self.slice_batch(weights,miniBatch,1,self.strides) #Takes a portion from minibatch:2*minibatch
                                            Z = tf.concat([Z1,Z2],axis=0)
                                        elif(self.Slice == 4):
                                                miniBatch = tf.cast(self.batch_size/4,dtype=tf.int32) #Slice the batch into 4 minibatches of size batch/4
                                                Z = self.slice_batch(weights,miniBatch,0,self.strides) #Takes a portion from 0:minibatch
                                                for i in range(3):
                                                    Z1 = self.slice_batch(weights,miniBatch,i+1,self.strides) #Takes a portion from i*minibatch:(i+1)*minibatch
                                                    Z = tf.concat([Z,Z1],axis=0)
                                        elif(self.Slice == 8):
                                                miniBatch = tf.cast(self.batch_size/8,dtype=tf.int32) #Slice the batch into 8 minibatches of size batch/8
                                                Z = self.slice_batch(weights,miniBatch,0,self.strides) #Takes a portion from 0:minibatch
                                                for i in range(7):
                                                    Z1 = self.slice_batch(weights,miniBatch,i+1,self.strides) #Takes a portion from i*minibatch:(i+1)*minibatch
                                                    Z = tf.concat([Z,Z1],axis=0)
                                        else:
                                            if(self.Wstd != 0):
                                                Werr = Merr_distr(list((self.batch_size,))+self.shape,self.Wstd,self.d_type,self.errDistr)
                                            else:
                                                Werr = self.Werr
                                            weights = tf.expand_dims(weights,axis=0)
                                            memW = tf.multiply(weights,Werr)
                                            if(self.Bstd != 0):
                                                Berr = Merr_distr([self.batch_size,self.filters],self.Bstd,self.d_type,self.errDistr)
                                            else:
                                                    Berr = self.Berr
                                            bias = tf.expand_dims(self.bias,axis=0)
                                            membias = tf.multiply(bias,Berr)
                                            membias = tf.reshape(membias,[self.batch_size,1,1,tf.shape(membias)[-1]])
                                            Z = tf.squeeze(tf.map_fn(self.conv,(tf.expand_dims(self.X,1),memW)
                            ,fn_output_signature=self.d_type),axis=1)#tf.nn.convolution(Xaux,memW,self.strides,self.padding)
                                            Z = tf.reshape(Z, [self.batch_size, tf.shape(Z)[1],tf.shape(Z)[2],tf.shape(Z)[3]])
                                            Z = Z+membias
                                ##################################################################################################################################
                                    else:
                                        strides = [1,self.strides,self.strides,1]
                                        if(self.Slice == 2): #Slice the batch into 2 minibatches of size batch/2
                                            miniBatch = tf.cast(self.batch_size/2,dtype=tf.int32)
                                            Z1 = self.slice_batch(weights,miniBatch,0,strides) #Takes a portion from 0:minibatch
                                            Z2 = self.slice_batch(weights,miniBatch,1,strides) #Takes a portion from minibatch:2*minibatch
                                            Z = tf.concat([Z1,Z2],axis=0)
                                        elif(self.Slice == 4):
                                                miniBatch = tf.cast(self.batch_size/4,dtype=tf.int32) #Slice the batch into 4 minibatches of size batch/4
                                                Z = self.slice_batch(weights,miniBatch,0,strides) #Takes a portion from 0:minibatch
                                                for i in range(3):
                                                    Z1 = self.slice_batch(weights,miniBatch,i+1,strides) #Takes a portion from i*minibatch:(i+1)*minibatch
                                                    Z = tf.concat([Z,Z1],axis=0)
                                        elif(self.Slice == 8):
                                                miniBatch = tf.cast(self.batch_size/8,dtype=tf.int32) #Slice the batch into 8 minibatches of size batch/8
                                                Z = self.slice_batch(weights,miniBatch,0,strides) #Takes a portion from 0:minibatch
                                                for i in range(7):
                                                    Z1 = self.slice_batch(weights,miniBatch,i+1,strides) #Takes a portion from i*minibatch:(i+1)*minibatch
                                                    Z = tf.concat([Z,Z1],axis=0)
                                        else:
                                                if(self.Wstd != 0):
                                                    Werr = Merr_distr(list((self.batch_size,))+self.shape,self.Wstd,self.d_type,self.errDistr)
                                                else:
                                                    Werr = self.Werr
                                                weights = tf.expand_dims(weights,axis=0)
                                                memW = tf.multiply(weights,Werr)
                                                if(self.Bstd != 0):
                                                    Berr = Merr_distr([self.batch_size,self.filters],self.Bstd,self.d_type,self.errDistr)
                                                else:
                                                    Berr = self.Berr
                                                bias = tf.expand_dims(self.bias,axis=0)
                                                membias = tf.multiply(bias,Berr)
                                                membias = tf.reshape(membias,[self.batch_size,1,1,tf.shape(membias)[-1]])
                                                inp_r, F = reshape(self.X,memW) #Makes the reshape from [batch,H,W,ch] to [1,H,W,Ch*batch] for input. For filters from [batch,fh,fw,Ch,out_ch]  to
                                                                        #[fh,fw,ch*batch,out_ch]
                                                Z = tf.nn.depthwise_conv2d(
                                    inp_r,
                                    filter=F,
                                    strides=strides,
                                    padding=self.padding)
                                                Z = Z_reshape(Z,memW,self.X,self.padding,self.strides) #Output shape from convolution is [1,newH,newW,batch*Ch*out_ch] so it is reshaped to [newH,newW,batch,Ch,out_ch]
                                                                #Where newH and newW are the new image dimensions. This depends on the value of padding
                                                                #Padding same: newH = H  and newW = W
                                                                #Padding valid: newH = H-fh+1 and newW = W-fw+1
                                                Z = tf.transpose(Z, [2, 0, 1, 3, 4]) #Get the property output shape [batch,nH,nW,Ch,out_ch]
                                                Z = tf.reduce_sum(Z, axis=3)            #Removes the input channel dimension by adding all this elements
                                                Z = membias+Z                                   #Add the bias
                                else:
                                    if(self.Wstd != 0):
                                        Werr = Merr_distr(list((self.pool,))+self.shape,self.Wstd,self.d_type,self.errDistr)
                                    else:
                                        Werr = self.Werr

                                    if(self.Bstd != 0):

                                        Berr = Merr_distr([self.pool,self.filters],self.Bstd,self.d_type,self.errDistr)
                                    else:
                                        Berr = self.Berr

                                    newBatch = tf.cast(tf.floor(tf.cast(self.batch_size/self.pool,dtype=tf.float16)),dtype=tf.int32)
                                    Z = tf.nn.conv2d(self.X[0:newBatch], weights*Werr[0],strides=[1,self.strides,self.strides,1],padding=self.padding)
                                    Z = tf.add(Z,self.bias*Berr[0]) #FInally, we add the bias error mask
                                    for i in range(self.pool-1):
                                        Z1 = tf.nn.conv2d(self.X[(i+1)*newBatch:(i+2)*newBatch], weights*Werr[i+1],strides=[1,self.strides,self.strides,1],padding=self.padding)
                                        Z1 = tf.add(Z1,self.bias*Berr[i+1])
                                        Z = tf.concat([Z,Z1],axis=0)
                        else:
                                if(self.isQuant==['yes','yes']):
                                        weights = self.LWQuant(self.W)*self.Werr
                                        self.membias = self.LBQuant(self.bias)*self.Berr
                                elif(self.isQuant==['yes','no']):
                                        weights = self.LWQuant(self.W)*self.Werr
                                        self.membias = self.bias*self.Berr
                                elif(self.isQuant==['no','yes']):
                                        weights = self.W*self.Werr
                                        self.membias = self.LBQuant(self.bias)*self.Berr
                                else:
                                        weights=self.W*self.Werr
                                        self.membias = self.bias*self.Berr
                                Z = self.membias*self.Berr+tf.nn.conv2d(self.X,weights,self.strides,self.padding)
                else:
                        if(self.Wstd != 0 or self.Bstd !=0):
                                if(self.Wstd !=0):
                                        Werr = self.infWerr
                                else:
                                        Werr = self.Werr
                                if(self.Bstd != 0):
                                        Berr = self.infBerr
                                else:
                                        Berr = self.Berr
                        else:
                                Werr = self.Werr
                                Berr = self.Berr
                        if(self.isQuant==['yes','yes']):
                                weights=self.LWQuant(self.W)*Werr
                                bias=self.LBQuant(self.bias)*Berr
                        elif(self.isQuant==['yes','no']):
                                weights=self.LWQuant(self.W)*Werr
                                bias =self.bias*Berr
                        elif(self.isQuant==['no','yes']):
                                weights=self.W*Werr
                                bias=self.LBQuant(self.bias)*Berr
                        else:
                                weights=self.W*Werr
                                bias = self.bias*Berr
                        Z = bias+tf.nn.conv2d(self.X,weights,self.strides,self.padding)
                return Z
        def slice_batch(self,weights,miniBatch,N,strides):
                if(self.Wstd != 0):
                        Werr = Merr_distr(list((miniBatch,))+self.shape,self.Wstd,self.d_type,self.errDistr)
                else:
                        Werr = self.Werr

                weights = tf.expand_dims(weights,axis=0)
                memW = tf.multiply(weights,Werr)
                if(self.Bstd != 0):
                        Berr = Merr_distr([miniBatch,self.filters],self.Bstd,self.d_type,self.errDistr)
                else:
                        Berr = self.Berr
                bias = tf.expand_dims(self.bias,axis=0)
                membias = tf.multiply(bias,Berr)
                membias = tf.reshape(membias,[miniBatch,1,1,tf.shape(membias)[-1]])
                if self.Op == 1:
                        Z = tf.squeeze(tf.map_fn(self.conv,(tf.expand_dims(self.X[N*miniBatch:(N+1)*miniBatch],1),memW),
            fn_output_signature=self.d_type),axis=1)
                        Z = tf.reshape(Z, [miniBatch, tf.shape(Z)[1],tf.shape(Z)[2],tf.shape(Z)[3]])
                        Z = Z+membias
                else:
                    inp_r, memW = reshape(self.X[N*miniBatch:(N+1)*miniBatch],memW)
                    Z = tf.nn.depthwise_conv2d(
                                    inp_r,
                                    filter=memW,
                                    strides=strides,
                                    padding=self.padding)
                    Z = Z_reshape(Z,Werr,self.X[N*miniBatch:(N+1)*miniBatch],self.padding,self.strides)
                    Z = tf.transpose(Z, [2, 0, 1, 3, 4])
                    Z = tf.reduce_sum(Z, axis=3)
                    Z = Z+membias
                #Z = tf.cond(tf.less_equal(inp_shape[1],100),cond2(miniBatch,N,memW,Werr,membias),cond1(miniBatch,N,memW,membias))
                return Z

        def conv(self,tupla):
                x,kernel = tupla
                return tf.nn.convolution(x,kernel,strides=self.strides,padding=self.padding)
        def validate_init(self):
                if not isinstance(self.filters, int):
                    raise TypeError('filters must be an integer. ' 'Found %s' %(type(self.filters),))
                if self.Wstd > 1 or self.Wstd < 0:
                    raise ValueError('Wstd must be a number between 0 and 1. \n' 'Found %d' %(self.Wstd,))
                if self.Bstd > 1 or self.Bstd < 0:
                    raise ValueError('Bstd must be a number between 0 and 1. \n' 'Found %d' %(self.Bstd,))
                if not isinstance(self.errDistr, str):
                    raise TypeError('errDistr must be a string. Only two distributions supported: "normal", "lognormal"'
                            'Found %s' %(type(self.errDistr),))
                if not isinstance(self.isQuant, list):
                    raise TypeError('isQuant must be a list, ["yes","yes"] , ["yes","no"], ["no","yes"] or ["no","no"]. ' 'Found %s' %(type(self.isQuant),))
                if self.pool is not None and not isinstance(self.pool, int):
                    raise TypeError('pool must be a integer. ' 'Found %s' %(type(self.pool),))
        def get_config(self):
                config = super(Conv_AConnect, self).get_config()
                config.update({
                        'filters': self.filters,
                        'kernel_size': self.kernel_size,
                        'Wstd': self.Wstd,
                        'Bstd': self.Bstd,
                        'errDistr': self.errDistr,
                        'pool': self.pool,
                        'isQuant': self.isQuant,
                        'bw': self.bw,
                        'strides': self.strides,
                        'padding': self.padding,
                        'Op': self.Op,
                        'Slice': self.Slice,
                        'd_type': self.d_type})
                return config
        @tf.custom_gradient
        def LWQuant(self,x):      # Gradient function for weights quantization
            if (self.bw[0]==1):
                y = tf.math.sign(x)
                def grad(dy):
                        dydx = tf.divide(dy,abs(x)+1e-5)
                        return dydx
            else:
                limit = math.sqrt(6/((x.get_shape()[0])+(x.get_shape()[1])))
                y = (tf.clip_by_value(tf.floor((x/limit)*(2**(self.bw[0]-1))+1),-(2**(self.bw[0]-1)-1), 2**(self.bw[0]-1)) -0.5)*(2/(2**self.bw[0]-1))*limit
                def grad(dy):
                        dydx = tf.multiply(dy,tf.divide((tf.clip_by_value(tf.floor((x/limit)*(2**(self.bw[0]-1))	+1),-(2**(self.bw[0]-1)-1),2**(self.bw[0]-1)) -0.5)*(2/(2**self.bw[0]-1))*limit,x+1e-5))
                        return dydx
            return y, grad

        @tf.custom_gradient
        def LBQuant(self,x):      # Gradient function for bias quantization
            if (self.bw[1]==1):
                y = tf.math.sign(x)
                def grad(dy):
                        dydx = tf.divide(dy,abs(x)+1e-5)
                        return dydx
            else:
                limit = (2**self.bw[1])/2 #bias quantization limits
                y = (tf.clip_by_value(tf.floor((x/limit)*(2**(self.bw[1]-1))+1),-(2**(self.bw[1]-1)-1), 2**(self.bw[1]-1)) -0.5)*(2/(2**self.bw[1]-1))*limit
                def grad(dy):
                        dydx = tf.multiply(dy,tf.divide((tf.clip_by_value(tf.floor((x/limit)*(2**(self.bw[1]-1))	+1),-(2**(self.bw[1]-1)-1),2**(self.bw[1]-1)) -0.5)*(2/(2**self.bw[1]-1))*limit,x+1e-5))
                        return dydx
            return y, grad

# --- RNN LAYERS/CELLS

import tensorflow as tf

import collections
import functools
import warnings

import numpy as np
import keras
from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.saving.saved_model import layer_serialization
from keras.utils import control_flow_util
from keras.utils import generic_utils
from keras.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls

# --- FastGRNN
class LessThanEqual(keras.constraints.Constraint):
  """Constrains the weights to be less than a value.
  """

  def __init__(self, max_value=1):
    self.max_value = max_value

  def __call__(self, w):
    return tf.cast(tf.less(w, self.max_value), backend.floatx()) * (w - self.max_value) + self.max_value
    #return tf.cast(tf.less(w, self.max_value), w.dtype) * (w - self.max_value) + self.max_value # Supports tensors of different dtype than float32

  def get_config(self):
    return {'max_value': self.max_value}

class NonNeg(keras.constraints.Constraint):
  """Constrains the weights to be non-negative.
  Taken from https://github.com/keras-team/keras/blob/v2.8.0/keras/constraints.py#L121-L128
  Edited to support tensors of different dtype than float32
  """

  def __call__(self, w):
    return w * tf.cast(tf.greater_equal(w, 0.), w.dtype)
 
from keras.layers.recurrent import _caching_device
from keras.layers.recurrent import _generate_zero_filled_state_for_cell
from keras.layers.recurrent import _config_for_enable_caching_device

class FastGRNNCell(keras.layers.Layer):

  def __init__(self,
               units,
               activation='sigmoid',
               update_activation='tanh',
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               zetaInit=1.0,
               nuInit=-4.0,
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               **kwargs):
    
    if units < 0:
      raise ValueError(f'Received an invalid value for argument `units`, '
                       f'expected a positive integer, got {units}.')
    
    # By default use cached variable under v2 mode, see b/143699808.
    if tf.compat.v1.executing_eagerly_outside_functions():
      self._enable_caching_device = kwargs.pop('enable_caching_device', True)
    else:
      self._enable_caching_device = kwargs.pop('enable_caching_device', False)
    
    super(FastGRNNCell, self).__init__(**kwargs)

    self.units = units
    self.activation = activations.get(activation)
    self.update_activation = activations.get(update_activation)

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.zetaInit = zetaInit
    self.nuInit = nuInit
    self.unit_forget_bias = unit_forget_bias

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.state_size = [self.units]
    self.output_size = self.units

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    default_caching_device = _caching_device(self)
    input_dim = input_shape[-1]

    # W matrix
    self.kernel = self.add_weight(
        shape=(input_dim, self.units),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device)
    
    # U matrix
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)

    
    # zeta (Greek Z)
    self.zeta = self.add_weight(
        shape=(1,),
        name='zeta',
        initializer=keras.initializers.Constant(value=self.zetaInit),
        constraint=tf.keras.constraints.NonNeg(),
        caching_device=default_caching_device)

    # Nu (Greek v)
    self.nu = self.add_weight(
        shape=(1,),
        name='nu',
        initializer=tf.keras.initializers.Constant(value=self.nuInit),
        constraint=LessThanEqual(1),
        caching_device=default_caching_device)
    


    if self.unit_forget_bias:

      def bias_initializer(_, *args, **kwargs):
        return backend.concatenate([self.bias_initializer((self.units,), *args, **kwargs)])

    else:
      bias_initializer = self.bias_initializer

    self.bias_h = self.add_weight(
        shape=(self.units,),
        name='bias_h',
        initializer=bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        caching_device=default_caching_device)
    
    self.bias_z = self.add_weight(
        shape=(self.units,),
        name='bias_z',
        initializer=bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        caching_device=default_caching_device)
    
    self.built = True

  def call(self, inputs, states, training=None):
    h_tm1 = states[0]  # previous memory state

    # z -> z_t
    # htilde -> h_t^~

    z = backend.dot(inputs, self.kernel) + backend.dot(h_tm1, self.recurrent_kernel) # W*xt + U*h_tm1
    htilde = z # W*xt + U*h_tm1

    z = self.activation(backend.bias_add(z, self.bias_z))
    htilde = self.update_activation(backend.bias_add(htilde, self.bias_h))

    h = (tf.keras.activations.sigmoid(self.zeta)*(1 - z) + tf.keras.activations.sigmoid(self.nu)) * htilde + z * h_tm1

    return h, [h]

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'update_activation':
            activations.serialize(self.update_activation),
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'zetaInit':
            self.zetaInit,
        'nuInit':
            self.nuInit,
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint)
    }
    config.update(_config_for_enable_caching_device(self))
    base_config = super(FastGRNNCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return list(_generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype))


class FastGRNNCell_AConnect(keras.layers.Layer):

  def __init__(self,
               units,
               activation='sigmoid',
               update_activation='tanh',
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal', # Orthogonal initializer doesn't seem to support float16 values (An operation inside: tf.linalg.qr throws an error)
               bias_initializer='zeros',
               zetaInit=1.0,
               nuInit=-4.0,
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,

               # AConnect
               Wstd=0, # standard deviation of the weights 
               Bstd=0, # standard deviation of the bias
               errDistr="normal",
               pool=None,
               isBin=False, 
               d_type=tf.dtypes.float32,
               # --

               **kwargs):
    if units < 0:
      raise ValueError(f'Received an invalid value for argument `units`, '
                       f'expected a positive integer, got {units}.')
    # By default use cached variable under v2 mode, see b/143699808.
    if tf.compat.v1.executing_eagerly_outside_functions():
      self._enable_caching_device = kwargs.pop('enable_caching_device', True)
    else:
      self._enable_caching_device = kwargs.pop('enable_caching_device', False)
    
    super(FastGRNNCell_AConnect, self).__init__(**kwargs)

    self.units = units
    self.activation = activations.get(activation)
    self.update_activation = activations.get(update_activation)

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.zetaInit = zetaInit
    self.nuInit = nuInit
    self.unit_forget_bias = unit_forget_bias


    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.state_size = [self.units]
    self.output_size = self.units

    # AConnect
    self.Wstd = Wstd
    self.Bstd = Bstd
    self.errDistr = errDistr
    self.pool = pool
    self.isBin = isBin
    self.d_type = d_type
    # --

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    default_caching_device = _caching_device(self)
    input_dim = input_shape[-1]

    # W matrix
    self.kernel = self.add_weight(
        shape=(input_dim, self.units),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device,
        # AConnect
        dtype = self.d_type
        # --
        )
    
    # U matrix
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device,
        # AConnect
        dtype = self.d_type
        # --
        )

    
    # zeta (Greek Z)
    self.zeta = self.add_weight(
        shape=(1,),
        name='zeta',
        initializer=keras.initializers.Constant(value=self.zetaInit),
        constraint=tf.keras.constraints.NonNeg(),
        caching_device=default_caching_device,
        # AConnect
        dtype = self.d_type
        # --
        )

    # Nu (Greek v)
    self.nu = self.add_weight(
        shape=(1,),
        name='nu',
        initializer=tf.keras.initializers.Constant(value=self.nuInit),
        constraint=LessThanEqual(1),
        caching_device=default_caching_device,
        # AConnect
        dtype = self.d_type
        # --
        )
    
    if self.unit_forget_bias:

      def bias_initializer(_, *args, **kwargs):
        return backend.concatenate([self.bias_initializer((self.units,), *args, **kwargs)])

    else:
      bias_initializer = self.bias_initializer

    self.bias_h = self.add_weight(
        shape=(self.units,),
        name='bias_h',
        initializer=bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        caching_device=default_caching_device,
        # AConnect
        dtype = self.d_type
        # --
        )
    
    self.bias_z = self.add_weight(
        shape=(self.units,),
        name='bias_z',
        initializer=bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        caching_device=default_caching_device,
        # AConnect
        dtype = self.d_type
        # --
        )
    
    # AConnect

    if (self.Wstd != 0):
      self.infWerr = Merr_distr(self.kernel.shape, self.Wstd, self.d_type, self.errDistr)
      self.infWerr = self.infWerr.numpy()                                                   

      self.infRWerr = Merr_distr(self.recurrent_kernel.shape, self.Wstd, self.d_type, self.errDistr)
      self.infRWerr = self.infRWerr.numpy()

      self.infZerr = Merr_distr(self.zeta.shape, self.Wstd, self.d_type, self.errDistr)
      self.infZerr = self.infZerr.numpy()

      self.infNUerr = Merr_distr(self.nu.shape, self.Wstd, self.d_type, self.errDistr)
      self.infNUerr = self.infNUerr.numpy()

    else:
      self.Werr = tf.constant(1, dtype=self.d_type)
      self.RWerr = tf.constant(1, dtype=self.d_type)

      self.Zerr = tf.constant(1, dtype=self.d_type)
      self.NUerr = tf.constant(1, dtype=self.d_type)

    if(self.Bstd != 0):
      self.infBHerr = Merr_distr(self.bias_h.shape, self.Bstd, self.d_type, self.errDistr)
      self.infBHerr = self.infBHerr.numpy()

      self.infBZerr = Merr_distr(self.bias_z.shape, self.Bstd, self.d_type, self.errDistr)
      self.infBZerr = self.infBZerr.numpy()

    else:
      self.BHerr = tf.constant(1, dtype=self.d_type)
      self.BZerr = tf.constant(1, dtype=self.d_type)

    # --

    self.built = True

  @tf.custom_gradient
  def sign(self, x):
    y = tf.math.sign(x)
    def grad(dy):
      dydx = tf.divide(dy, abs(x)+1e-5)
      return dydx
    return y, grad

  def call(self, inputs, states, training=None):
    # AConnect
    inputs = tf.cast(inputs, dtype=self.d_type)
    batch_size = tf.shape(inputs)[0]
    states = [tf.cast(s, dtype=self.d_type) for s in states]
    # --

    h_tm1 = states[0]  # previous memory state

    # AConnect 
    if (training): # Training # Preparation of matrices/error matrices

      if (self.Wstd != 0 or self.Bstd != 0):

        if(self.isBin==True or self.isBin=="yes"):
          kernel = self.sign(self.kernel)
          recurrent_kernel = self.sign(self.recurrent_kernel)

          # Don't binarize 'zeta' or 'nu' parameters
          #zeta = self.sign(self.zeta)
          #nu = self.sign(self.nu)
        else:
          kernel = self.kernel
          recurrent_kernel = self.recurrent_kernel

        zeta = self.zeta
        nu = self.nu

        if self.pool is None:
          num_matrices = batch_size
          
        else:
          # When using pool, we shape the inputs and matrices in a way that allows us to use batch_dot function to calculate the output, preventing the use of a for loop (this sometimes requires padding)
          # 'padding' gives how many elements need to be added for the 'batch_size' to be divisible by 'pool' (The outer modulo makes the padding 0 when the remainder (batch_size % pool) is 0)
          padding = tf.math.floormod(self.pool - tf.math.floormod(batch_size, self.pool), self.pool)
          newBatch = tf.math.floordiv(batch_size + padding, self.pool)

          # Reshape inputs from [batch_size, ...] to [pool, newBatch, ...]
          inputs = tf.reshape(tf.pad(inputs, [[0, padding], [0, 0]]), (self.pool, newBatch, tf.shape(inputs)[-1]))
          h_tm1 = tf.reshape(tf.pad(h_tm1, [[0, padding], [0, 0]]), (self.pool, newBatch, tf.shape(h_tm1)[-1]))

          num_matrices = self.pool

        if (self.Wstd != 0):
          Werr = Merr_distr([num_matrices, *kernel.shape], self.Wstd, self.d_type, self.errDistr)
          RWerr = Merr_distr([num_matrices, *recurrent_kernel.shape], self.Wstd, self.d_type, self.errDistr)

          Zerr = Merr_distr([num_matrices, *zeta.shape], self.Wstd, self.d_type, self.errDistr)
          NUerr = Merr_distr([num_matrices, *nu.shape], self.Wstd, self.d_type, self.errDistr)
        else:
          Werr = self.Werr
          RWerr = self.RWerr

          Zerr = self.Zerr
          NUerr = self.NUerr

        memW = kernel*Werr
        memRW = recurrent_kernel*RWerr

        memZ = zeta*Zerr
        memNU = nu*NUerr
        
        if (self.Bstd !=0):
          BHerr = Merr_distr([num_matrices, *self.bias_h.shape], self.Bstd, self.d_type, self.errDistr)
          BZerr = Merr_distr([num_matrices, *self.bias_z.shape], self.Bstd, self.d_type, self.errDistr)
        else:
          BHerr = self.BHerr
          BZerr = self.BZerr

        membias_h = self.bias_h*BHerr
        membias_z = self.bias_z*BZerr
        
        if self.pool is not None: # When using pool, some matrices require a specific shape to enable broadcasting in certain operations
          memZ = tf.expand_dims(memZ, 1)
          memNU = tf.expand_dims(memNU, 1)

          membias_z = tf.expand_dims(membias_z, 1)
          membias_h = tf.expand_dims(membias_h, 1)

      else: # Wstd = Bstd = 0
        # TODO: Check why we multiply by the error matrices if Wstd = Bstd = 0 means the error matrices are a constant 1 (so multiplying by them does nothing at all) [unless we somehow change them outside of the code before training, which is not an expected case anyway]
        
        if(self.isBin==True or self.isBin=='yes'):
          recurrent_kernel = self.sign(self.recurrent_kernel)*self.RWerr
          kernel = self.sign(self.kernel)*self.Werr

          # Don't binarize 'zeta' or 'nu' parameters
          #zeta = self.sign(self.zeta)*self.Zerr
          #nu = self.sign(self.nu)*self.NUerr
        else:
          recurrent_kernel = self.recurrent_kernel*self.RWerr
          kernel = self.kernel*self.Werr

        zeta = self.zeta*self.Zerr
        nu = self.nu*self.NUerr
        
        bias_h = self.bias_h*self.BHerr
        bias_z = self.bias_z*self.BZerr

    else: # Inference # Preparation of matrices/error matrices

      if(self.Wstd != 0):
        Werr = self.infWerr
        RWerr = self.infRWerr

        Zerr = self.infZerr
        NUerr = self.infNUerr

      else:
        Werr = self.Werr
        RWerr = self.RWerr

        Zerr = self.Zerr
        NUerr = self.NUerr

      if(self.isBin==True or self.isBin=='yes'):
        kernel = tf.math.sign(self.kernel)*Werr
        recurrent_kernel = tf.math.sign(self.recurrent_kernel)*RWerr

        # Don't binarize 'zeta' or 'nu' parameters
        #zeta = tf.math.sign(self.zeta)*Zerr
        #nu = tf.math.sign(self.nu)*NUerr
      else:
        kernel = self.kernel*Werr
        recurrent_kernel = self.recurrent_kernel*RWerr

      zeta = self.zeta*Zerr
      nu = self.nu*NUerr

      if(self.Bstd != 0):
        BHerr = self.infBHerr
        BZerr = self.infBZerr

      else:
        BHerr = self.BHerr 
        BZerr = self.BZerr 

      bias_h = self.bias_h * BHerr
      bias_z = self.bias_z * BZerr
    # --

    # z -> z_t
    # htilde -> h_t^~

    # AConnect 
    if training and (self.Wstd != 0 or self.Bstd != 0): # Training (applying error) # Calculating the output

      z = backend.batch_dot(inputs, memW) + backend.batch_dot(h_tm1, memRW) # W*xt + U*h_tm1
      htilde = z # W*xt + U*h_tm1

      z = self.activation(tf.add(z, membias_z))
      htilde = self.update_activation(tf.add(htilde, membias_h))

      h = (tf.keras.activations.sigmoid(memZ)*(1 - z) + tf.keras.activations.sigmoid(memNU)) * htilde + z * h_tm1

      if self.pool is not None: 
        # When using a pool of error matrices, the input is reshaped from [batch_size, input_dim] to [pool, newBatch, input_dim] 
        # so the output ends up with shape [pool, newBatch, units] and has to be reshaped to [batch_size, units]
        # Also sometimes the 'inputs' is padded, so we remove the padding from the output
        h = tf.reshape(h, (batch_size + padding, tf.shape(h)[-1]))[:batch_size]

    else: # Inference # Calculating the output
      z = backend.dot(inputs, kernel) + backend.dot(h_tm1, recurrent_kernel) # W*xt + U*h_tm1
      htilde = z # W*xt + U*h_tm1

      z = self.activation(backend.bias_add(z, bias_z))
      htilde = self.update_activation(backend.bias_add(htilde, bias_h))

      h = (tf.keras.activations.sigmoid(zeta)*(1 - z) + tf.keras.activations.sigmoid(nu)) * htilde + z * h_tm1
    # --

    return h, [h]

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'update_activation':
            activations.serialize(self.update_activation),
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'zetaInit':
            self.zetaInit,
        'nuInit':
            self.nuInit,
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        # AConnect
        'Wstd':
            self.Wstd,
        'Bstd':
            self.Bstd,
        'errDistr':
            self.errDistr,
        'pool':
            self.pool,
        'isBin':
            self.isBin,
        'd_type':
            self.d_type
        # --
    }
    config.update(_config_for_enable_caching_device(self))
    base_config = super(FastGRNNCell_AConnect, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return list(_generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype))

# --- LSTM
# Modified version of tf.keras.layers.LSTMCell https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/python/keras/layers/recurrent.py#L2043-L2314

from keras.layers.recurrent import _caching_device

class LSTMCell_AConnect(keras.layers.recurrent.LSTMCell):
  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal', # Orthogonal initializer doesn't seem to support float16 values (An operation inside: tf.linalg.qr throws an error)
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,

               # AConnect
               Wstd=0, # standard deviation of the weights 
               Bstd=0, # standard deviation of the bias
               errDistr="normal",
               pool=None,
               isBin=False, 
               d_type=tf.dtypes.float32,
               # --

               **kwargs):
    
    self.Wstd = Wstd
    self.Bstd = Bstd
    self.errDistr = errDistr
    self.pool = pool
    self.isBin = isBin
    self.d_type = d_type

    super(LSTMCell_AConnect, self).__init__(units,
                                            activation=activation,
                                            recurrent_activation=recurrent_activation,
                                            use_bias=use_bias,
                                            kernel_initializer=kernel_initializer,
                                            recurrent_initializer=recurrent_initializer,
                                            bias_initializer=bias_initializer,
                                            unit_forget_bias=unit_forget_bias,
                                            kernel_regularizer=kernel_regularizer,
                                            recurrent_regularizer=recurrent_regularizer,
                                            bias_regularizer=bias_regularizer,
                                            kernel_constraint=kernel_constraint,
                                            recurrent_constraint=recurrent_constraint,
                                            bias_constraint=bias_constraint,
                                            dropout=dropout,
                                            recurrent_dropout=recurrent_dropout,
                                            **kwargs)
  
  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    default_caching_device = _caching_device(self)
    input_dim = input_shape[-1]
    
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 4),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device,
        # AConnect
        dtype = self.d_type
        # --
        )
    
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device,
        # AConnect
        dtype = self.d_type
        # --
        )

    if self.use_bias:
      if self.unit_forget_bias:

        def bias_initializer(_, *args, **kwargs):
          return backend.concatenate([
              self.bias_initializer((self.units,), *args, **kwargs),
              initializers.get('ones')((self.units,), *args, **kwargs),
              self.bias_initializer((self.units * 2,), *args, **kwargs),
          ])
      else:
        bias_initializer = self.bias_initializer

      self.bias = self.add_weight(
          shape=(self.units * 4,),
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          caching_device=default_caching_device)
      
    else:
      self.bias = None

    # AConnect
    if (self.Wstd != 0):
      self.infWerr = Merr_distr(self.kernel.shape, self.Wstd, self.d_type, self.errDistr)
      self.infWerr = self.infWerr.numpy()                                                   

      self.infRWerr = Merr_distr(self.recurrent_kernel.shape, self.Wstd, self.d_type, self.errDistr)
      self.infRWerr = self.infRWerr.numpy()

    else:
      self.Werr = tf.constant(1, dtype=self.d_type)
      self.RWerr = tf.constant(1, dtype=self.d_type)

    if self.use_bias:
      if(self.Bstd != 0):
        self.infBerr = Merr_distr(self.bias.shape, self.Bstd, self.d_type, self.errDistr)
        self.infBerr = self.infBerr.numpy()

      else:
        self.Berr = tf.constant(1, dtype=self.d_type)
    # --

    self.built = True
  
  def _compute_carry_and_output(self, x, h_tm1, c_tm1, recurrent_kernel):
      """Computes carry and output using split kernels."""
      x_i, x_f, x_c, x_o = x
      h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
      i = self.recurrent_activation(
          x_i + backend.dot(h_tm1_i, recurrent_kernel[:, :self.units]))
      f = self.recurrent_activation(x_f + backend.dot(
          h_tm1_f, recurrent_kernel[:, self.units:self.units * 2]))
      c = f * c_tm1 + i * self.activation(x_c + backend.dot(
          h_tm1_c, recurrent_kernel[:, self.units * 2:self.units * 3]))
      o = self.recurrent_activation(
          x_o + backend.dot(h_tm1_o, recurrent_kernel[:, self.units * 3:]))
      return c, o

  def _compute_carry_and_output_batch(self, x, h_tm1, c_tm1, recurrent_kernel):
      """Computes carry and output using split kernels."""
      x_i, x_f, x_c, x_o = x
      h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
      i = self.recurrent_activation(
          x_i + backend.batch_dot(h_tm1_i, recurrent_kernel[:, :, :self.units]))
      f = self.recurrent_activation(x_f + backend.batch_dot(
          h_tm1_f, recurrent_kernel[:, :, self.units:self.units * 2]))
      c = f * c_tm1 + i * self.activation(x_c + backend.batch_dot(
          h_tm1_c, recurrent_kernel[:, :, self.units * 2:self.units * 3]))
      o = self.recurrent_activation(
          x_o + backend.batch_dot(h_tm1_o, recurrent_kernel[:, :, self.units * 3:]))
      return c, o

  @tf.custom_gradient
  def sign(self, x):
    y = tf.math.sign(x)
    def grad(dy):
      dydx = tf.divide(dy, abs(x)+1e-5)
      return dydx
    return y, grad

  def call(self, inputs, states, training=None):
    # AConnect
    inputs = tf.cast(inputs, dtype=self.d_type)
    batch_size = tf.shape(inputs)[0]
    states = [tf.cast(s, dtype=self.d_type) for s in states]
    # --

    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)

    # AConnect
    if (training): # Training # Preparation of matrices/error matrices

      if (self.Wstd != 0 or self.Bstd != 0):
    
        if(self.isBin==True or self.isBin=="yes"):
          kernel = self.sign(self.kernel)
          recurrent_kernel = self.sign(self.recurrent_kernel)
        else:
          kernel = self.kernel
          recurrent_kernel = self.recurrent_kernel

        if (self.pool is None):
          num_matrices = batch_size

        else:
          # When using pool, we shape the inputs and matrices in a way that allows us to use batch_dot function to calculate the output, preventing the use of a for loop (this sometimes requires padding)
          # 'padding' gives how many elements need to be added for the 'batch_size' to be divisible by 'pool' (The outer modulo makes the padding 0 when the remainder (batch_size % pool) is 0)
          padding = tf.math.floormod(self.pool - tf.math.floormod(batch_size, self.pool), self.pool)
          newBatch = tf.math.floordiv(batch_size + padding, self.pool)

          # Reshape inputs from [batch_size, ...] to [pool, newBatch, ...]
          inputs = tf.reshape(tf.pad(inputs, [[0, padding], [0, 0]]), (self.pool, newBatch, tf.shape(inputs)[-1]))
          h_tm1 = tf.reshape(tf.pad(h_tm1, [[0, padding], [0, 0]]), (self.pool, newBatch, tf.shape(h_tm1)[-1]))

          if dp_mask is not None:
            dp_mask = [tf.reshape(tf.pad(x, [[0, padding], [0, 0]]), (self.pool, newBatch, tf.shape(x)[-1])) for x in dp_mask]
          if rec_dp_mask is not None:
            rec_dp_mask = [tf.reshape(tf.pad(x, [[0, padding], [0, 0]]), (self.pool, newBatch, tf.shape(x)[-1])) for x in rec_dp_mask]

          if self.implementation == 1:
            c_tm1 = tf.reshape(tf.pad(c_tm1, [[0, padding], [0, 0]]), (self.pool, newBatch, tf.shape(c_tm1)[-1]))

          num_matrices = self.pool

        if (self.Wstd !=0):
          Werr = Merr_distr([num_matrices, *kernel.shape], self.Wstd, self.d_type, self.errDistr)
          RWerr = Merr_distr([num_matrices, *recurrent_kernel.shape], self.Wstd, self.d_type, self.errDistr)
        else:
          Werr = self.Werr
          Rerr = self.RWerr

        memW = kernel*Werr
        memRW = recurrent_kernel*RWerr

        if (self.use_bias):

          if (self.Bstd !=0):
            Berr = Merr_distr([num_matrices, *self.bias.shape], self.Bstd, self.d_type, self.errDistr)
          else:
            Berr = self.Berr

          membias = self.bias*Berr

          if self.pool is not None: # When using pool, some matrices require a specific shape to enable broadcasting in certain operations
            membias = tf.expand_dims(membias, 1)

      else:

        if(self.isBin==True or self.isBin=='yes'):
          recurrent_kernel = self.sign(self.recurrent_kernel)*self.RWerr
          kernel = self.sign(self.kernel)*self.Werr
        else:
          recurrent_kernel = self.recurrent_kernel*self.RWerr
          kernel = self.kernel*self.Werr
        
        if (self.use_bias):
          bias = self.bias*self.Berr

        
    else: # Inference # Preparation of matrices/error matrices
      if(self.Wstd != 0):
        Werr = self.infWerr
        RWerr = self.infRWerr
      else:
        Werr = self.Werr
        RWerr = self.RWerr

      if(self.isBin==True or self.isBin=='yes'):
        kernel = tf.math.sign(self.kernel)*Werr
        recurrent_kernel = tf.math.sign(self.recurrent_kernel)*RWerr
      else:
        kernel = self.kernel*Werr
        recurrent_kernel = self.recurrent_kernel*RWerr

      if (self.use_bias):

        if(self.Bstd != 0):
          Berr = self.infBerr
        else:
          Berr = self.Berr 

        bias = self.bias * Berr
    # --

    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        inputs_i = inputs * dp_mask[0]
        inputs_f = inputs * dp_mask[1]
        inputs_c = inputs * dp_mask[2]
        inputs_o = inputs * dp_mask[3]
      else:
        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs

      # AConnect
      if training and (self.Wstd != 0 or self.Bstd != 0): # Training (applying error) # Calculating the output
        
        k_i, k_f, k_c, k_o = tf.split(
            memW, num_or_size_splits=4, axis=2)
        x_i = backend.batch_dot(inputs_i, k_i)
        x_f = backend.batch_dot(inputs_f, k_f)
        x_c = backend.batch_dot(inputs_c, k_c)
        x_o = backend.batch_dot(inputs_o, k_o)

        if self.use_bias:
          if self.pool is None:
            axis = 1
          else:
            axis = 2

          b_i, b_f, b_c, b_o = tf.split(
              membias, num_or_size_splits=4, axis=axis)
          x_i = tf.add(x_i, b_i)
          x_f = tf.add(x_f, b_f)
          x_c = tf.add(x_c, b_c)
          x_o = tf.add(x_o, b_o)
          
          if self.pool is not None:
            pass # TODO

      else: # Inference # Calculating the output
        k_i, k_f, k_c, k_o = tf.split(
            kernel, num_or_size_splits=4, axis=1)
        x_i = backend.dot(inputs_i, k_i)
        x_f = backend.dot(inputs_f, k_f)
        x_c = backend.dot(inputs_c, k_c)
        x_o = backend.dot(inputs_o, k_o)
        if self.use_bias:
          b_i, b_f, b_c, b_o = tf.split(
              bias, num_or_size_splits=4, axis=0)
          x_i = backend.bias_add(x_i, b_i)
          x_f = backend.bias_add(x_f, b_f)
          x_c = backend.bias_add(x_c, b_c)
          x_o = backend.bias_add(x_o, b_o)
      # --

      if 0 < self.recurrent_dropout < 1.:
        h_tm1_i = h_tm1 * rec_dp_mask[0]
        h_tm1_f = h_tm1 * rec_dp_mask[1]
        h_tm1_c = h_tm1 * rec_dp_mask[2]
        h_tm1_o = h_tm1 * rec_dp_mask[3]
      else:
        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
      x = (x_i, x_f, x_c, x_o)
      h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)

      # AConnect
      if training and (self.Wstd != 0 or self.Bstd != 0): # Training (applying error) # Calculating the output
        c, o = self._compute_carry_and_output_batch(x, h_tm1, c_tm1, memRW)
        if self.pool is not None:
          # When using a pool of error matrices, the input is reshaped from [batch_size, input_dim] to [pool, newBatch, input_dim] 
          # so the output ends up with shape [pool, newBatch, units] and has to be reshaped to [batch_size, units]
          # Also sometimes the 'inputs' is padded, so we remove the padding from the output
          c = tf.reshape(c, (batch_size + padding, tf.shape(c)[-1]))[:batch_size]
          o = tf.reshape(o, (batch_size + padding, tf.shape(o)[-1]))[:batch_size]

      else: # Inference # Calculating the output
        c, o = self._compute_carry_and_output(x, h_tm1, c_tm1, recurrent_kernel)
      # --

    else: # implementation != 1
      if 0. < self.dropout < 1.:
        inputs = inputs * dp_mask[0]
      
      # AConnect
      if training and (self.Wstd != 0 or self.Bstd != 0): # Training (applying error) # Calculating the output
        z = backend.batch_dot(inputs, memW)
        z += backend.batch_dot(h_tm1, memRW)
        if self.use_bias:
          z = tf.add(z, membias)

        if self.pool is not None:
          # When using a pool of error matrices, the input is reshaped from [batch_size, input_dim] to [pool, newBatch, input_dim] 
          # so the output ends up with shape [pool, newBatch, units] and has to be reshaped to [batch_size, units]
          # Also sometimes the 'inputs' is padded, so we remove the padding from the output
          z = tf.reshape(z, (batch_size + padding, tf.shape(z)[-1]))[:batch_size]

      else: # Inference # Calculating the output
        z = backend.dot(inputs, kernel)
        z += backend.dot(h_tm1, recurrent_kernel)
        if self.use_bias:
          z = backend.bias_add(z, bias)
      # --

      z = tf.split(z, num_or_size_splits=4, axis=1)
      c, o = self._compute_carry_and_output_fused(z, c_tm1)
 
    h = o * self.activation(c)
    return h, [h, c]

  def get_config(self):
    config = {
        # AConnect
        'Wstd':
            self.Wstd,
        'Bstd':
            self.Bstd,
        'errDistr':
            self.errDistr,
        'pool':
            self.pool,
        'isBin':
            self.isBin,
        'd_type':
            self.d_type
        # --
    }
    config.update(_config_for_enable_caching_device(self))
    base_config = super(LSTMCell_AConnect, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
    
    
# --- GRU
# Modified version of tf.keras.layers.GRUCell https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/python/keras/layers/recurrent.py#L1496-L1762

from keras.layers.recurrent import _caching_device
from keras.layers.recurrent import _config_for_enable_caching_device

class GRUCell_AConnect(keras.layers.recurrent.GRUCell):

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               reset_after=False,

               # AConnect
               Wstd=0, # standard deviation of the weights 
               Bstd=0, # standard deviation of the bias
               errDistr="normal",
               pool=None,
               isBin=False, 
               d_type=tf.dtypes.float32,
               # --

               **kwargs):
    
    self.Wstd = Wstd
    self.Bstd = Bstd
    self.errDistr = errDistr
    self.pool = pool
    self.isBin = isBin
    self.d_type = d_type

    super(GRUCell_AConnect, self).__init__(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=kwargs.pop('implementation', 2),
        reset_after=reset_after,
        **kwargs)

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    input_dim = input_shape[-1]
    default_caching_device = _caching_device(self)

    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 3),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device,
        # AConnect
        dtype = self.d_type
        # --
        )
    
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 3),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device,
        # AConnect
        dtype = self.d_type
        # --
        )

    if self.use_bias:
      if not self.reset_after:
        bias_shape = (3 * self.units,)
      else:
        # separate biases for input and recurrent kernels
        # Note: the shape is intentionally different from CuDNNGRU biases
        # `(2 * 3 * self.units,)`, so that we can distinguish the classes
        # when loading and converting saved weights.
        bias_shape = (2, 3 * self.units)
      self.bias = self.add_weight(shape=bias_shape,
                                  name='bias',
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint,
                                  caching_device=default_caching_device,
                                  # AConnect
                                  dtype = self.d_type
                                  # --
                                  )
    else:
      self.bias = None

    # AConnect
    if (self.Wstd != 0):
      self.infWerr = Merr_distr(self.kernel.shape, self.Wstd, self.d_type, self.errDistr)
      self.infWerr = self.infWerr.numpy()                                                   

      self.infRWerr = Merr_distr(self.recurrent_kernel.shape, self.Wstd, self.d_type, self.errDistr)
      self.infRWerr = self.infRWerr.numpy()

    else:
      self.Werr = tf.constant(1, dtype=self.d_type)
      self.RWerr = tf.constant(1, dtype=self.d_type)

    if self.use_bias:
      if(self.Bstd != 0):
        self.infBerr = Merr_distr(self.bias.shape, self.Bstd, self.d_type, self.errDistr)
        self.infBerr = self.infBerr.numpy()

      else:
        self.Berr = tf.constant(1, dtype=self.d_type)
    # --

    self.built = True

  @tf.custom_gradient
  def sign(self, x):
    y = tf.math.sign(x)
    def grad(dy):
      dydx = tf.divide(dy, abs(x)+1e-5)
      return dydx
    return y, grad


  def call(self, inputs, states, training=None):
    # AConnect
    inputs = tf.cast(inputs, dtype=self.d_type)
    batch_size = tf.shape(inputs)[0]
    states = [tf.cast(s, dtype=self.d_type) for s in states] if tf.nest.is_nested(states) else tf.cast(states, dtype=self.d_type)
    # --

    h_tm1 = states[0] if tf.nest.is_nested(states) else states  # previous memory

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=3)

    # AConnect
    if (training): # Training # Preparation of matrices/error matrices

      if (self.Wstd != 0 or self.Bstd != 0):
    
        if(self.isBin==True or self.isBin=="yes"):
          kernel = self.sign(self.kernel)
          recurrent_kernel = self.sign(self.recurrent_kernel)
        else:
          kernel = self.kernel
          recurrent_kernel = self.recurrent_kernel

        if (self.pool is None):
          num_matrices = batch_size

        else:
          # When using pool, we shape the inputs and matrices in a way that allows us to use batch_dot function to calculate the output, preventing the use of a for loop (this sometimes requires padding)
          # 'padding' gives how many elements need to be added for the 'batch_size' to be divisible by 'pool' (The outer modulo makes the padding 0 when the remainder (batch_size % pool) is 0)
          padding = tf.math.floormod(self.pool - tf.math.floormod(batch_size, self.pool), self.pool)
          newBatch = tf.math.floordiv(batch_size + padding, self.pool)

          # Reshape inputs from [batch_size, ...] to [pool, newBatch, ...]
          inputs = tf.reshape(tf.pad(inputs, [[0, padding], [0, 0]]), (self.pool, newBatch, tf.shape(inputs)[-1]))
          h_tm1 = tf.reshape(tf.pad(h_tm1, [[0, padding], [0, 0]]), (self.pool, newBatch, tf.shape(h_tm1)[-1]))

          if dp_mask is not None:
            dp_mask = [tf.reshape(tf.pad(x, [[0, padding], [0, 0]]), (self.pool, newBatch, tf.shape(x)[-1])) for x in dp_mask]
          if rec_dp_mask is not None:
            rec_dp_mask = [tf.reshape(tf.pad(x, [[0, padding], [0, 0]]), (self.pool, newBatch, tf.shape(x)[-1])) for x in rec_dp_mask]

          num_matrices = self.pool

        if (self.Wstd !=0):
          Werr = Merr_distr([num_matrices, *kernel.shape], self.Wstd, self.d_type, self.errDistr)
          RWerr = Merr_distr([num_matrices, *recurrent_kernel.shape], self.Wstd, self.d_type, self.errDistr)
        else:
          Werr = self.Werr
          Rerr = self.RWerr

        memW = kernel*Werr
        memRW = recurrent_kernel*RWerr

        if (self.use_bias):

          if (self.Bstd !=0):
            Berr = Merr_distr([num_matrices, *self.bias.shape], self.Bstd, self.d_type, self.errDistr)
          else:
            Berr = self.Berr

          membias = self.bias*Berr

          if self.pool is not None: # When using pool, some matrices require a specific shape to enable broadcasting in certain operations
            membias = tf.expand_dims(membias, 1)

      else: # Inference # Preparation of matrices/error matrices

        if(self.isBin==True or self.isBin=='yes'):
          recurrent_kernel = self.sign(self.recurrent_kernel)*self.RWerr
          kernel = self.sign(self.kernel)*self.Werr
        else:
          recurrent_kernel = self.recurrent_kernel*self.RWerr
          kernel = self.kernel*self.Werr
        
        if (self.use_bias):
          bias = self.bias*self.Berr

        
    else: # Inference
      if(self.Wstd != 0):
        Werr = self.infWerr
        RWerr = self.infRWerr
      else:
        Werr = self.Werr
        RWerr = self.RWerr

      if(self.isBin==True or self.isBin=='yes'):
        kernel = tf.math.sign(self.kernel)*Werr
        recurrent_kernel = tf.math.sign(self.recurrent_kernel)*RWerr
      else:
        kernel = self.kernel*Werr
        recurrent_kernel = self.recurrent_kernel*RWerr

      if (self.use_bias):

        if(self.Bstd != 0):
          Berr = self.infBerr
        else:
          Berr = self.Berr 

        bias = self.bias * Berr
    # --

    
    if self.use_bias:
      if not self.reset_after:
        if training and (self.Wstd != 0 or self.Bstd != 0):
          input_bias, recurrent_bias = membias, None
        else:
          input_bias, recurrent_bias = bias, None
      else:
        if training and (self.Wstd != 0 or self.Bstd != 0):
          input_bias, recurrent_bias = tf.unstack(membias, axis=1)
        else:
          input_bias, recurrent_bias = tf.unstack(bias)


    if self.implementation == 1:
      if 0. < self.dropout < 1.:
        inputs_z = inputs * dp_mask[0]
        inputs_r = inputs * dp_mask[1]
        inputs_h = inputs * dp_mask[2]
      else:
        inputs_z = inputs
        inputs_r = inputs
        inputs_h = inputs

      # AConnect
      if training and (self.Wstd != 0 or self.Bstd != 0): # Training (applying error) # Calculating the output
        x_z = backend.batch_dot(inputs_z, memW[:, :, :self.units]) # [:, ...]
        x_r = backend.batch_dot(inputs_r, memW[:, :, self.units:self.units * 2]) # [:, ...]
        x_h = backend.batch_dot(inputs_h, memW[:, :, self.units * 2:]) # [:, ...]

        if self.use_bias:
          if self.pool is None:
            x_z = tf.add(x_z, input_bias[:, :self.units]) # [:, ...]
            x_r = tf.add(x_r, input_bias[:, self.units: self.units * 2]) # [:, ...]
            x_h = tf.add(x_h, input_bias[:, self.units * 2:]) # [:, ...]
          else:
            x_z = tf.add(x_z, input_bias[:, :, :self.units]) # [:, :, ...]
            x_r = tf.add(x_r, input_bias[:, :, self.units: self.units * 2]) # [:, :, ...]
            x_h = tf.add(x_h, input_bias[:, :, self.units * 2:]) # [:, :, ...]

        if 0. < self.recurrent_dropout < 1.:
          h_tm1_z = h_tm1 * rec_dp_mask[0]
          h_tm1_r = h_tm1 * rec_dp_mask[1]
          h_tm1_h = h_tm1 * rec_dp_mask[2]
        else:
          h_tm1_z = h_tm1
          h_tm1_r = h_tm1
          h_tm1_h = h_tm1

        recurrent_z = backend.batch_dot(h_tm1_z, memRW[:, :, :self.units]) # [:, ...]
        recurrent_r = backend.batch_dot(
            h_tm1_r, memRW[:, :, self.units:self.units * 2]) # [:, ...]
        if self.reset_after and self.use_bias:
          if self.pool is None:
            recurrent_z = tf.add(recurrent_z, recurrent_bias[:, :self.units]) # [:, ...]
            recurrent_r = tf.add(
                recurrent_r, recurrent_bias[:, self.units:self.units * 2]) # [:, ...]
          else:
            recurrent_z = tf.add(recurrent_z, recurrent_bias[:, :, :self.units]) # [:, :, ...]
            recurrent_r = tf.add(
                recurrent_r, recurrent_bias[:, :, self.units:self.units * 2]) # [:, :, ...]

        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)

        # reset gate applied after/before matrix multiplication
        if self.reset_after:
          recurrent_h = backend.batch_dot(
              h_tm1_h, memRW[:, :, self.units * 2:]) # [:, ...]
          if self.use_bias:
            if self.pool is None:
              recurrent_h = tf.add(
                  recurrent_h, recurrent_bias[:, self.units * 2:]) # [:, ...]
            else:
              recurrent_h = tf.add(
                  recurrent_h, recurrent_bias[:, :, self.units * 2:]) # [:, :, ...]
          recurrent_h = r * recurrent_h
        else:
          recurrent_h = backend.batch_dot(
              r * h_tm1_h, memRW[:, :, self.units * 2:]) # [:, ...]

        if self.pool is not None:
          # When using a pool of error matrices, the input is reshaped from [batch_size, input_dim] to [pool, newBatch, input_dim] 
          # so the output ends up with shape [pool, newBatch, units] and has to be reshaped to [batch_size, units]
          # Also sometimes the 'inputs' is padded, so we remove the padding from the output
          z = tf.reshape(z, (batch_size + padding, tf.shape(z)[-1]))[:batch_size]
          h_tm1 = tf.reshape(h_tm1, (batch_size + padding, tf.shape(h_tm1)[-1]))[:batch_size]
          x_h = tf.reshape(x_h, (batch_size + padding, tf.shape(x_h)[-1]))[:batch_size]
          recurrent_h = tf.reshape(recurrent_h, (batch_size + padding, tf.shape(recurrent_h)[-1]))[:batch_size]

      else: # Inference # Calculating the output
        x_z = backend.dot(inputs_z, kernel[:, :self.units])
        x_r = backend.dot(inputs_r, kernel[:, self.units:self.units * 2])
        x_h = backend.dot(inputs_h, kernel[:, self.units * 2:])

        if self.use_bias:
          x_z = backend.bias_add(x_z, input_bias[:self.units])
          x_r = backend.bias_add(x_r, input_bias[self.units: self.units * 2])
          x_h = backend.bias_add(x_h, input_bias[self.units * 2:])

        if 0. < self.recurrent_dropout < 1.:
          h_tm1_z = h_tm1 * rec_dp_mask[0]
          h_tm1_r = h_tm1 * rec_dp_mask[1]
          h_tm1_h = h_tm1 * rec_dp_mask[2]
        else:
          h_tm1_z = h_tm1
          h_tm1_r = h_tm1
          h_tm1_h = h_tm1

        recurrent_z = backend.dot(h_tm1_z, recurrent_kernel[:, :self.units])
        recurrent_r = backend.dot(
            h_tm1_r, recurrent_kernel[:, self.units:self.units * 2])
        if self.reset_after and self.use_bias:
          recurrent_z = backend.bias_add(recurrent_z, recurrent_bias[:self.units])
          recurrent_r = backend.bias_add(
              recurrent_r, recurrent_bias[self.units:self.units * 2])

        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)

        # reset gate applied after/before matrix multiplication
        if self.reset_after:
          recurrent_h = backend.dot(
              h_tm1_h, recurrent_kernel[:, self.units * 2:])
          if self.use_bias:
            recurrent_h = backend.bias_add(
                recurrent_h, recurrent_bias[self.units * 2:])
          recurrent_h = r * recurrent_h
        else:
          recurrent_h = backend.dot(
              r * h_tm1_h, recurrent_kernel[:, self.units * 2:])
      # --

      hh = self.activation(x_h + recurrent_h)

    else: # implementation != 1
      if 0. < self.dropout < 1.:
        inputs = inputs * dp_mask[0]

      # AConnect
      if training and (self.Wstd != 0 or self.Bstd != 0): # Training (applying error) # Calculating the output
        # inputs projected by all gate matrices at once
        matrix_x = backend.batch_dot(inputs, memW)
        if self.use_bias:
          # biases: bias_z_i, bias_r_i, bias_h_i
          matrix_x = tf.add(matrix_x, input_bias)

        x_z, x_r, x_h = tf.split(matrix_x, 3, axis=-1)

        if self.reset_after:
          # hidden state projected by all gate matrices at once
          matrix_inner = backend.batch_dot(h_tm1, memRW)
          if self.use_bias:
            matrix_inner = tf.add(matrix_inner, recurrent_bias)
        else:
          # hidden state projected separately for update/reset and new
          matrix_inner = backend.batch_dot(
              h_tm1, memRW[:, :, :2 * self.units]) # [:, ...]

        recurrent_z, recurrent_r, recurrent_h = tf.split(
            matrix_inner, [self.units, self.units, -1], axis=-1)

        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)

        if self.reset_after:
          recurrent_h = r * recurrent_h
        else:
          recurrent_h = backend.batch_dot(
              r * h_tm1, memRW[:, :, 2 * self.units:]) # [:, ...]

        if self.pool is not None:
          # When using a pool of error matrices, the input is reshaped from [batch_size, input_dim] to [pool, newBatch, input_dim] 
          # so the output ends up with shape [pool, newBatch, units] and has to be reshaped to [batch_size, units]
          # Also sometimes the 'inputs' is padded, so we remove the padding from the output
          z = tf.reshape(z, (batch_size + padding, tf.shape(z)[-1]))[:batch_size]
          h_tm1 = tf.reshape(h_tm1, (batch_size + padding, tf.shape(h_tm1)[-1]))[:batch_size]
          x_h = tf.reshape(x_h, (batch_size + padding, tf.shape(x_h)[-1]))[:batch_size]
          recurrent_h = tf.reshape(recurrent_h, (batch_size + padding, tf.shape(recurrent_h)[-1]))[:batch_size]

      else: # Inference # Calculating the output
        # inputs projected by all gate matrices at once
        matrix_x = backend.dot(inputs, kernel)
        if self.use_bias:
          # biases: bias_z_i, bias_r_i, bias_h_i
          matrix_x = backend.bias_add(matrix_x, input_bias)

        x_z, x_r, x_h = tf.split(matrix_x, 3, axis=-1)

        if self.reset_after:
          # hidden state projected by all gate matrices at once
          matrix_inner = backend.dot(h_tm1, recurrent_kernel)
          if self.use_bias:
            matrix_inner = backend.bias_add(matrix_inner, recurrent_bias)
        else:
          # hidden state projected separately for update/reset and new
          matrix_inner = backend.dot(
              h_tm1, recurrent_kernel[:, :2 * self.units])

        recurrent_z, recurrent_r, recurrent_h = tf.split(
            matrix_inner, [self.units, self.units, -1], axis=-1)

        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)

        if self.reset_after:
          recurrent_h = r * recurrent_h
        else:
          recurrent_h = backend.dot(
              r * h_tm1, recurrent_kernel[:, 2 * self.units:])
      # --   

      hh = self.activation(x_h + recurrent_h)

    # previous and candidate state mixed by update gate
    h = z * h_tm1 + (1 - z) * hh
    new_state = [h] if tf.nest.is_nested(states) else h
    return h, new_state

  def get_config(self):
    config = {
        # AConnect
        'Wstd':
            self.Wstd,
        'Bstd':
            self.Bstd,
        'errDistr':
            self.errDistr,
        'pool':
            self.pool,
        'isBin':
            self.isBin,
        'd_type':
            self.d_type
        # --
    }
    config.update(_config_for_enable_caching_device(self))
    base_config = super(GRUCell_AConnect, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

# ^ --- RNN LAYERS/CELLS

############################AUXILIAR FUNCTIONS##################################################
def reshape(X,F): #Used to reshape the input data and the noisy filters
    batch_size=tf.shape(X)[0]
    H = tf.shape(X)[1]
    W = tf.shape(X)[2]
    channels_img = tf.shape(X)[3]
    channels = channels_img
    fh = tf.shape(F)[1]
    fw = tf.shape(F)[2]
    out_channels = tf.shape(F)[-1]
    F = tf.transpose(F, [1, 2, 0, 3, 4])
    F = tf.reshape(F, [fh, fw, channels*batch_size, out_channels])
    inp_r = tf.transpose(X, [1, 2, 0, 3])
    inp_r = tf.reshape(inp_r, [1, H, W, batch_size*channels_img])
    return inp_r, F

def Z_reshape(Z,F,X,padding,strides): #Used to reshape the output of the layer
    batch_size=tf.shape(X)[0]
    H = tf.shape(X)[1]
    W = tf.shape(X)[2]
    channels_img = tf.shape(X)[3]
    channels = channels_img
    fh = tf.shape(F)[1]
    fw = tf.shape(F)[2]
    out_channels = tf.shape(F)[-1]
    #tf.print(fh)
    if padding == "SAME":
        return tf.reshape(Z, [tf.floor(tf.cast((H)/strides,dtype=tf.float16)), tf.floor(tf.cast((W)/strides,dtype=tf.float16)), batch_size, channels, out_channels])
    if padding == "VALID":
        return tf.reshape(Z, [tf.floor(tf.cast((H-fh)/strides,dtype=tf.float16))+1, tf.floor(tf.cast((W-fw)/strides,dtype=tf.float16))+1, batch_size, channels, out_channels])
    #return out

def Merr_distr(shape,stddev,dtype,errDistr): #Used to reshape the output of the layer
    N =  tf.random.normal(shape=shape,
                        stddev=stddev,
                        dtype=dtype)

    if errDistr == "normal":
      Merr = tf.math.abs(1+N)
    elif errDistr == "lognormal":
      Merr = tf.math.exp(-N)
    return Merr