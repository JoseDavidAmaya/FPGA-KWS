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
                                        Werr = tf.constant(1,dtype=self.d_type)
                                if(self.Bstd != 0):
                                        Berr = self.infBerr
                                else:
                                        Berr = tf.constant(1,dtype=self.d_type)
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

                super(Conv_AConnect, self).build(input_shape)
        def call(self,X,training):
                self.X = tf.cast(X, dtype=self.d_type)
                self.batch_size = tf.shape(self.X)[0]
                if(training):
                        if(self.Wstd != 0 or self.Bstd != 0):
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
                                        Werr = tf.constant(1,dtype=self.d_type)
                                if(self.Bstd != 0):
                                        Berr = self.infBerr
                                else:
                                        Berr = tf.constant(1,dtype=self.d_type)
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
