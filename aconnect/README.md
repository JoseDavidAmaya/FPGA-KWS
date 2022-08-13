# Library for A-Connect
Here you will find the main script for the library. 

## How to

You can copy and paste this folder in your project folder or simply install the library latest stable version using pip3. Then in your project import the library writing the code below.
```
import aconnect.layers
import aconnect.scripts
```

## Content description

### layers.py

High level script with the both layers available for A-Connect. Fully connected layer (FC_AConnect) and Convolutional layer (Conv_AConnect).

#### Fully-Connected A-Connect layer
FC_AConnect(output_size,Wstd=0,Bstd=0,isBin="no",pool=None,Slice=1,d_type=tf.dtypes.float32,weights_regularizer=None,bias_regularizer=None)

Custom fully-connected or dense layer with A-Connect during the training process. This layer has this attributes:

1. **output_size**: This is the number of neurons that you want to use.
2. **Wstd**: Standard deviation for the weights. Should be a number between 0 and 1. By default is 0.
3. **Bstd**: Standard deviation for the bias. Should be a number between 0 and 1. By default is 0.
4. **isBin**: String, "yes" or "no", for using or not using weights binarization. By default is "no".
5. **pool**: Number of error matrices that you want to use for model regularization or mismatch mitigation. This value should be equal or less than the batch size. By default is None. (Same number of error matrices as the batch size).
6. **Slice**: Integer for batch slicing. Used for reducing memory consumption. The layer only accepts slice into 2,4 or 8 minibatches. By default is 1 (no slice).
7. **d_type**: Data type of the weights, bias and error matrices. By default is floating point 32 bits. (tf.float32)
8. **weights_regularizer**: Any of tf.keras.regularizers for the weights. By default is None.
9. **bias_regularizer**: Any of tf.keras.regularizers for the biases. By default is None.

#### Convolutional A-Connect layer
Conv_AConnect(filters,kernel_size,strides=1,padding="VALID",Wstd=0,Bstd=0,pool=None,isBin='no',Op=1,Slice=1,d_type=tf.dtypes.float32,weights_regularizer=None,bias_regularizer=None)

Custom convolutional layer with A-Connect during the training process. This layer has this attributes:

1. **filters**: This is the number of filters that you want to use. 
2. **kernel_size**: Kernel size. Must be a tuple (n,n).
3. **strides**: Strides for the convolution. Must be a integer to generate a stride of (1,n,n,1).
4. **padding**:  Integer for padding. Use "VALID" if you want to reduce the input size. Use "SAME" if you want to keep the input size.
5. **Wstd**: Standard deviation for the kernel. Should be a number between 0 and 1. By default is 0.
6. **Bstd**: Standard deviation for the bias. Should be a number between 0 and 1. By default is 0.
7. **pool**: Number of error matrices that you want to use for model regularization or mismatch mitigation. This value should be equal or less than the batch size. By default is None. (Same number of error matrices as the batch size).
8. **isBin**: String, "yes" or "no", for using or not using weights binarization. By default is "no".
9. **Op**:  Integer 1 or 2. There are two options to make the batch convolution. The first option, 1, uses a function from tensorflow called tf.map_fn, this function applies the convolution to all the elements in axis 0 or batch axis. This option has less memory consumption, but it is slower. The second one, 2, applies some reshapes to the input data and the kernel to do a depth-wise convolution. This option is faster when it is used for smaller images, but has higher memory usage.
10. **Slice**: Integer for batch slicing. Used for reducing memory consumption. The layer only accepts slice into 2,4 or 8 minibatches. By default is 1 (no slice).
11. **d_type**: Data type of the weights, bias and error matrices. By default is floating point 32 bits. (tf.float32)
12. **weights_regularizer**: Any of tf.keras.regularizers for the weights. By default is None.
13. **bias_regularizer**: Any of tf.keras.regularizers for the biases. By default is None.

### scripts.py

High level script with auxiliar functions for neural network testing.

#### Monte Carlo method
MonteCarlo(net,Xtest,Ytest,M,Wstd,Bstd,force,Derr=0,net_name="Network",custom_objects=None,dtype='float32',optimizer=tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9),loss=['sparse_categorical_crossentropy'],metrics=['accuracy'],top5=False)

Applies the monte carlo method to all the layers that has weights in a saved neural network model. Only works for fully connected and convolutional networks

1. **net**: String with the path and the format of your pre-trained neural network. 
2. **Xtest**: Numpy array with the test images.
3. **Ytest**: Numpy array with the test labels.
4. **M**: Integer. Number of samples for the test.
5. **Wstd**: Float. Simulation error for the weights.
6. **Bstd**: Float. Simulation error for the bias.
7. **force**: String. Must be yes or no. Used if you want to test the network with a different Wstd or Bstd from that one you used during training process.
8. **Derr**: Float. Adds a deterministic error to a binary weights.
9. **net_name**: String. Name that you want to use for saving the results in a txt.
10. **custom_objects**: Python dictionary. When you are using a custom layers, you need to specify this layers for loading the model with this attribute.
11. **dtype**: Data type of the layer parameters.
12. **optimizer**: Optimizer used during the training process.
13. **loss**: Loss function during the training process.
14. **metrics**: Metrics used during the training process.
15. **top5**: Boolean. If you specified a top-5 function in metrics, please use True here. Otherwise use False.

This function returns the top-1 accuracy and the media of this accuracy.
#### Classify
classify(net,Xtest,Ytest,top5)

Function that calculates the network accuracy during the inference.

1. **net**: Model trained.
2. **Xtest**: Test images.
3. **Ytest**: Test labels.
4. **top5***: Boolean, if you want top-5 accuracy.

#### Load MNIST dataset
load_ds(imgSize=[28,28], Quant=8)

Function for loading the standard or a custom MNIST dataset. With this function you cand load MNIST 28x28 8/4 bits or MNIST 11x11 8/4 bits.

1. **imgSize**: Tuple with the image size. Must be [28,28] or [11,11]
2. **Quant**: Integer for image quantization. 8 or 4 bits.

Returns (x_train,y_train),(x_test,y_test) with the size and configuration that you used.

#### Plot Box

plotBox(data,labels,legends,color,color_fill,path)

Function for plotting a box with the monte carlo accuracy results.

1. **data**: A list with the data from the monte carlo. Or a list of 2 list for plotting 2 NN results. Or a list of 3 list for plotting 3 NN results.
2. **labels**: List with the labels for X-axis.
3. **color**: List with the RGB values of the color for the lines.
4. **color_fill**: List with the RGB values of the color for filling the boxes.
5. **path**: String with the path and the file name for saving the graph.
