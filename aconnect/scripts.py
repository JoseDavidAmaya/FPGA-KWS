import numpy as np
import tensorflow as tf
import os
#Function to make the monte carlo simulation. To see more please go to the original file in Scripts
def MonteCarlo(net=None,Xtest=None,Ytest=None,M=100,Wstd=0,Bstd=0,errDistr="normal",
        force="no",Derr=0,net_name="Network",custom_objects=None,dtype='float32',
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1,momentum=0.9),
        loss=['sparse_categorical_crossentropy'],
        metrics=['accuracy'],top5=False,run_model_eagerly=False,evaluate_batch_size=None):
        """
        Input Parameters:
        net: Name of the network model you want to test (it must be saved in the folder Models)
        Xtest and Ytest: Validation/Testing dataset
        M: Number of samples for the Monte Carlo
        Wstd and Bstd: Weights and Bias error for the simulation. It must be a float between 0-1
        force: String, should be "yes" or "no" when you want to use a Wstd or Bstd different from the used during training i.e.
                If you trained A-Connect with 50% and you want to test it with an error of 70% you must define force="yes"
        errDistr: String. Options: "normal" or "lognormal". States the distribution applied over the error matrices (noise)
        Derr: If you want to introduce a deterministic error when you are using BW in the network. Float between 0-1
        net_name = String with the name you want to use to save the simulation results
        custom_objects: Python dictionary with the name of all the custom elements that you used in your model i.e. If you use an A-Connect model with Conv and FC A-Connect custom_objects should be
        custom_objects= {'ConvAConnect':ConvAConnect.ConvAConnect,'AConnect':AConnect.AConnect}
        SRAMsz: Matrix dimension for the static error matrix that you want to generate. It is depend on the dimension of the layer weights
        SRAMBsz: Vector dimension for the static error vector that you want to generate. It is depend on the dimension of the layer weights
        optimizer,loss,metrics: The values that you used during the training
        run_model_eagerly: Set to True to run the noisy model eagerly, can help to increase the performance in certain cases
        evaluate_batch_size: Batch size used when evaluating the model, higher values increase performance at the expense of a higher memory usage
        This function returns the noisy accuracy values and the mean of this values
        """

        ### Script to change the error matrix during inference or introduce the error to the layer weights
        ### HOw to
        """
        Input Parameters
        net: Loaded model
        Wstd: Weights standard deviation for simulation
        Bstd: Bias standard deviation for simulation
        force: When you want to use the training deviation or the simulation deviation
        Derr: Deterministic error
        SRAMsz: ERror matrix size
        SRAMBsz: Error vector size
        This function returns a NoisyNet and the values of Wstd and Bstd used
        """
        def add_Wnoise(net,Wstd,Bstd,errDistr,force,Derr,dtype='float32'):
                layers = net.layers #Get the list of layers used in the model
                Nlayers = np.size(layers) #Get the number of layers

                for i in range(Nlayers): #Iterate over the number of layers
                        if layers[i].count_params() != 0: #If the layer does not have training parameters it is omitted

                                if hasattr(layers[i],'kernel') or hasattr(layers[i],'W'):  #Does the layer have weights or kernel?

                                        Wsz = np.shape(layers[i].weights[0]) #Takes the weights/kernel size
                                        Bsz = np.shape(layers[i].weights[1]) #Takes the bias size
                                        MBerr_aux = np.random.randn(Bsz[0])
                                        if hasattr(layers[i],'strides'): #If the layer have the attribute strides means that it is a convolutional layer
                                                Merr_aux = np.random.randn(Wsz[0],Wsz[1],Wsz[2],Wsz[3]).astype(dtype)
                                        else:
                                                Merr_aux = np.random.randn(Wsz[0], Wsz[1]).astype(dtype) #If the layer does not have strides, it is a FC layer

                                        if hasattr(layers[i], 'Wstd'): #Does the layer have Wstd? if it is true is an A-Connect or DropConnect network
                                                if(layers[i].Wstd != 0): #IF the value it is different from zero, the layer is working with the algorithm
                                                        Wstd_layer = layers[i].Wstd
                                                        if force == "no": #Do you want to take the training or simulation Wstd value?
                                                                Wstd = Wstd_layer
                                                        else:
                                                                Wstd = Wstd
                                                else: #If it is false, it means that is working as a normal FC layer
                                                        Wstd_layer = 0
                                                        Wstd = Wstd
                                        else:
                                                Wstd = Wstd #If it is false, is a FC layers
                                        if hasattr(layers[i], 'Bstd'): #The same logic is applied for Bstd
                                                if(layers[i].Bstd != 0):
                                                        Bstd_layer = layers[i].Bstd
                                                        if force == "no":
                                                                Bstd = Bstd_layer
                                                        else:
                                                                Bstd = Bstd
                                                else:
                                                        Bstd = Bstd
                                        else:
                                                Bstd = Bstd

                                        #Create the error matrix taking into account the Wstd and Bstd
                                        Werr = Merr_distr(Merr_aux,Wstd,Wstd_layer,errDistr)
                                        Berr = Merr_distr(MBerr_aux,Bstd,Bstd_layer,errDistr)
                                        #Now if the layer have Werr or Berr is an A-Conenct or DropConnect layer
                                        if hasattr(layers[i],'Werr') or hasattr(layers[i],'Berr') or hasattr(layers[i],'infWerr') or hasattr(layers[i],'infBerr'):
                                                #print(i)#

                                                if(layers[i].isQuant[0] == 'yes'): 
                                                        if(Derr != 0): #Introduce the deterministic error when BW are used
                                                                weights = layers[i].weights[0]
                                                                wp = weights > 0
                                                                wn = weights <= 0
                                                                wn = wn.numpy()
                                                                wp = wp.numpy()
                                                                Werr = Derr*wn*Werr + Werr*wp
                                                if hasattr(layers[i], 'Wstd'):
                                                        if(layers[i].Wstd != 0):
                                                                layers[i].infWerr = Werr #Change the inference error matrix
                                                        else:
                                                                #print(layers[i].Werr)
                                                                layers[i].Werr = Werr
                                                else:
                                                                layers[i].Werr = Werr
                                                if hasattr(layers[i], 'Bstd'):
                                                        if(layers[i].Bstd != 0):
                                                                layers[i].infBerr = Berr #Change the inference error matrix
                                                        else:
                                                                layers[i].Berr = Berr
                                                else:
                                                        layers[i].Berr = Berr
                                        #if the layer is not A-Conenct or DropCOnnect the error must be introduced to the weights because it is a normal FC or normal Conv layer
                                        else:
                                                weights = layers[i].weights[0]*Werr #Introduce the mismatch to the weights
                                                bias = layers[i].weights[1]*Berr #Introduce the mismatch to the bias
                                                local_weights = [weights,bias] #Create the tuple of modified values
                                                layers[i].set_weights(local_weights) #Update the values of the weights

                NoisyNet = tf.keras.Sequential(layers)
                return NoisyNet,Wstd,Bstd


        def classify(net,Xtest,Ytest,top5,ev_batch_size=None):
                if top5:
                        _, accuracy, top5acc = net.evaluate(Xtest,Ytest,verbose=0,batch_size=ev_batch_size)
                        return accuracy, top5acc
                else:
                        _,accuracy = net.evaluate(Xtest,Ytest,verbose=0,batch_size=ev_batch_size)
                        return accuracy

        def MCsim(net=net,Xtest=Xtest,Ytest=Ytest,M=M,Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,
                force=force,Derr=Derr,net_name=net_name,custom_objects=custom_objects,dtype=dtype,
                optimizer=optimizer,loss=loss,metrics=metrics,top5=top5):

                acc_noisy = np.zeros((M,1)) #Initilize the variable where im going to save the noisy accuracy
                top5acc_noisy = np.zeros((M,1)) #Initilize the variable where im going to save the noisy accuracy   top5
                local_net = tf.keras.models.load_model(net,custom_objects = custom_objects) #Load the trained model
                local_net.save_weights(filepath=(net_name+'_weights.h5')) #Save the weights. It is used to optimize the script RAM consumption
                #print(local_net.summary()) #Print the network summary
                if top5:
                    print('Simulation Nr.\t | \tWstd\t | \tBstd\t | \tAccuracy | \tTop-5 Accuracy\n')
                    print('---------------------------------------------------------------------------------------')
                else:
                    print('Simulation Nr.\t | \tWstd\t | \tBstd\t | \tAccuracy\n')
                    print('---------------------------------------------------------------------------------------')
        #       global parallel

                for i in range(M): #Iterate over M samples
                        [NetNoisy,Wstdn,Bstdn] = add_Wnoise(local_net,Wstd,Bstd,errDistr,force,Derr,dtype=dtype) #Function that adds the new noisy matrices to the layers
                        NetNoisy.compile(optimizer,loss,metrics,run_eagerly=run_model_eagerly) #Compile the model. It is necessary to use the model.evaluate
                        if top5:
                                acc_noisy[i],top5acc_noisy[i] = classify(NetNoisy, Xtest, Ytest,top5,ev_batch_size=evaluate_batch_size) #Get the accuracy of the network
                                top5acc_noisy[i] = 100*top5acc_noisy[i]
                                acc_noisy[i] = 100*acc_noisy[i]
                                print('\t%i\t | \t%.1f\t | \t%.1f\t | \t%.2f | \t%.2f\n' %(i,Wstd*100,Bstd*100,acc_noisy[i],top5acc_noisy[i]))
                        else:
                                acc_noisy[i] = classify(NetNoisy, Xtest, Ytest,top5,ev_batch_size=evaluate_batch_size) #Get the accuracy of the network
                                acc_noisy[i] = 100*acc_noisy[i]
                                print('\t%i\t | \t%.1f\t | \t%.1f\t | \t%.2f\n' %(i,Wstd*100,Bstd*100,acc_noisy[i]))
                        local_net.load_weights(filepath=(net_name+'_weights.h5')) #Takes the original weights value.
        #               return acc_noisy

                #pool = Pool(mp.cpu_count())
                #acc_noisy = pool.map(parallel, range(M))
                #pool.close()
                media = np.median(acc_noisy)
                Xmin = np.amin(acc_noisy)
                Xmax = np.amax(acc_noisy)
                IQR = np.percentile(acc_noisy,75) - np.percentile(acc_noisy,25)
                stats = [media,IQR,Xmax,Xmin]
                print('---------------------------------------------------------------------------------------')
                print('Median: %.2f%%\n' % media)
                print('IQR Accuracy: %.2f%%\n' % IQR)
                print('Min. Accuracy: %.2f%%\n' % Xmin)
                print('Max. Accuracy: %.2f%%\n'% Xmax)

                os.remove(net_name+'_weights.h5')   #Delete created weight file
                #np.savetxt(net_name+'_simerr_'+str(int(100*Wstd))+'_'+str(int(100*Bstd))+'.txt',acc_noisy,fmt="%.2f") #Save the accuracy of M samples in a txt
                #np.savetxt(net_name+'_stats'+'_simerr_'+str(int(100*Wstd))+'_'+str(int(100*Bstd))+'.txt',stats,fmt="%.2f") #Save the median and iqr of M samples in a txt
                #if top5:
                #        np.savetxt(net_name+'_TOP5'+'_simerr_'+str(int(100*Wstd))+'_'+str(int(100*Bstd))+'.txt',top5acc_noisy,fmt="%.2f") #Save the accuracy of M samples in a txt
                return acc_noisy, stats
        return  MCsim(net=net,Xtest=Xtest,Ytest=Ytest,M=M,Wstd=Wstd,Bstd=Bstd,errDistr=errDistr,
                force=force,Derr=Derr,net_name=net_name,custom_objects=custom_objects,dtype=dtype,
                optimizer=optimizer,loss=loss,metrics=metrics,top5=top5)

#Function to do inference. You also could have the top-5 accuracy if you passed to the model metrics and then setting top5=True
def classify(net,Xtest,Ytest,top5,ev_batch_size=None):
        def classify(net,Xtest,Ytest,top5):
                if top5:
                        _, accuracy, top5acc = net.evaluate(Xtest,Ytest,verbose=0,batch_size=ev_batch_size)
                        return accuracy, top5acc
                else:
                        _,accuracy = net.evaluate(Xtest,Ytest,verbose=0,batch_size=ev_batch_size)
                        return accuracy
        return classify(net,Xtest,Ytest,top5)

#Function to load the MNIST dataset. THis function could load the standard 28x28 8 or 4 bits dataset, or 11x11 8 or 4 bits dataset.
def load_ds(imgSize=[28,28], Quant=8):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        if (imgSize != [28,28]):
                x_train, x_test = np.expand_dims(x_train,-1),np.expand_dims(x_test,-1) #Need an extra dimension to apply tf.image.resize
                x_train = tf.image.resize(x_train,[imgSize[0],imgSize[1]],method=tf.image.ResizeMethod.BILINEAR,antialias=True) #This function applies a resize similar to imresize in matlab
                x_test = tf.image.resize(x_test,[imgSize[0],imgSize[1]],method=tf.image.ResizeMethod.BILINEAR,antialias=True)
                x_train, x_test = np.squeeze(x_train,-1),np.squeeze(x_test,-1) #Remove the extra dimension
        x_train = tf.cast(x_train,tf.uint8)
        x_test = tf.cast(x_test,tf.uint8)
        if(Quant !=8):
                xlsb = 256/2**Quant
                x_train = np.floor(np.divide(x_train,xlsb))
                x_test = np.floor(np.divide(x_test,xlsb))
                x_train = tf.cast(x_train,tf.uint8)
                x_test = tf.cast(x_test,tf.uint8)
        return (x_train,y_train),(x_test,y_test)

#Function to plot the box chart
def plotBox(data,labels,legends,color,color_fill,path):
        import matplotlib.pyplot as plt
        """
        Script to plot and save a box chart with custom style.
        """

        def plotChart(ax,x,color=color,color_fill=color_fill,labels=labels): #Script that plots the box, needed in the script below
                boxprops = dict(linestyle='-', linewidth=3,facecolor=color_fill,color=color)
                whiskerprops = dict(color=color, linewidth=3)
                flierprops = dict(marker='.', markerfacecolor=color, markersize=6,
                              markeredgecolor=color)
                capprops = dict(color=color)
                medianprops = dict(color=color)
                b = ax.boxplot(x,notch=False,widths=0.2,labels=labels,patch_artist=True,
                            boxprops=boxprops,whiskerprops=whiskerprops,flierprops=flierprops,
                            capprops=capprops,medianprops=medianprops
                            )
                return b

        def plotBox(data,labels,legends,color,color_fill,path,figsize=(3,5)): #Top script  to plot the box with custom style
                """
                HOW TO:
                data: Data that you want to plot, should be a list or a list of list (maximum 3 list) i.e. data= [data1,data2,data3] where data1 = [x1,x2,...],
                data2=[y1,y2,...], data3=[z1,z2,...]
                labels: Labels for the x-axis. Must have the same dimension as the data that you are going to plot.
                legends: String. Legends for the plot. Must have the same dimension as the data parameter e.g. data = [data1,data2], legends=[legend1,legend2]
                color: Color for the lines. Should be a list of size 3 with RGB Color.
                color_fill: Color for filling the boxes. Should be a list of size 3 with RGB Color.
                path: String. Where you want to save the image and the name of the archive. By default you do not need to indicate a saving format. By default all
                the images are saved in png format."""
                font = {'family':'Arial','style':'normal','weight' : 'semibold',
                    'size'   : 14}
                plt.rc('font',**font)
                fig = plt.figure(figsize=figsize)
                ax = fig.add_axes([0.1,0.1,0.8,0.8])
                ax.set_xlabel("Simulation Error (%)",fontdict={'family':'Arial','style':'normal','weight' : 'semibold',
                    'size'   : 15})
                ax.set_ylabel("Validation Accuracy (%)",fontdict={'family':'Arial','style':'normal','weight' : 'semibold',
                    'size'   : 16})
                d_size = len(data)
                if d_size == 4:
                    b1 = plotChart(ax,data,color=color,color_fill=color_fill,labels=labels)
                    ax.legend([b1["boxes"][0]],legends, loc='lower left',prop={'family':'Arial','style':'normal','weight' : 'semibold',
                    'size'   : 12})
                elif d_size == 2:
                    b1 = plotChart(ax,data[0],color=color[0],color_fill=color_fill[0],labels=labels)
                    b2 = plotChart(ax,data[1],color=color[1],color_fill=color_fill[1],labels=labels)
                    ax.legend([b1["boxes"][0], b2["boxes"][0]],[legends[0], legends[1]], loc='lower left',prop={'family':'Arial','style':'normal','weight' : 'semibold',
                    'size'   : 12})
                elif d_size == 3:
                    b1 = plotChart(ax,data[0],color=color[0],color_fill=color_fill[0],labels=labels)
                    b2 = plotChart(ax,data[1],color=color[1],color_fill=color_fill[1],labels=labels)
                    b3 = plotChart(ax,data[2],color=color[2],color_fill=color_fill[2],labels=labels)
                    ax.legend([b1["boxes"][0], b2["boxes"][0],b3["boxes"][0]],[legends[0], legends[1], legends[2]], loc='lower left',prop={'family':'Arial','style':'normal','weight' : 'semibold',
                    'size'   : 12})
                else:
                    print("Not supported size")

                ax.spines['top'].set_linewidth(1.4)
                ax.spines['right'].set_linewidth(1.4)
                ax.spines['bottom'].set_linewidth(1.4)
                ax.spines['left'].set_linewidth(1.4)
                plt.savefig(path, bbox_inches='tight')
        return plotBox(data,labels,legends,color,color_fill,path)

def Merr_distr(Merr,stddev,stddev_layer,errDistr): #Used to reshape the output of the layer

    N = stddev*Merr

    if errDistr == "normal":
      Merr = np.abs(1+N)
    elif errDistr == "lognormal":
      Merr = np.exp(-N)*np.exp(0.5*(np.power(stddev_layer,2)-np.power(stddev,2)))
    return Merr
