# scriptsn: scripts new
# There should only be one scripts.py, however the new MonteCarlo function works differently as it expects a model as input instead of a path to load the model from. scripts.py is kept for backwards compatibility, this MonteCarlo function should work for most layers including all the AConnect layers.

import numpy as np
import tensorflow as tf
import os

from aconnect.layers import Merr_distr

def Merr_distr_s(shape, stddev, stddev_layer, errDistr):
    
    N = np.random.randn(*shape)*stddev

    if errDistr == "normal":
        Merr = np.abs(1+N)
    elif errDistr == "lognormal":
        Merr = np.exp(-N)*np.exp(0.5*(np.power(stddev_layer,2)-np.power(stddev,2)))
    return Merr


# Get all layers of the model
def getModelLayers(model):
    def isLayer(x):
        return(isinstance(x, tf.keras.layers.Layer) and not isinstance(x, tf.keras.Model))

    modelLayers = [*model.layers]
    allLayers = False
    toRemove = []
    while not allLayers:
        allLayers = True
        for l in toRemove:
            modelLayers.remove(l)
        toRemove = []
        for l in modelLayers:
            if isinstance(l, tf.keras.layers.Bidirectional):
                allLayers = False
                modelLayers.extend([l.forward_layer, l.backward_layer])
                toRemove.append(l)
                
            if not isLayer(l):
                allLayers = False
                toRemove.append(l)
                modelLayers.extend(l.layers)
                
    return(modelLayers)

# Takes a list of layers and replaces the RNN layers with their cell/cells
def getLayerCells(modelLayers):
    
    allCells = False
    toRemove = []
    while not allCells:
        allCells = True
        for l in toRemove:
            modelLayers.remove(l)
        toRemove = []
        for l in modelLayers:
            if hasattr(l, "cell"):
                allCells = False
                toRemove.append(l)
                modelLayers.extend([l.cell])
                
            if hasattr(l, "cells"):
                allCells = False
                toRemove.append(l)
                modelLayers.extend(l.cells)
                
    return(modelLayers)

## Code to add noise to layers

# matrixParams contains information for each error matrix that can be present in an AConnect layer
# The first two values correspond to error matrix names and corresponding weights/bias matrix names, i.e. for ["infWerr", "kernel", *, *], it means that the 'kernel' matrix gets multiplied with the 'infWerr' matrix
#   The corresponding weights/bias matrix names are used to determine the shape and dtype of the error matrix
# The third value is used to determine if the matrix corresponds to a weight (True) or a bias (False)
# The fourth value is used to determine if deterministic error (Derr) will be applied (True) or not (False)
# [errMatrix, matrix, isWeight, addDerr]
matrixParams = [["infWerr", "kernel", True, True],
                ["infRWerr", "recurrent_kernel", True, True],
                ["infZerr", "zeta", True, False],
                ["infNUerr", "nu", True, False],

                ["infBerr", "bias", False, False],
                ["infBHerr", "bias_h", False, False],
                ["infBZerr", "bias_z", False, False],

                ["infWerr", "W", True, True],

                ["Werr", "kernel", True, True],
                ["RWerr", "recurrent_kernel", True, True],
                ["Zerr", "zeta", True, False],
                ["NUerr", "nu", True, False],

                ["Werr", "W", True, True],

                ["Berr", "bias", False, False],
                ["BHerr", "bias_h", False, False],
                ["BZerr", "bias_z", False, False]]


def changeInfMatrixToTfVariable(net): # Changes all the err/inferr matrices of the AConnect layers from a tf.Tensor/np.array to a tf.Variable, so that their values can be changed without having to recompile the model
    layers = getModelLayers(net)
    layers = getLayerCells(layers) # Replaces RNN layers with their cells

    for l in layers:
        if l.count_params() != 0:
            if (hasattr(l, 'Wstd') and hasattr(l, 'Bstd')) or (hasattr(l, 'cell') and (hasattr(l.cell, 'Wstd') and hasattr(l.cell, 'Bstd'))): # Layer is AConnect
 
                if hasattr(l, 'cell'): # RNN/LSTM layer
                    obj = l.cell
                    
                else: # FC/CNN layer
                    obj = l
                
                for params in matrixParams:
                    names = params[:2] # params[:2] are the names of the error matrix and corresponding weights/bias matrix
                    if all([hasattr(obj, x) for x in names]): # Check that both attributes are present
                        errMatrix, matrix = names
                        matrixObj = getattr(obj, matrix)
                        setattr(obj, errMatrix, tf.Variable(tf.ones(matrixObj.shape), dtype=matrixObj.dtype))

    net._reset_compile_cache()

def addWNoise(net, Wstd, Bstd, errDistr, Derr, useTfVariable=False):

    # Function to add deterministc error
    def addDerr(Merr, weights, isBin):
        if (isBin==True or isBin=='yes') and Derr != 0:
            wp = weights > 0
            wn = weights <= 0 
            wn = wn.numpy()
            wp = wp.numpy()

            Merr = Derr*wn*Merr + Merr*wp
        
        return Merr

    # The function to update the error matrices depends on the type of the matrices
    def updateNumpy(obj, attr, value):
        setattr(obj, attr, value)

    def updateTfVariable(obj, attr, value):
        getattr(obj, attr).assign(value)
    
    if useTfVariable:
        update = updateTfVariable
    else:
        update = updateNumpy

    layers = getModelLayers(net)
    layers = getLayerCells(layers) # Replaces RNN layers with their cells

    for l in layers:
        # Don't apply noise to normalization layers
        toIgnore = [tf.keras.layers.Normalization, tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization]
        if any([isinstance(l, x) for x in toIgnore]):
            continue

        if l.count_params() != 0:
            if (hasattr(l, 'Wstd') and hasattr(l, 'Bstd')) or (hasattr(l, 'cell') and (hasattr(l.cell, 'Wstd') and hasattr(l.cell, 'Bstd'))): # Layer is AConnect
               
                if hasattr(l, 'cell'): # RNN/LSTM layer
                    obj = l.cell

                else: # FC/CNN layer
                    obj = l

                for params in matrixParams:
                    names = params[:2] # params[:2] are the names of the error matrix and corresponding weights/bias matrix
                    isWeight = params[2]
                    toAddDerr = params[3]

                    if all([hasattr(obj, x) for x in names]): # Check that both attributes are present
                        errMatrix, matrix = names
                        matrixObj = getattr(obj, matrix)

                        if isWeight:
                            Xstd = Wstd # Error being applied
                            Ostd = obj.Wstd # Error used to train the layer/cell
                        else:
                            Xstd = Bstd
                            Ostd = obj.Bstd
                        
                        newErrMatrix = Merr_distr_s(matrixObj.shape, Xstd, Ostd, errDistr)

                        if toAddDerr:
                            newErrMatrix = addDerr(newErrMatrix, matrixObj, obj.isBin)
 
                        update(obj, errMatrix, newErrMatrix)

            else: # Base layers (Not AConnect)
                for w in l.weights:

                    if w.dtype.is_integer: # Ignore weights with integer values (i.e. weight 'count' in 'Normalization' layer) # Note we don't apply noise to 'Normalization' layer right now, but we might do it in the future
                        continue

                    if "bias" in w.name:
                        if Bstd != 0:
                            w.assign(w * Merr_distr(w.shape, Bstd, dtype=w.dtype, errDistr=errDistr))
                    else:
                        if Wstd != 0:
                            w.assign(w * Merr_distr(w.shape, Wstd, dtype=w.dtype, errDistr=errDistr))

    if not useTfVariable:
        net._reset_compile_cache()

def evaluate_net(net, xtest, ytest, batch_size): # Function that evaluates the model and returns a dictionary of every metric and its value
    return net.evaluate(xtest, ytest, batch_size=batch_size, return_dict=True, verbose=False)

# TODO: Fix memory problems, the RAM gets filled up and causes the runtime to end
#           Tried: clear_session and gc.collect, didn't work
#           Workaround: Use tf.Variable instead tf.Tensor in the inference matrices, so recompiling the model is not needed for every sample
# TODO?: Add option to choose the function 'evaluate_net', so that we may use models that aren't evaluated with 'evaluate' method

def MonteCarlo(net, x_test, y_test, M=1, Wstd=0, Bstd=0, metrics=None, errDistr="normal", Derr=0, save_path=None, batch_size=None, runEagerly=None, useTfVariable=False):
    """
    Perform MonteCarlo simulation on the model specified by 'net'

    net: model
    x_test, y_test: Data for testing/evaluating the model
    M: Number of samples (integer)
    Wstd, Bstd: Standard deviation of the error to apply to the weights and biases
    metrics: List of strings defining which metrics to take into account, if it is None then the metrics will be those returned when evaluating the model
    errDistr: Either "normal" or "lognormal"
    Derr: Introduce a deterministic error when using binary weights in the network. Float in range [0,1]
    save_path: Path of a folder to save the results, generally the folder should be the name of the model
    batch_size: Batch_size to use when evaluating the model (higher values means faster evaluation, but higher memmory usage)
    runEagerly: Whether to run the model eagerly at evaluation (can increase performance in certain cases, for example when using a batch_size the same size of the dataset) (Setting it to None keeps the default)
    useTfVariable: Set to True to change all the inferece matrices of the layers from tf.Tensor to tf.Variable before running the montecarlo
                   this prevents recompiling the model in every sample, preventing the increase of memory usage
    """

    if metrics is None:
        metrics = net.metrics_names
    
    results = {m:np.zeros(M) for m in metrics}

    if useTfVariable: # Using tf.Variable instead of tf.Tensor in the inferences matrices allows us to change the matrix values without recompiling the model every time
        changeInfMatrixToTfVariable(net)

    local_net_weights = net.get_weights() # Save the original weights of the model so that they can be restored later

    header = 'Simulation Nr.\t | \tWstd %\t | \tBstd %\t'
    for metric in metrics:
        header = header + (" | \t%s\t" % metric)
    print(header)
    print('---------------------------------------------------------------------------------------')

    for i in range(M):
        addWNoise(net, Wstd, Bstd, errDistr, Derr, useTfVariable) # Add noise to model

        if runEagerly is not None:
            net.run_eagerly = runEagerly
       
        temp_results = evaluate_net(net, x_test, y_test, batch_size)

        out = '\t%i\t | \t%.1f\t | \t%.1f\t' % (i, Wstd*100, Bstd*100)

        # Save and print the result of every metric
        for metric in results.keys():
            if metric == "accuracy":
                res = 100*temp_results[metric] # Multiply the accuracy by 100 to get percentage
            else:
                res = temp_results[metric] # Other metrics (such as loss) are unchanged
            
            results[metric][i] = res
            out = out + (" | \t%.2f\t" % res)
        print(out)

        net.set_weights(local_net_weights) # Reset the weights to the original values (before adding the noise)

    # Calculate statistics
    median = {m:np.median(results[m]) for m in metrics}
    IQR = {m:(np.percentile(results[m], 75) - np.percentile(results[m], 25)) for m in metrics} 
    stats = [median, IQR]

    min = {m:np.amin(results[m]) for m in metrics}
    max = {m:np.amax(results[m]) for m in metrics}

    # Save data
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

        Wstdp = int(Wstd*100)
        Bstdp = int(Bstd*100)

        for metric in metrics:
            with open(os.path.join(save_path, "{0}_{1}_{2}.txt".format(metric, Wstdp, Bstdp)), "w") as output:
                for i in results[metric]:
                    output.write(str(i) + "\n")

        with open(os.path.join(save_path, "stats_{0}_{1}.txt".format(Wstdp, Bstdp)), "w") as output:
            #output.write(str(stats))
            for metric in metrics:
                output.write("%s: %2.2f/%2.2f\n" % (metric, median[metric], IQR[metric]))

    print('---------------------------------------------------------------------------------------')
    print('Median: %s' % median)
    print('IQR: %s' % IQR)
    print('Min.: %s' % min)
    print('Max.: %s' % max)
