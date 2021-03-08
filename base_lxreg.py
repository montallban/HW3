'''
Author: Andrew H. Fagg
Modified by: Alan Lee
'''
import numpy as np
from symbiotic_metrics import *
from job_control import *
import pickle
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import time
import re
import sys
from functools import partial


from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.models import Sequential

#################################################################
# Default plotting parameters
# FIGURESIZE=(10,6)
# FONTSIZE=18

# plt.rcParams['figure.figsize'] = FIGURESIZE
# plt.rcParams['font.size'] = FONTSIZE

# plt.rcParams['xtick.labelsize'] = FONTSIZE
# plt.rcParams['ytick.labelsize'] = FONTSIZE

#################################################################
def extract_data(bmi, args):
    '''
    Translate BMI data structure into a data set for training/evaluating a single model
    
    @param bmi Dictionary containing the full BMI data set, as loaded from the pickle file.
    @param args Argparse object, which contains key information, including Nfolds, 
            predict_dim, output_type, rotation
            
    @return Numpy arrays in standard TF format for training set input/output, 
            validation set input/output and testing set input/output; and a
            dictionary containing the lists of folds that have been chosen
    '''
    # Number of folds in the data set
    ins = bmi['MI']
    Nfolds = len(ins)
    
    # Check that argument matches actual number of folds
    assert (Nfolds == args.Nfolds), "Nfolds must match folds in data set"
    
    # Pull out the data to be predicted
    outs = bmi[args.output_type]
    
    # Check that predict_dim is valid
    assert (args.predict_dim is None or (args.predict_dim >= 0 and args.predict_dim < outs[0].shape[1]))
    
    # Rotation and number of folds to use for training
    r = args.rotation
    Ntraining = args.Ntraining
    dropout = args.dropout
    
    # Compute which folds belong in which set
    folds_training = (np.array(range(Ntraining)) + r) % Nfolds
    folds_validation = (np.array([Nfolds-2]) +r ) % Nfolds
    folds_testing = (np.array([Nfolds-1]) + r) % Nfolds
    # Log these choices
    folds = {'folds_training': folds_training, 'folds_validation': folds_validation,
            'folds_testing': folds_testing}
    
    # Combine the folds into training/val/test data sets (pairs of input/output numpy arrays)
    ins_training = np.concatenate(np.take(ins, folds_training))
    outs_training = np.concatenate(np.take(outs, folds_training))
        
    ins_validation = np.concatenate(np.take(ins, folds_validation))
    outs_validation = np.concatenate(np.take(outs, folds_validation))
        
    ins_testing = np.concatenate(np.take(ins, folds_testing))
    outs_testing = np.concatenate(np.take(outs, folds_testing))
    
    # If a particular dimension is specified, then extract it from the outputs
    if args.predict_dim is not None:
        outs_training = outs_training[:,[args.predict_dim]]
        outs_validation = outs_validation[:,[args.predict_dim]]
        outs_testing = outs_testing[:,[args.predict_dim]]
    
    return ins_training, outs_training, ins_validation, outs_validation, ins_testing, outs_testing, folds

def augment_args(args):
    # if you specify exp index, it translates that into argument values that you're overiding
    '''
    Use the jobiterator to override the specified arguments based on the experiment index. 
    @return A string representing the selection of parameters to be used in the file name
    '''
    index = args.exp_index
    if(index == -1):
        return ""
    
    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    # Overides Ntraining and rotation
    if args.dropout != 0:
        p = {'Ntraining': [1,2,3,5,10,18], 
             'rotation': range(20),
             'dropout': [None, 0.1, 0.2, 0.5]}
    elif(args.LxReg)
        if args.l1 != 0:
            p = {'Ntraining': [1,2,3,5,10,18], 
                 'rotation': range(20),
                 'l1': [None, 0.1, 0.2, 0.3, 0.5]}
        elif args.l2 != 0:
            p = {'Ntraining': [1,2,3,5,10,18], 
                 'rotation': range(20),
                 'l2': [None, 0.1, 0.2, 0.3, 0.5]}     

    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())
    
    # Check bounds
    assert (args.exp_index >= 0 and args.exp_index < ji.get_njobs()), "exp_index out of range"

    # Print the parameters specific to this exp_index
    print(ji.get_index(args.exp_index))
    
    # Push the attributes to the args object and return a string that describes these structures
    # destructively modifies the args 
    # string encodes info about the arguments that have been overwritten
    return ji.set_attributes_by_index(args.exp_index, args)
    
def generate_fname(args, params_str):
    '''
    Generate the base file name for output files/directories.
    
    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.
    
    Expand this as needed
    '''
    # Hidden unit configuration
    # number of units in each of the hidden layers
    hidden_str = '_'.join(str(x) for x in args.hidden)
    
    # Dimension being predicted
    # info about what is being predicted
    # dim can be 0 or 1 corresponding to shoulder or elbow
    # if None, predict both of the dimensions
    if args.predict_dim is None:
        predict_str = args.output_type
    else:
        predict_str = '%s_%d'%(args.output_type, args.predict_dim)
        
    # Put it all together, including #of training folds and the experiment rotation
    return "%s/%s_%s_hidden_%s_%s"%(args.results_path, args.exp_type, predict_str, hidden_str, params_str)

def deep_network_basic(in_n, hidden, out_n, hidden_activation, output_activation, lrate,dropout, l1, l2, metrics_):
    model = Sequential();

    # Construct model
    # First, check for Lx regularization
    if(args.LxReg):
        model.add(InputLayer(input_shape=(in_n,)))
        for idx, layer_n in enumerate(hidden):
            title = "hidden" + str(idx)
            # Check for either l1 or l2 regularization
            # And add hidden layers 
            if l1 != 0:
                model.add(Dense(layer_n, use_bias=True, name=title,
                activation=hidden_activation, kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l1(l1)))
            if l2 != 0:
                model.add(Dense(layer_n, use_bias=True, name=title,
                activation=hidden_activation, kernel_initializer="he_normal",
                kernel_regularizer=keras.regularizers.l2(l2)))
            # add output layer
            model.add(Dense(out_n, use_bias=True, name="output", activation=output_activation))    
            else:
                # if LxReg but l1 = 0 and l2 = 0, then make a model without LxReg
                model.add(input_shape=(in_n,))
                # Add dropout in the input layer and in the hidden layers
                for idx, layer_n in enumerate(hidden):
                    title = "hidden" + str(idx)
                    model.add(Dense(layer_n, use_bias=True, name=title, activation=hidden_activation))
                model.add(Dense(out_n, use_bias=True, name="output", activation=output_activation))      
                          
    # If no LxReg, then build model without it, possibly with dropout
    else:
        model.add(Dropout(dropout, input_shape=(in_n,)))
        # Add dropout in the input layer and in the hidden layers
        for idx, layer_n in enumerate(hidden):
            title = "hidden" + str(idx)
            model.add(Dense(layer_n, use_bias=True, name=title, activation=hidden_activation))
            model.add(Dropout(dropout))
        model.add(Dense(out_n, use_bias=True, name="output", activation=output_activation))

    # Optiemizer
    opt = tf.keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    
    # Bind the optimizer and the loss function to the model
    model.compile(loss='mse', optimizer=opt, metrics=metrics_)
    
    # Generate an ASCII representation of the architecture
    print(model.summary())
    return model

########################################################


def execute_exp(args=None):
    '''
    Perform the training and evaluation for a single model
    @args Argparse arguments
    '''
    # Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])
        
    # Modify the args in specific situations
    # Call augment_args
    params_str = augment_args(args)
    
    print("Params:", params_str)
    
    # Compute output file name base
    fbase = generate_fname(args, params_str)
    
    print("File name base:", fbase)
    
    # Is this a test run?
    if(args.nogo):
        # Don't execute the experiment
        print("Test run only")
        return None
    
    # Load the data
    fp = open(args.dataset, "rb")
    bmi = pickle.load(fp)
    fp.close()
    
    # Extract the data sets.  This process uses rotation and Ntraining (among other exp args)
    ins, outs, ins_validation, outs_validation, ins_testing, outs_testing, folds = extract_data(bmi, args)
    
    # Metrics
    # fvaf at 1 = perfect prediction, at 0 = no predictionz
    fvaf = FractionOfVarianceAccountedFor(outs.shape[1])
    rmse = tf.keras.metrics.RootMeanSquaredError()

    # Build the model: you are responsible for providing this function
    print(ins.shape[1])
    model = deep_network_basic(ins.shape[1], tuple(args.hidden), outs.shape[1],     # Size of inputs, hidden layer(s) and outputs
                               hidden_activation='elu',
                              output_activation=None,
                              lrate=args.lrate,
                              dropout=args.dropout,
                              l1=args.l1, 
                              l2=args.l2,
                              metrics_=[fvaf, rmse])
    
    # Report if verbosity is turned on
    if args.verbose >= 1:
        print(model.summary())
    
    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                      restore_best_weights=True,
                                                      min_delta=args.min_delta)

    checkpoint_cb = keras.callbacks.ModelCheckpoint("", save_best_only=True)

    
    # Learn
    history = model.fit(x=ins, y=outs, epochs=args.epochs, verbose=args.verbose>=2,
                        validation_data=(ins_validation, outs_validation), 
                        callbacks=[early_stopping_cb])
        
    # Generate log data
    results = {}
    results['args'] = args
    results['predict_training'] = model.predict(ins)
    results['predict_training_eval'] = model.evaluate(ins, outs)
    results['predict_validation'] = model.predict(ins_validation)
    results['predict_validation_eval'] = model.evaluate(ins_validation, outs_validation)
    results['predict_testing'] = model.predict(ins_testing)
    results['predict_testing_eval'] = model.evaluate(ins_testing, outs_testing)
    results['folds'] = folds
    results['history'] = history.history
    
    # Save results
    results['fname_base'] = fbase
    fp = open("%s_results.pkl"%(fbase), "wb")
    pickle.dump(results, fp)
    fp.close()
    
    # Save the model
    model.save("%s_model"%(fbase))
    return model
               
def create_parser():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='BMI Learner')
    parser.add_argument('-rotation', type=int, default=1, help='Cross-validation rotation')
    parser.add_argument('-epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('-dataset', type=str, default='home/mcmontalbano/datasets/bmi_dataset.pkl', help='Data set file')
    parser.add_argument('-Ntraining', type=int, default=1, help='Number of training folds')
    parser.add_argument('-output_type', type=str, default='theta', help='Type to predict')
    parser.add_argument('-exp_index', type=int, default=1, help='Experiment index')
    parser.add_argument('-Nfolds', type=int, default=20, help='Maximum number of folds')
    parser.add_argument('-results_path', type=str, default='results2', help='Results directory')
    parser.add_argument('-hidden', nargs='+', type=int, default=[1000, 100, 10 ,100, 1000], help='Number of hidden units per layer (sequence of ints)')
    parser.add_argument('-lrate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('-min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('-patience', type=int, default=100, help="Patience for early termination")
    parser.add_argument('-verbose', '-v', action='count', default=0, help="Verbosity level")
    parser.add_argument('-predict_dim', type=int, default=0, help="Dimension of the output to predict")
    parser.add_argument('-nogo', action='store_false', help='Do not perform the experiment')
    parser.add_argument('-exp_type', type=str, default='bmi', help='High level name for this set of experiments')
    parser.add_argument('-dropout', type=float, default=0, help='Enter the dropout rate.' )
    parser.add_argument('-LxReg', action="store_false", help='Enter l1, l2, or none.')    
    parser.add_argument('-l1', type=float, default=0, help='Enter value for l1 in ridge regression.')    
    parser.add_argument('-l2', type=float, default=0, help='Enter value for l2 in ridge regression.')    

    return parser

def check_args(args):
    '''
    Check that key arguments are within appropriate bounds.  Failing an assert causes a hard failure with meaningful output
    '''
    assert (args.rotation >= 0 and args.rotation < args.Nfolds), "Rotation must be between 0 and Nfolds"
    assert (args.Ntraining >= 1 and args.Ntraining <= (args.Nfolds-2)), "Ntraining must be between 1 and Nfolds-2"
    assert (args.lrate > 0.0 and args.lrate < 1), "Lrate must be between 0 and 1"
    
#################################################################
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)
    execute_exp(args)
