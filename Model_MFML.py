#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:37:54 2023

@author: vvinod
"""
import numpy as np
from qml.math import cho_solve
from qml import kernels
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
import time
#main module
class ModelMFML:
    '''
    Class to perform model difference MFML.
    '''
    def __init__(self, reg:float=1e-9, kernel:str='matern', sigma:float=715.0,
                 order:int=1, metric:str='l2', gammas:np.ndarray=None, 
                 p_bar:bool=False):
        '''
        Initiation function for the ModelMFML class

        Parameters
        ----------
        reg : float, optional
            The Lavrentiev regularization parameter for KRR. 
            The default is 1e-9.
        kernel : str, optional
            Kernel type to be used for KRR. The default is 'matern'.
        sigma : float, optional
            Kernel width for KRR kernel. Ignored if linear kernel is chosen. 
            The default is 715.0.
        order : int, optional
            Order of the Matern kernel.  Ignored by other kernel types. 
            The default is 1.
        metric : str, optional
            Metric of the Matern kernel. Ignored by other kernel types. 
            The default is 'l2'.
        gammas : np.ndarray, optional
            The individual gammas for the Sargan kernel. Ignored by the rest. 
            The default is None.
        p_bar : bool, optional
            Enables or disables (viewing) tqdm progress bar. Default is False.

        Returns
        -------
        None.

        '''
        #init params
        self.reg = reg
        self.kernel = kernel
        self.sigma = sigma
        self.order = order
        self.metric = metric
        self.gammas = gammas
        #train params
        self.X_train_parent = None
        self.X_trains = None
        self.y_trains = None
        self.indexes = None
        self.fidelities = None
        #model params
        self.models = None
        self.coeffs = None
        self.intercept = None
        self.LCCoptimizer = None
        #score params
        self.mae = 0.0
        self.rmse = 0.0
        #time params
        self.train_time = 0.0
        self.predict_time = 0.0
        
        self.p_bar = p_bar
        
    def property_differences(fidelities=None):
        '''
        Function to generate property differences between fidelities.
        The first object in the diff_array is the baseline fidelity. This
        is not a difference between fidelities. The subsequent objects
        are differences between corresponding fidelities.

        Parameters
        ----------
        fidelities : np.ndarray(str)
            Array of fidelities of properties to be used in the MFML.

        Returns
        -------
        diff_array : np.ndarray(object)
            Array of differences of property of interest.
        index_array : np.ndarray(object)
            Array of indexes corresponding to the properties. 
            This indicates where the properties are with 
            respect to the features.
        '''
        #if user specifies fidelities
        #if not isinstance(fidelities, type(None)):
        #    self.fidelities = np.copy(fidelities)
        #instatiate variables to store diff and indexes
        energy_array = np.zeros((len(fidelities)), dtype=object)
        index_array = np.zeros((len(fidelities)), dtype=object)
        #load lowest fidelity file
        #currently hard-coded to a directory. Should be flexible.
        E0 = np.loadtxt('Data/energies/'+fidelities[0])
        energy_array[0] = E0[:,1]
        #save index_0
        index_array[0] =  np.asarray([np.arange(0,energy_array[0].shape[0]),
                                      np.arange(0,energy_array[0].shape[0])]).T
        #run energy difference loop
        for i in tqdm(range(0,len(fidelities)-1),
                      desc='Generating energy differences and indexing...',
                      leave=False):
            E_diff = []
            index = []
            #load fidelities i and i+1
            Ei = np.loadtxt('Data/energies/'+fidelities[i])
            Eip1 = np.loadtxt('Data/energies/'+fidelities[i+1])
            #loop to check location with respect to baseline fidelity
            #this is currently also assumed to be the feature index.
            #quadratic issue here - FIX LATER
            for j in tqdm(range(len(Ei)),leave=False):
                for k in tqdm(range(len(Eip1)),leave=False):
                    if Ei[j,0]==Eip1[k,0]:
                        #where time stamp of Ei is the same as 
                        #time stamp of E0
                        index.append([np.where(E0[:,0]==Eip1[k,0])[0][0], k])
            #save as np arrays of specific types.
            index = np.asarray(index,dtype=int)
            #E_diff = np.asarray(E_diff,dtype=float)
            index_array[i+1] = np.copy(index)
            energy_array[i+1] = np.copy(Eip1[:,1])
        #return array of differences and corresponding indexes
        return energy_array, index_array
    
    def shuffle_indexes(self, n_trains=None, seed=0):
        '''
        Function to shuffle the indexes for MFML training. 
        This method of shuffling ensures that the nested 
        structure of the training data is retained.

        Parameters
        ----------
        n_trains : np.ndarray(int)
            Array of training sizes across fidelities.
        seed : int
            Seed to be used during shuffling using numpy.

        Returns
        -------
        shuffled_index_array : np.ndarray(object)
            Array of shuffled indexes with sizes corresponding to n_trains.

        '''
        #if no n_trains specified, use the entire training data
        if  isinstance(n_trains, type(None)):
            n_trains = np.asarray([self.y_trains[i].shape[0] 
                                   for i in range(self.y_trains.shape[0])])
            print("Training sizes not provided.", 
                  f"\nManually setting training sizes as: {n_trains}")
        #instantiate variable to store shuffled indexes
        shuffled_index_array = np.zeros((n_trains.shape[0]), dtype=object)
        #shuffle index_0
        ind_i = self.indexes[int(n_trains.shape[0]-1)]
        #####shuffle index_i#########
        np.random.seed(seed)
        np.random.shuffle(ind_i)
        ind_tilda_i = ind_i[0:n_trains[-1],:]
        shuffled_index_array[int(n_trains.shape[0]-1)] = ind_tilda_i
        #shuffle subsequent fidelities
        for i in range(n_trains.shape[0]-2,-1,-1):
            ind_im1 = self.indexes[i]
            #set difference between im1 and i
            index_common = []
            for j in range(ind_im1.shape[0]):
                if (ind_im1[j,0] in ind_tilda_i[:,0]):
                    index_common.append(j)
            ind_im1_temp = np.delete(ind_im1,index_common,axis=0)
            #shuffle ind_im1_temp
            np.random.seed(seed)
            np.random.shuffle(ind_im1_temp)
            #append shuffled ind_i to this
            ind_tilda_i = np.concatenate((ind_im1[index_common,:],
                                          ind_im1_temp[0:n_trains[i]-
                                                       len(ind_tilda_i),:]))
            shuffled_index_array[i] = ind_tilda_i
            #prepare for next run of the loop
            ind_i = ind_im1
        #return the shuffled indexes
        return shuffled_index_array
    
    def y_train_breakup(self):
        n=self.indexes.shape[0]
        y_trains = np.zeros((2*n-1),dtype=object)
        count = 0
        #upper triangle
        for i in tqdm(range(n), desc='Extracting upper y_trains', 
                      leave=self.p_bar):
            ind_i = self.indexes[i][:,1]
            y_trains[count] = np.copy(self.y_trains[i][ind_i])
            count += 1
        #lower triangle
        for i in tqdm(range(n-1), desc='Exctracting lower y_trains', leave=self.p_bar):
            ind_i = self.indexes[i]
            ind_ip1 = self.indexes[i+1]
            #find correct indexes
            c_i = []
            for row in ind_ip1:
                temp_i = np.where(ind_i[:,0] == row[0])[0]
                if np.size(temp_i) != 0:
                    c_i.append(ind_i[temp_i[0], 1])
            c_i = np.asarray(c_i)
            y_trains[count] = np.copy(self.y_trains[i][c_i])
            count += 1
        
        self.y_trains = y_trains
    
    def X_train_breakup(self):
        '''
        Function to break up X_train_parent into different X_trains 
        for each fidelity

        Returns
        -------
        None. Stored in self.X_trains

        '''
        n=self.indexes.shape[0]
        X_trains = np.zeros((n),dtype=object)
        
        for i in tqdm(range(n),desc='Extracting X_trains', leave=self.p_bar):
            ind_i = self.indexes[i][:,0]
            X_trains[i] = self.X_train_parent[ind_i]
        self.X_trains = np.copy(X_trains)
        
    def kernel_generators(self, X1:np.ndarray, X2:np.ndarray = None):
        '''
        Function to return various kernels to be used in the MFML if the 
        representations are not FCHL. 

        Parameters
        ----------
        X1 : np.ndarray
            Array of representations. Usually refers to the training 
            representations. 
        X2 : np.ndarray, optional
            Array of representations. Refers to the test representations. 
            If not specified, then it is considered to be same as X1.
            The default is None.

        Returns
        -------
        K : np.ndarray
            Kernel matrix of representations.

        '''
        #Case for training kernel
        if isinstance(X2,type(None)):
            X2=np.copy(X1) #make X2 a copy of X1 if X2 is not specified
        #generating kernels
        if self.kernel=='sargan':
            assert self.gammas is not None, 'sargan kernels require additional parameter of gammas. See qml.kernel.sargan_kernel for more details. Terminated.'
            K = kernels.sargan_kernel(X1, X2, self.sigma, self.gammas)
        elif self.kernel=='gaussian':
            K = kernels.gaussian_kernel(X1, X2, self.sigma)
        elif self.kernel=='laplacian':
            K = kernels.laplacian_kernel(X1, X2, self.sigma)
        elif self.kernel=='linear':
            K = kernels.linear_kernel(X1, X2)
        elif self.kernel=='matern':
            K = kernels.matern_kernel(X1, X2, sigma=self.sigma, 
                                      order=self.order, 
                                      metric=self.metric)
        else:
            K = None
        return K
    
    def KRR(self, K:np.ndarray, y:np.ndarray):
        '''
        Function to perform KRR given a generic kernel and corresponding 
        learning feature.
        
        Parameters
        ----------
        K : np.ndarray (n_mols x n_mols)
            Kernel matrix generated from the representations.
        y : np.ndarray (n_mols x 1)
            Reference property array.
        
        Returns
        -------
        alpha : np.ndarray
            Array of KRR coefficients.
        
        '''
        #copy to prevent overwriting in class atributes
        Ktemp = np.copy(K)
        #regularization
        Ktemp[np.diag_indices_from(Ktemp)] += self.reg #regularisation
        alpha = cho_solve(Ktemp,y) #perform KRR
        #return the coeff for this submodel
        return alpha
    
    def train(self, X_train_parent:np.ndarray, fidelities:np.ndarray=None, 
              y_trains:np.ndarray=None, indexes:np.ndarray=None, 
              shuffle:bool=False, n_trains:np.ndarray=None, seed:int=0):
        '''
        Function to train the MFML model with model difference instead of property difference. 
        It can be shown that the two approaches are equivalent.

        Parameters
        ----------
        X_train_parent : np.ndarray
            The feature set for the lowest fidelity. Nestedness of subsequent fidelities is assumed.
        fidelities : np.ndarray, optional
            The fidelity list from which to load fidelity properties. 
            Ignored if y_trains and indexes are specified. 
            The default is None.
        y_trains : np.ndarray, optional
            The training properties. Contains the different fidelity properties as objects. 
            The default is None.
        indexes : np.ndarray, optional
            Indexes of the features and the properties for MFML. 
            The default is None.
        shuffle : bool, optional
            Whether to shuffle the training samples. Shuffling in MFML retains the nested structure. 
            The default is False.
        n_trains : np.ndarray, optional
            The number of training samples to be used at each fidelity. 
            Ignored if shuffle is False; all training samples are used as is.
            The default is None.
        seed : int, optional
            random seed for shuffling. Ignored if Shuffle set to False. 
            The default is 0.

        Returns
        -------
        None.
        Models and train time are saved to the class attributes.

        '''
        #measure time of process
        tstart = time.time()
       
        #save X_train into the class attribure, will be used in predictions
        self.X_train_parent = np.copy(X_train_parent)
        
        if isinstance(y_trains, type(None)) and isinstance(indexes, type(None)):
            print('No indexes or training properties specified. Checking for class referenced properties and indexes.')
            if isinstance(self.y_trains,type(None)) and isinstance(self.indexes,type(None)):
                assert not isinstance(fidelities,type(None)) or not isinstance(self.fidelities,type(None)), 'Please specify either fidelities or the training properties and indexes.'
                if isinstance(self.fidelities, type(None)):
                    self.fidelities = fidelities
                self.y_trains, self.indexes = self.property_differences()
        else:
            self.y_trains = y_trains
            self.indexes = indexes
        
        
        nfids = self.indexes.shape[0]
        #shuffling indexes
        if shuffle:
            for i in range(nfids):
                self.indexes = self.shuffle_indexes(n_trains=n_trains, seed=seed)
        
        #get the different X_trains
        self.X_train_breakup() #saves in self.X_trains
        self.y_train_breakup() #saves in self.y_trains
        #initiate arrays to store values
        alpha_array = np.zeros((2*nfids-1), dtype=object)
        count = 0
        #upper training
        for i in tqdm(range(nfids),desc='Training upper ML models...', 
                      leave=self.p_bar):
            K_train_upper_i = self.kernel_generators(X1 = self.X_trains[i], 
                                                     X2 = self.X_trains[i])
            alpha_array[count] = self.KRR(K = K_train_upper_i, 
                                          y = self.y_trains[count])
            count += 1
        #lower training
        for i in tqdm(range(nfids-1),desc='Training lower ML models', 
                      leave=self.p_bar):
            K_train_i = self.kernel_generators(X1 = self.X_trains[i+1], 
                                               X2 = self.X_trains[i+1])
            alpha_array[count] = self.KRR(K = K_train_i, 
                                          y = self.y_trains[count])
            count += 1
        
        
        tend = time.time()
        self.train_time = tend-tstart
        self.models = np.copy(alpha_array)
        
    def predict(self, X_test:np.ndarray, X_val:np.ndarray=None,
                y_test:np.ndarray=None, y_val:np.ndarray=None, 
                optimiser:str='default', **optargs):
        '''
        Function to predict using the MFML model. User can choose between default coefficients of +-1
        or solve for coefficients using linear regression on a validation set.

        Parameters
        ----------
        X_test : np.ndarray
            Features of the test set to be evaluated.
        X_val : np.ndarray, optional
            Features of the validation set. Ignored if default optimiser is used. 
            The default is None.
        y_test : np.ndarray, optional
            Property of the test set. Only used to calculate scores. 
            The default is None.
        y_val : np.ndarray, optional
            Properties of the validation set. Ignored if default optimiser is used. 
            The default is None.
        optimiser : str, optional
            The type of optimiser to be used. Currently llows for 
            'OLS' - ordinary least squares, 
            'LRR' - Linear Ridge Regression, 
            and 'default' which is based on SGCT. 
            The default is 'default'.
        **optargs : dict
            Arguements to pass to the optimization method. For KRR, it takes 
            'kernel_type', 'sigma','reg'. 
            For MLPR the kwargs correspond to the values as described in 
            the sklearn package.

        Returns
        -------
        final_preds : np.ndarray
            Predictions on the test set.

        '''
        tstart = time.time()
        nfids = self.indexes.shape[0]
        
        #final preds
        test_preds = np.zeros((y_test.shape[0], 2*nfids-1), dtype=float)
        if not isinstance(y_val,type(None)):
            val_preds = np.zeros((y_val.shape[0], 2*nfids-1), dtype=float)
        
        #for prediction
        count = 0
        #upper triangle preds
        for i in tqdm(range(nfids), desc='Upper MFML predictions', leave=self.p_bar):
            if not isinstance(y_val,type(None)):
                K_val_i = self.kernel_generators(X1 = self.X_trains[i],
                                                 X2 = X_val)
                val_preds[:,count] = np.dot(self.models[count],K_val_i)
            K_test_i = self.kernel_generators(X1 = self.X_trains[i],
                                              X2 = X_test)
            test_preds[:,count] = np.dot(self.models[count], K_test_i)
            count += 1
        #lower triangle preds
        for i in tqdm(range(nfids-1), desc='Lower MFML predictions', 
                      leave=self.p_bar):
            K_test_i = self.kernel_generators(X1 = self.X_trains[i+1],
                                              X2 = X_test)
            if not isinstance(y_val,type(None)):
                K_val_i = self.kernel_generators(X1 = self.X_trains[i+1],
                                                 X2 = X_val)
                val_preds[:,count] = np.dot(self.models[count], K_val_i)
            
            test_preds[:,count] = np.dot(self.models[count], K_test_i)
            count += 1
        
        #ordinary least squares
        if optimiser=='OLS':
            assert not isinstance(y_val,type(None)), "Validation set must be provided for OLS optimization"
            #print(optargs)
            defaultKwargs = { 'copy_X': True, 'fit_intercept': False }
            defaultKwargs.update(**optargs)
            regressor = LinearRegression(**defaultKwargs)
            regressor.fit(val_preds, y_val)
            final_preds = regressor.predict(test_preds)
            self.LCCoptimizer = regressor
            #self.coeffs = regressor.coef_
            #self.intercept = regressor.intercept_
        
        #linear ridge regression
        elif optimiser=='LRR':
            assert not isinstance(y_val,type(None)), "Validation set must be provided for LRR optimization"
            defaultKwargs = {'alpha':1e-9, 'fit_intercept':False, 
                             'copy_X':True, 'max_iter':None, 
                             'tol':1e-4, 'solver':'svd', 'random_state':0}
            
            defaultKwargs.update(**optargs)#{**defaultKwargs,**optargs}
            regressor = Ridge(**defaultKwargs)
            regressor.fit(val_preds,y_val)
            final_preds = regressor.predict(test_preds)
            self.LCCoptimizer = regressor
            #self.coeffs = regressor.coef_
            #self.intercept = regressor.intercept_
        
        #LASSO - linear
        elif optimiser == 'LASSO':
            assert not isinstance(y_val,type(None)), "Validation set must be provided for OLS optimization"
            #print(optargs)
            defaultKwargs = {'alpha':1.0, 'fit_intercept':False, 
                             'precompute':False, 'copy_X':True, 
                             'max_iter':1000, 'tol':1e-4, 
                             'warm_start':False, 'positive':False, 
                             'random_state':0, 
                             'selection':'cyclic'}
            defaultKwargs.update(**optargs)
            regressor = Lasso(**defaultKwargs)
            regressor.fit(val_preds,y_val)
            final_preds = regressor.predict(test_preds)
            self.LCCoptimizer = regressor
        
        
        #Multi-layer perceptron - non-linear
        elif optimiser=='MLPR':
            defaultKwargs = {'hidden_layer_sizes':(100,), 'activation':'relu',
                             'solver':'adam', 'alpha':0.0001, 
                             'batch_size':'auto', 'learning_rate':'constant', 
                             'learning_rate_init':0.001, 'power_t':0.5, 
                             'max_iter':200, 'shuffle':True, 
                             'random_state':None, 'tol':0.0001, 
                             'verbose':False, 'warm_start':False, 
                             'momentum':0.9, 'nesterovs_momentum':True,
                             'early_stopping':False, 'validation_fraction':0.1, 
                             'beta_1':0.9, 'beta_2':0.999, 'epsilon':1e-08,
                             'n_iter_no_change':10, 'max_fun':15000}
            defaultKwargs.update(**optargs) #{**defaultKwargs, **optargs}
            MLPR = MLPRegressor(**defaultKwargs)
            MLPR.fit(val_preds, y_val)
            final_preds = MLPR.predict(test_preds)
            self.LCCoptimizer = MLPR
            #self.coeffs = np.asarray(MLPR.coefs_,dtype=object)
        
        #KRR optimizer - non-linear
        elif optimiser=='KRR':
            defaultKwargs = {'sigma':700.0,'reg':1e-9, 
                             'kernel_type':'gaussian', 'order':1, 
                             'metric':'l2'}
            defaultKwargs.update(**optargs) #{**defaultKwargs, **optargs}

            K_val = kernels.gaussian_kernel(val_preds, val_preds, 
                                            sigma=defaultKwargs['sigma'])
            K_val[np.diag_indices_from(K_val)]+=defaultKwargs['reg']
            opt_alpha = cho_solve(K_val, y_val)

            K_eval = kernels.gaussian_kernel(val_preds, test_preds, 
                                             sigma=defaultKwargs['sigma'])
            final_preds = np.dot(opt_alpha,K_eval)
            self.coeffs = opt_alpha
        
        #default +-1 across fidelities    
        else:
            final_preds = np.zeros((y_test.shape[0]),dtype=float)
            count = 0
            for i in range(nfids):
                final_preds[:] += test_preds[:,count]
                count += 1
            for i in range(nfids-1):
                final_preds -= test_preds[:,count]
                count += 1
        
        tend = time.time()
        self.predict_time = tend-tstart
        
        #calculate MAE and RMSE if y_test given
        if not isinstance(y_test,type(None)):
            self.mae = np.mean(np.abs(final_preds-y_test))
            self.rmse = np.sqrt(np.mean((final_preds-y_test)**2))
        
        return final_preds
    
    def kernel_space_optimiser(self, X_val:np.ndarray, val_trues:np.ndarray):
        nfids = self.indexes.shape[0]
        #create the big lambda matrix
        bigLam = np.zeros((2*nfids-1, 2*nfids-1),dtype=float)
        Mval = np.zeros((2*nfids-1),dtype=float)
        
        #perform KRR for validation set
        val_K = self.kernel_generators(X1=X_val)
        alpha_val = self.KRR(val_K, val_trues)
        
        #prepare X_trains for this process by unpacking
        unpacked_X_trains = np.zeros((2*nfids-1),dtype=object)
        count = 0
        for i in range(nfids):
            unpacked_X_trains[count] = self.X_trains[i]
            count += 1
        for i in range(nfids-1):
            unpacked_X_trains[count] = self.X_trains[i+1]
            count += 1
        
        for s in tqdm(range(2*nfids-1), leave=self.p_bar, 
                      desc='Calculating subspace coefficients...'):
            #K_s = self.kernel_generators(X1 = unpacked_X_trains[s])
            K_sval = self.kernel_generators(X1=unpacked_X_trains[s], X2=X_val)
            Mval[s] = np.dot(self.models[s], np.dot(K_sval, alpha_val))
            for m in range(2*nfids-1):
                K_sm = self.kernel_generators(X1=unpacked_X_trains[s], 
                                              X2=unpacked_X_trains[m]) 
                temp_inn = np.dot(self.models[s], np.dot(K_sm, self.models[m]))
                bigLam[s,m] = np.copy(temp_inn)
                #bigLam[m,s] = np.copy(temp_inn)
        '''
        for s in tqdm(range(2*nfids-1), leave=self.p_bar, 
                      desc='Calculating subspace coefficients...'):
            K_s = self.kernel_generators(X1 = unpacked_X_trains[s])
            bigLam[s,s] = np.dot(self.models[s], np.dot(K_s, self.models[s]))
            #Build the M-validation matrix
            K_sval = self.kernel_generators(X1=unpacked_X_trains[s], X2=X_val)
            Mval[s] = np.dot(self.models[s], np.dot(K_sval, alpha_val))
            
            for m in range(s+1, 2*nfids-1):
                K_sm = self.kernel_generators(X1=unpacked_X_trains[s], 
                                              X2=unpacked_X_trains[m])
                temp_inn = np.dot(self.models[s], np.dot(K_sm, self.models[m]))
                bigLam[s,m] = np.copy(temp_inn)
                bigLam[m,s] = np.copy(temp_inn)
        '''
        #perform Cho-decomp to solve bigLam * C = Mval
        #bigLam[np.diag_indices_from(bigLam)] += self.reg
        kernel_coeffs = cho_solve(bigLam, Mval)
        #kernel_coeffs = kernel_coeffs/np.sum(kernel_coeffs)
        #np.save('Xun1',unpacked_X_trains)
        #np.save('val_al1',alpha_val)
        #np.save('Mval1',Mval)
        #np.save('bigLam1',bigLam)
        return kernel_coeffs

#Other optimizers for LCC
'''
elif optimiser=='kernel_space':
    assert not isinstance(y_val,type(None)), "Validation set cannot be None for kernel space optimisation"
    final_preds = np.zeros((y_test.shape[0]), dtype=float)
    #perform kernel space coefficient optimisation
    self.coeffs = self.kernel_space_optimiser(X_val=X_val, 
                                              val_trues=y_val)
    for s in range(2*nfids-1):
        final_preds += self.coeffs[s]*test_preds[:,s]
'''