import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from TrueNonNested_Model_MFML import ModelMFML as MF

def KRR(X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, sigma:float, reg: float, type_kernel:str='gaussian'):
    #generate the correct kernel matrix as prescribed by args parser
    import qml.kernels as k
    from qml.math import cho_solve
    if type_kernel=='matern':
        K_train = k.matern_kernel(X_train,X_train,sigma, order=1, metric='l2')
        K_test = k.matern_kernel(X_train,X_test,sigma, order=1, metric='l2')
    elif type_kernel=='laplacian':
        K_train = k.laplacian_kernel(X_train,X_train,sigma)
        K_test = k.laplacian_kernel(X_train,X_test,sigma)
    elif type_kernel=='gaussian':
        K_train = k.gaussian_kernel(X_train,X_train,sigma)
        K_test = k.gaussian_kernel(X_train,X_test,sigma)
    elif type_kernel=='linear':
        K_train = k.linear_kernel(X_train,X_train)
        K_test = k.linear_kernel(X_train,X_test)
    elif type_kernel=='sargan':
        K_train = k.sargan_kernel(X_train,X_train,sigma,gammas=None)
        K_test = k.sargan_kernel(X_train,X_test,sigma,gammas=None)
    
    #regularize 
    K_train[np.diag_indices_from(K_train)] += reg
    #train
    alphas = cho_solve(K_train,y_train)
    #predict
    preds = np.dot(alphas, K_test)
    #MAE calculation
    mae = np.mean(np.abs(preds-y_test))
    
    return mae

def SF_learning_curve(X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, 
                      sigma:float=30, reg:float=1e-10, navg:int=10, ker:str='laplacian'):
    full_maes = np.zeros((9),dtype=float)
    for n in tqdm(range(navg), desc='avg loop for SF LC'):
        maes = []
        X_train,y_train = shuffle(X_train, y_train, random_state=42)
        for i in range(1,10):
            #start_time = time.time()
            temp = KRR(X_train[:2**i],X_test,y_train[:2**i],y_test,sigma=sigma,reg=reg,type_kernel=ker)
            maes.append(temp)
        full_maes += np.asarray(maes)
    
    full_maes = full_maes/navg
    return full_maes

def truenonnested_same_hyperparams(X_train, energies, indexes, X_test, y_test, X_val, y_val, reg:str=1e-9, sig:float=200.0, ker:str='laplacian', navg:int=10):
    
    all_maes = np.zeros((9),dtype=float)
    all_olsmaes = np.zeros((9),dtype=float)
    
    
    nfids = indexes.shape[0]
    regs = np.full(2*nfids-1,reg)
    sigmas = np.full(2*nfids-1,sig)
    kernels = np.full(2*nfids-1,ker)
    
    for n in tqdm(range(navg),desc='avg-run loop for nested same hyperparams.'):
        maes = []
        ols_maes = []
        for i in tqdm(range(1,10),desc='n train loop',leave=False):
            n_trains = np.asarray([2**(i+4),2**(i+3),2**(i+2),2**(i+1),2**(i)])[5-nfids:]
            
            #instantiate models
            model = MF(reg=regs, kernel=kernels, sigma=sigmas,
                   order=1, metric='l2', gammas=None, 
                   p_bar=False)
            
            #train models
            model.train(X_train_parent=X_train, y_trains=energies, 
                    indexes=indexes, 
                    shuffle=True, n_trains=n_trains, 
                    seed=n)
            
            
            #default predictions
            _ = model.predict(X_test=X_test, X_val=X_val,
                              y_test=y_test, y_val=y_val, 
                              optimiser='default')
            maes.append(model.mae)
            
            #OLS predictions
            _ = model.predict(X_test=X_test, X_val=X_val,
                              y_test=y_test, y_val=y_val, 
                              optimiser='OLS')
            ols_maes.append(model.mae)
        
        #store MAEs into overall arrays
        all_maes[:] += np.asarray(maes)
        all_olsmaes[:] += np.asarray(ols_maes)
        
    
    return all_maes/navg, all_olsmaes/navg
    

def main():
    indexes = np.load('CheMFi/raws/indexes.npy',allow_pickle=True) #STO3G first
    X_train = np.load(f'CheMFi/raws/X_train_{rep}.npy')
    X_test = np.load(f'CheMFi/raws/X_test_{rep}.npy')
    X_val = np.load(f'CheMFi/raws/X_val_{rep}.npy')
    energies = np.load(f'CheMFi/raws/energies_{prop}.npy',allow_pickle=True) #STO3G first
    for i in range(5):
        avg=np.mean(energies[i])
        energies[i] = energies[i] - avg
    y_test = np.load(f'CheMFi/raws/y_test_{prop}.npy') - avg
    y_val = np.load(f'CheMFi/raws/y_val_{prop}.npy') - avg
    
    all_maes = np.zeros((5),dtype=object)
    def_maes = np.zeros((5),dtype=object)
    #run for different baselines
    all_maes[0] = SF_learning_curve(X_train[:768,:], X_test, energies[-1][:768], y_test, 
                      sigma=150.0, reg=1e-10, navg=10, ker='matern')
    def_maes[0] = np.copy(all_maes[0])
    for fb in range(4):
        def_maes[fb+1],all_maes[fb+1]= truenonnested_same_hyperparams(X_train, energies[fb:], 
                                                                      indexes[fb:], X_test, 
                                                                      y_test, X_val, 
                                                                      y_val, reg=1e-10, 
                                                                      sig=150.0, 
                                                                      ker='matern', navg=10)
    
    np.save(f'CheMFi/outs/TrueNonNestedSamedefallMAEs_{prop}_CM.npy',def_maes,allow_pickle=True)
    np.save(f'CheMFi/outs/TrueNonNestedSameOLSallMAEs_{prop}_CM.npy',all_maes,allow_pickle=True)

prop='SCF'
rep='CM'
main()