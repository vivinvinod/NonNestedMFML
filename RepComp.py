import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

def main():
    indexes = np.load('CheMFi/raws/nested_indexes.npy',allow_pickle=True)
    X_train = np.load(f'CheMFi/raws/X_train_{rep}.npy')
    X_test = np.load(f'CheMFi/raws/X_test_{rep}.npy')
    X_val = np.load(f'CheMFi/raws/X_val_{rep}.npy')
    energies = np.load(f'CheMFi/raws/energies_{prop}.npy',allow_pickle=True) #STO3G first
    for i in range(5):
        avg=np.mean(energies[i])
        energies[i] = energies[i] - avg
    #the last energies array object is the TZVP which is also the target fidelity
    y_test = np.load(f'CheMFi/raws/y_test_{prop}.npy') - avg
    y_val = np.load(f'CheMFi/raws/y_val_{prop}.npy') - avg
    
    all_maes = np.zeros((5),dtype=object)
    def_maes = np.zeros((5),dtype=object)
    #run for different baselines
    sf_maes = SF_learning_curve(X_train[:768,:], X_test, energies[-1][:768], y_test, 
                      sigma=150.0, reg=1e-10, navg=10,ker='matern')
    
    np.save(f'CheMFi/outs/sf_{prop}_{rep}.npy',sf_maes)
    
rep='sortCM'
prop='SCF'
main()