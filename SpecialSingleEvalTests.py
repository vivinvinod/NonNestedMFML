import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from TrueNonNested_Model_MFML import ModelMFML as MF

def main_withTZVP():
    prop='SCF'
    rep='CM'
    indexes = np.load('CheMFi/raws/indexes.npy',allow_pickle=True) #STO3G first
    X_train = np.load(f'CheMFi/raws/X_train_{rep}.npy')
    X_test = np.load(f'CheMFi/raws/X_test_{rep}.npy')
    X_val = np.load(f'CheMFi/raws/X_val_{rep}.npy')
    energies = np.load(f'CheMFi/raws/energies_{prop}.npy',allow_pickle=True) #STO3G first
    for i in range(5):
        avg=np.mean(energies[i])
        energies[i] = energies[i] - avg
    #energies = energies[:-1]
    y_test = np.load(f'CheMFi/raws/y_test_{prop}.npy') - avg
    y_val = np.load(f'CheMFi/raws/y_val_{prop}.npy') - avg
    
    
    nfids = indexes.shape[0]
    regs = np.full(2*nfids-1,1e-10)
    sigmas = np.full(2*nfids-1,150.0)
    kernels = np.full(2*nfids-1,'matern')
    
    n_trains = np.asarray([2**(13), 2**(12), 2**(11), 2**(10),2**(9)])
    #init
    model = MF(reg=regs, kernel=kernels, sigma=sigmas,
               order=1, metric='l2', gammas=None, 
               p_bar=True)
    #train models
    model.train(X_train_parent=X_train, y_trains=energies, 
            indexes=indexes, 
            shuffle=True, n_trains=n_trains, 
            seed=42)
    _ = model.predict(X_test=X_test, X_val=X_val,
                      y_test=y_test, y_val=y_val, 
                      optimiser='OLS')
    print('MAE for non-nested o-MFML on **including** TZVP fidelity in ground state energy training data: ',model.mae)

def main_noTZVP():
    prop='SCF'
    rep='CM'
    indexes = np.load('CheMFi/raws/indexes.npy',allow_pickle=True)[:-1] #STO3G first
    X_train = np.load(f'CheMFi/raws/X_train_{rep}.npy')
    X_test = np.load(f'CheMFi/raws/X_test_{rep}.npy')
    X_val = np.load(f'CheMFi/raws/X_val_{rep}.npy')
    energies = np.load(f'CheMFi/raws/energies_{prop}.npy',allow_pickle=True)[:-1] #STO3G first
    for i in range(4):
        avg=np.mean(energies[i])
        energies[i] = energies[i] - avg
    #energies = energies[:-1]
    y_test = np.load(f'CheMFi/raws/y_test_{prop}.npy') - avg
    y_val = np.load(f'CheMFi/raws/y_val_{prop}.npy') - avg
    
    
    nfids = indexes.shape[0]
    regs = np.full(2*nfids-1,1e-10)
    sigmas = np.full(2*nfids-1,150.0)
    kernels = np.full(2*nfids-1,'matern')
    
    n_trains = np.asarray([2**(13), 2**(12), 2**(11), 2**(10)])#[5-nfids:]
    #init
    model = MF(reg=regs, kernel=kernels, sigma=sigmas,
               order=1, metric='l2', gammas=None, 
               p_bar=True)
    #train models
    model.train(X_train_parent=X_train, y_trains=energies, 
            indexes=indexes, 
            shuffle=True, n_trains=n_trains, 
            seed=42)
    _ = model.predict(X_test=X_test, X_val=X_val,
                      y_test=y_test, y_val=y_val, 
                      optimiser='OLS')
    print('MAE for non-nested o-MFML **without** TZVP fidelity in ground state energy  training data: ',model.mae)
    
    
main_withTZVP()
main_noTZVP()
