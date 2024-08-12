# Non-Nested Configuration of MFML and o-MFML
This code repository accompanies the manuscript currently hosted as preprint at [https://arxiv.org/abs/2407.17087] and contains the scripts used to generate the various results and plots of therein. The CheMFi dataset was used in this work and can be found at [https://zenodo.org/records/11636903] with a preprint of the data descriptor at [https://arxiv.org/abs/2406.14149] for reference. The scripts included in this repository and their corresponding use are listed below for ready reference.

* The script `RepComp.py` compares the different representations for a single fidelity KRR model on excitation energies and ground state energies from CheMFi. 
* `Model_MFML.py` is the module that was developed in [this previous work](https://iopscience.iop.org/article/10.1088/2632-2153/ad2cef) and contains both both MFML and o-MFML implementations 
* `TrueNonnested_Model_MFML.py` is the development of MFML and o-MFML with a non-nested configuration of these approaches as documented in the preprint.
* `NestedCheMFiAllLCs.py` produces the output corresponding to the nested configurations of MFML and o-MFML.
* `NonNestedCheMFiAllLCs.py` produces the output required to test the non-nested configuration of MFML and o-MFML.
* The jupyter notebook `PlottingRoutines.ipynb` contains the functions to produce the plots that result from the outputs of the various scripts.
