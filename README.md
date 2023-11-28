# MAHyNet: Parallel Hybrid Network for RNA-Protein Binding Sites Prediction Based on Multi-Head Attention and Expectation pooling
**Introduction**
****
   In this work, a new parallel network that integrates the multi-head attention mechanism and the expectation pooling is proposed, named MAHyNet. The left-branch network of this model mixes convolutional neural network and gated recurrent neural network, and its right-branch network is a two-layer convolutional neural network, which can extract the features of one-hot and RNA base physicochemical properties, respectively.
****
**Requirements**
****
* Keras = 2.1.6  
* tensorflow-gpu =1.8.0  
* h5py  
* pool  
* tqdm  
* sklearn
****
**Non-10-fold cross-validation**
****
```
python generate_hdf5.py
python generate_hdf5_ph.py
python train_data.py 0 0 53 
python save_result.py
```
****
**10-fold cross-validation**
****
```
python generate_hdf5_10.py 
python generate_hdf5_10ph.py
python train_ten_data.py 0 0 53  
python see_10_fold.py  
```
