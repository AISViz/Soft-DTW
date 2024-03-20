# Soft-Dynamic Time Warping (Soft-DTW) for TensorFlow-Keras
[Soft-DTW](https://github.com/mblondel/soft-dtw) is a variation of Dynamic Time Wrapping. A gamma factor is introduced in the minimum function. The authors developed this distance function in python for generic use and primary implementation is available [here](https://github.com/mblondel/soft-dtw). 

This repository provides the implementation of Soft-DTW as loss function for batch processing in Keras/Tensorflow models. First, Euclidean distance matrix is calculated for whole batch at once. In the next step, each sample in the batch is traversed sequentially to calculate loss (distance). To speed up the process, [same](https://github.com/mblondel/soft-dtw/blob/master/sdtw/soft_dtw_fast.pyx) Cython function **`min(a,b,c, gamma)`** implemented by authors is used. 


# python libraries

`conda install numpy scipy scikit-learn cython nose`

`python3 -m pip install tensorflow==2.10`

# Setup

`cython sdtw/soft_dtw_fast.pyx`

`python setup.py build`

`python setup.py build_ext --inplace`
