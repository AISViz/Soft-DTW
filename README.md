# Soft-Dynamic Time Warping (Soft-DTW) for TensorFlow-Keras
Soft-DTW loss function for batch processing in keras Tensforflow. To speed up the process, Cython function **`min(a,b,c, gamma)`** is used. The function For each batch,  first calculates the euclidean distance matrix


# python libraries

`conda install numpy scipy scikit-learn cython nose`

`python3 -m pip install tensorflow==2.10`

# Setup

`cython sdtw/soft_dtw_fast.pyx`

`python setup.py build`

`python setup.py build_ext --inplace`
