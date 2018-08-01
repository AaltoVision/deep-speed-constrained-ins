# deep-speed-constrained-ins

Strapdown inertial navigation systems are sensitive to the quality of the data provided by the accelerometer and gyroscope. Low-grade IMUs in handheld smart-devices pose a problem for inertial odometry on these devices. We propose a scheme for constraining the inertial odometry problem by complementing non-linear state estimation by a CNN-based deep-learning model for inferring the momentary speed based on a window of IMU samples. We show the feasibility of the model using a wide range of data from an iPhone, and present proof-of-concept results for how the model can be combined with an inertial navigation system for three-dimensional inertial navigation.

Requirements:

* Python 2
* pytorch
* pandas
* numpy
* Matplotlib
* numpy
* Jupyter notebook

To produce paper figures:

Go trough the cells in  DCI-training-0.0.2.ipynb, by default the script loads a pretrained model and evaluates the performance.


