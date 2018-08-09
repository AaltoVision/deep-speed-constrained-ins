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

Download and unzip ADVIO dataset files
[ADVIO](https://github.com/AaltoVision/ADVIO)

```
cd data
for i in $(seq -f "%02g" 1 23);
do
  wget -O advio-$i.zip https://zenodo.org/record/1321157/files/advio-$i.zip
  unzip advio-$i.zip
  rm advio-$i.zip	
done
```

Synchronize the accelerometer and gyroscope in the ADVIO data.

```
python python/sync-data
```


