## Summary

Strapdown inertial navigation systems are sensitive to the quality of the data provided by the accelerometer and gyroscope. Low-grade IMUs in handheld smart-devices pose a problem for inertial odometry on these devices. We propose a scheme for constraining the inertial odometry problem by complementing non-linear state estimation by a CNN-based deep-learning model for inferring the momentary speed based on a window of IMU samples.

This repository provides the codes for replicationg the speed regression setup in [1]. Please, if you use this code/data, please cite the original paper presenting it.


## Codes

[Codes on GitHub](https://github.com/AaltoVision/deep-speed-constrained-ins)

## Referencing

[1] Santiago Cortés, Arno Solin, and Juho Kannala, “Deep Learning Based Speed Estimation for Constraining Strapdown Inertial Navigation on Smartphones”, *IEEE International Workshop on Machine Learning for Signal Processing (MLSP)*, Aalborg, Denmark, 2018.

