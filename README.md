# SimpleKalmanFilter

A simple 1D Kalman filter implementation, which can be used to estimate the state of a system given noisy measurements.

The `SimpleKalmanFilter1D` class is initialized with the `process noise` and `measurement noise`, and an `initial state` (default is 0). It sets up the Kalman filter using the `kalmankit` library, which provides the functionality to perform the prediction and update steps of the filter.

