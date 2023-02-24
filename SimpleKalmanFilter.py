import numpy as np
from kalmankit import KalmanFilter
from kalmankit.utils import check_none_and_broadcast

class SimpleKalmanFilter1D(object):

    def __init__(self, process_noise, measurement_noise, initial_state=0): 

        self._A = np.array([[1.]])
        self._xk = np.array([[1]]) # initial state
        self._B = check_none_and_broadcast(None, self._A)
        self._Pk = np.array([[1]]) # initial state covariance
        self._H = np.array([[1.]])
        self._Q = np.float64(process_noise) # process noise
        self._R = np.float64(measurement_noise) # measurement noise

        self._kf = KalmanFilter(
            A  = self._A,
            xk = self._xk, # initial state
            B  = self._B,
            Pk = self._Pk, # initial state covariance
            Q  = self._Q, # process noise
            H  = self._H,
            R  = self._R, # measurement noise
        )

        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
    
    @property
    def process_noise(self):
        return self._process_noise
    
    @process_noise.setter
    def process_noise(self, value):
        self._process_noise = value
        self._Q = np.float64(value)
        self._kf.Q = self._Q
    
    @property
    def measurement_noise(self):
        return self._measurement_noise

    @measurement_noise.setter
    def measurement_noise(self, value):
        self._measurement_noise = value
        self._R = np.float64(value)
        self._kf.R = self._R

    def filter(self, measurement):
        self._xk_prior, self._Pk_prior = self._kf.predict(
            Ak=self._A, xk=self._xk, Bk=self._B, uk=None, Pk=self._Pk, Qk=self._Q
        )

        # update step, correct prior estimates
        self._xk_posterior, self._Pk_posterior = self._kf.update(
            Hk=self._H, xk=self._xk_prior, Pk=self._Pk_prior, zk=measurement, Rk=self._R
        )

        self._xk = self._xk_posterior
        self._Pk = self._Pk_posterior

        return self._xk_posterior[0][0]