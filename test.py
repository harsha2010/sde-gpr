from detection import *

# test time series
N = 100
X_train = np.random.randn(N)
X_train = np.cumsum(X_train)
t_train = np.arange(N)

X_test = np.random.randn(N)
X_test[int(N/2):] = 5 * X_test[int(N/2):]
X_test = np.cumsum(X_test)
t_test = np.arange(N)

### Test Estimation ###
# drk = SE_Kernel(l=0.5, sigma=1)
# drk = Poly_Kernel(power=1, scale=1)
# dfk = Poly_Kernel(power=0)
# model = fit_model(X_train, t_train, drk, dfk, 0.01)
# model.plot_drift(min(X_train), max(X_train), X_train, t_train)
# model.plot_diffusion(min(X_train), max(X_train), X_train, t_train)

### Test Detection ###
# train/test kernels
train_drk = SE_Kernel(l=1, sigma=1)
train_drk = Poly_Kernel(power=0)
train_dfk = Poly_Kernel(power=0)
test_drk = SE_Kernel(l=1)
test_drk = Poly_Kernel(power=1, scale=3)
test_dfk = Poly_Kernel(power=0)

W = 10

# Testing detector
detector = Detector(train_drk, train_dfk, test_drk, test_dfk, 0.01)
detector.anomaly_score(X_train, t_train, X_test, t_test, W)
detector.plot_scores(X_test, t_test)

