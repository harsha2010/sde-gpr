import detection
import estimation
from evaluation import *
from data import nyc

# Load appropriate classes and functions
poly = estimation.Poly_Kernel
se = estimation.SE_Kernel
fit_model = estimation.fit_model
Detector = detection.Detector


if __name__=="__main__":

    ### Detection Parameters
    W = 20

    ### Bootstrapping parameters
    alpha = 0.01
    B = 10000

    ### Kernel functions
    train_drift_kernel = se(l=0.25, sigma=0.2)
    diffusion_kernel = poly(power=0,scale=1)
    test_drift_kernel = poly(power=1, scale=5)

    ### Initialise detector
    detector = Detector(train_drift_kernel, diffusion_kernel, test_drift_kernel, diffusion_kernel, 0.01)

    ### Load in NYC data
    x_train, t_train, ts_train, x_test, t_test, ts_test, \
    anomaly_ts, true, true_corrected, data_train, data_test = nyc.load()

    ### Bootstrapping
    detector.bootstrap(x_train, t_train, alpha, B, W)

    ### Testing
    detector.anomaly_score(x_train, t_train, x_test, t_test, W)

    ### Plot results
    plot_nyc_results(data_test, ts_test, detector.scores, detector.threshold, anomaly_ts, save_fig=True)

    ### Calculate relevant metrics
    save_nyc_results(true, true_corrected, detector.scores, detector.threshold, alpha)
