from estimation import *

def girsanov(X, t, model):
    # Return one of the two model's contribution to the log-likelihood ratio
    dX = np.diff(X)
    dt = np.diff(t)

    UD = model.lamperti_transform_drift(X[:-1])

    # Discretised Girsanov's theorem
    S1 = np.multiply(UD, dX)
    S2 = np.multiply(np.square(UD), dt)
    return sum(S1 - (1/2)*S2)


class Detector:
    def __init__(self, train_drift_kernel, train_diffusion_kernel, test_drift_kernel, test_diffusion_kernel, regularisation_const):

        self.train_drift_kernel = train_drift_kernel
        self.train_diffusion_kernel = train_diffusion_kernel
        self.test_drift_kernel = test_drift_kernel
        self.test_diffusion_kernel = test_diffusion_kernel
        self.reg = regularisation_const

        self.train = None
        self.threshold = None
        self.scores = None


    def LLR(self, X, t, train_model, test_model):
        # Compute the LLR of test and train models at X
        return girsanov(X, t, test_model) - girsanov(X, t, train_model)


    def bootstrap(self, X_train, t_train, alpha, B, W):
        # alpha:  desired FPR
        # B:      number of bootstrap samples
        # W:      test sliding window length

        self.anomaly_score(X_train, t_train, X_train, t_train, W)

        scores = np.array(self.scores[W:])
        scores = scores[~np.isnan(scores)]  # Remove nan values from array
        bootstrap_samples = np.random.choice(scores, B)

        # Get 1-alpha threshold
        nu = np.percentile(bootstrap_samples, (1 - alpha) * 100)

        self.threshold = nu


    def anomaly_score(self, X_train, t_train, X_test, t_test, W):
        # W: test sliding window length

        # Fit training model, unless already fit during bootstrap
        if self.train is None:
            self.train = fit_model(X_train, t_train, self.train_drift_kernel, self.train_diffusion_kernel, self.reg)

        # Initialise scores at zero
        scores = np.zeros(len(t_test))

        for i in range(len(t_test)-W):
            Xi = X_test[i:i + W]
            ti = t_test[i:i + W]
            test = fit_model(Xi, ti, self.test_drift_kernel, self.test_diffusion_kernel, self.reg)
            scores[i+W] = self.LLR(Xi, ti, self.train, test)

        self.scores = scores

        return scores

    def plot_scores(self, X_test, t_test):
        assert self.scores is not None, "Set the anomaly scores."
        fig, ax = plt.subplots(nrows=2,ncols=1)
        ax[0].plot(t_test, X_test, color='k')
        ax[1].plot(t_test, self.scores, color='r')
        plt.show()

    # def plot_nyc_results(self, X_test, ts_test, anomaly_ts):
    #     assert self.scores is not None, "Set the anomaly scores."
    #     assert self.threshold is not None, "Set the detection threshold."
    #     fig, ax = plt.subplots(nrows=2,ncols=1)
    #     ax[0].plot(ts_test, X_test, color='k')
    #     for anomaly in anomaly_ts:
    #         ax[0].axvline(anomaly, color='r', linestyle='--')
    #     ax[1].plot(ts_test, self.scores, color='r')
    #     ax[1].axhline(self.threshold, color='k', linestyle='--')
    #     plt.show()
