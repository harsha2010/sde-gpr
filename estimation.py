import numpy as np
import matplotlib.pyplot as plt

### SDE model based on GPR

class SDE_Model:

    def __init__(self, X, kernel_drift, kernel_diffusion, KDIy, KI, KIy):

        # Kernel objects
        self.kernel_drift = kernel_drift
        self.kernel_diffusion = kernel_diffusion

        # Storage of inverse matrices and accociated matrix vector products for efficiency
        self.KDIy = KDIy
        self.KI = KI
        self.KIy = KIy

        # Data for kernel evaluations
        self.X = X
        self.X_hat = X[:-1]

    def drift(self, x):
        # x can be scalar or vector valued
        kx = self.kernel_drift.K(x, self.X_hat)

        # Evaluate the mean of drift GPR at x
        return kx @ self.KIy

    def diffusion(self, x):
        # x can be scalar or vector valued
        kdx = self.kernel_diffusion.K(x, self.X_hat)

        # Evaluate the mean of diffusion GPR at x
        return kdx @ self.KDIy

    def lamperti_transform_drift(self, x):
        # Drift of equivalent unitary diffusion SDE, based on Lamperti transorm

        # Kept in vector form for speed
        a = np.power(abs(self.diffusion(x)), -1/2)
        b = self.drift(x)

        kdx_grad = self.kernel_diffusion.dK(x, self.X_hat)
        diffusion_gradient = kdx_grad @ self.KDIy

        return np.multiply(a , (b - ((1/4) * diffusion_gradient)))

    def plot_drift(self, xlow, xhigh, X=None, t=None):
        x = np.linspace(xlow, xhigh, 1000)

        plt.figure()
        plt.plot(x, self.drift(x), color='k')
        if X is not None and t is not None:
            dX = np.diff(X)
            dt = np.diff(t)
            y = np.divide(dX,dt)
            plt.scatter(X[:-1],y,color='b',alpha=0.2)
        plt.show()

    def plot_diffusion(self, xlow, xhigh, X=None, t=None):
        x = np.linspace(xlow, xhigh, 1000)

        plt.figure()
        plt.plot(x, self.diffusion(x), color='k')
        if X is not None and t is not None:
            dX = np.diff(X)
            dt = np.diff(t)
            y = np.divide(np.square(dX),dt)
            plt.scatter(X[:-1],y,color='b',alpha=0.2)
        plt.show()


def gpr(X, t, kernel_drift, kernel_diffusion, regularisation_const, calculate_marginal_likelihood=False):
    dX = np.diff(X)
    dt = np.diff(t)

    # Compute regression data pairs
    X_hat = X[:-1]
    y_tilde = np.divide(np.square(dX), dt)
    y = np.divide(dX, dt)

    # Regularisation matrix
    lam = regularisation_const * np.eye(len(X_hat))

    # Diffusion GPR
    KD = kernel_diffusion.K(X_hat, X_hat)
    KDI = np.linalg.lstsq(KD + lam, np.eye(len(X_hat)), rcond=None)[0]
    KDIy = KDI @ y_tilde

    # Compute diffusion at data
    D = KD @ KDIy

    sigma = np.diag(np.divide(D, dt))

    # Drift GPR
    K = kernel_drift.K(X_hat, X_hat)
    KI = np.linalg.lstsq(K + sigma, np.eye(len(X_hat)), rcond=None)[0]
    KIy = KI @ y

    # Report marginal likelihood if option selected

    if calculate_marginal_likelihood:
        data_fit = y.T @ KDI @ y
        complexity = float(np.linalg.slogdet(K + sigma)[1])
        return KDIy, KI, KIy, -(1/2)*(data_fit + complexity)

    return KDIy, KI, KIy, []


### Main functions for fitting a model and calculating marginal likelihood

def fit_model(X, t, kernel_drift, kernel_diffusion, regularisation_const):
    KDIy, KI, KIy, _ = gpr(X, t, kernel_drift, kernel_diffusion, regularisation_const)
    return SDE_Model(X, kernel_drift, kernel_diffusion, KDIy, KI, KIy)

def marginal_likelihood(X, t, kernel_drift, kernel_diffusion, regularisation_const):
    _, _, _, ML = gpr(X, t, kernel_drift, kernel_diffusion, regularisation_const, calculate_marginal_likelihood=True)
    return ML


### Kernel classes

class SE_Kernel:

    def __init__(self, l, sigma=1):
        self.length_scale = l
        self.amplitude_scale = sigma

    def difference_matrix(self, x, y):
        x = np.array(x); y = np.array(y)
        x = x.reshape((len(x), 1)); y = y.reshape((1, len(y)))
        Y = x - y   # Uses broadcasting for speed
        return Y

    def K(self, x, y):
        Y = self.difference_matrix(x, y)
        return np.square(self.amplitude_scale) * np.exp(-np.square(Y) / (2 * np.square(self.length_scale)))

    def dK(self, x, y):
        Y = self.difference_matrix(x, y)
        E = np.square(self.amplitude_scale) * np.exp(-np.square(Y) / (2 * np.square(self.length_scale)))
        return np.multiply(E, -Y/np.square(self.length_scale))

class Poly_Kernel:

    def __init__(self, power, scale=1, constant=1):
        self.p = power
        self.s = scale
        self.c = constant

    def poly_matrix(self, x, y, c):
        x = np.array(x); y = np.array(y)
        Y = np.outer(x, y) + c * np.ones((len(x), len(y)))
        return Y

    def K(self, x, y):
        Y = self.poly_matrix(x, y, self.c)
        return np.square(self.s) * np.power(Y, self.p)

    def dK(self, x, y):
        # Returns derivative of kernel function wrt x
        Y = self.poly_matrix(x, y, self.c)
        return np.square(self.s) * self.p * np.multiply(np.power(Y, self.p - 1), y)
