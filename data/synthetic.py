import numpy as np

def load(train_test_split=0.5,x0=0, N=8000):
    x = [x0]
    x_prev = x0
    true = np.zeros(N)
    s = int(N*train_test_split)
    for i in range(N-1):
        if i<int(3*N/5) or i>int(7*N/10):
            x_new = 0.1 * x_prev + -0.3 * x_prev ** 3 + np.sqrt(0.2) * np.random.randn()
            x.append(x_new)
            x_prev = x_new
        else:
            x_new = 0.75 * x_prev + np.sqrt(0.15) * np.random.randn()
            x.append(x_new)
            x_prev = x_new
            true[i+1] = 1
    x_train = x[:s]; t_train = np.arange(s)
    x_test = x[s:]; t_test = np.arange(s,N,1)
    return x_train, t_train, x_test, t_test, true[s:]
