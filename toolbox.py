import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import statsmodels.api as sm
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Define correlation coefficient function
def correlation_coefficient_cal(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    product_sum = 0
    x_sum = 0
    y_sum = 0
    for i in range(0, len(x)):
        x_mean_y_mean_product = (x[i]-x_mean)*(y[i]-y_mean)
        x_mean_square = (x[i]-x_mean)**2
        y_mean_square = (y[i]-y_mean)**2
        product_sum += x_mean_y_mean_product
        x_sum += x_mean_square
        y_sum += y_mean_square
    r = (product_sum)/(((x_sum)**(1/2))*((y_sum)**(1/2)))
    return r

# Define ACF function
def autocorr(y, lags, label):
    # y_bar
    y_bar = np.mean(y)

    # denominator
    den = 0
    for i in range(len(y)):
        yt_y_bar_square = (y[i] - y_bar) ** 2
        den += yt_y_bar_square

    ry = []
    # numerator
    for l in range(0, lags + 1):
        num = 0
        for j in range(l, len(y)):
            yt_y_bar_yt_lags_y_bar = (y[j] - y_bar) * (y[j - l] - y_bar)
            num += yt_y_bar_yt_lags_y_bar
        est_autocorr = num / den
        ry.append(est_autocorr)

    # list of estimate autocorrelation
    ryy = ry[::-1]
    Ry = ryy[:-1] + ry
    x = np.arange(-lags, lags + 1, 1)

    # Insignificant Region
    m = 1.96 / (len(y) ** (1 / 2))

    # ACF Plot
    plt.stem(x, Ry)
    plt.axhspan(-m, m, alpha=0.25, color='blue')
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title('The ACF of {} with lags = {}'.format(label, lags))
    plt.show()

# Define a function that return a list of estimate autocorrelation on lags values of time series
def autocorr_values(y, lags):
    # y_bar
    y_bar = np.mean(y)

    # denominator
    den = 0
    for i in range(len(y)):
        yt_y_bar_square = (y[i] - y_bar) ** 2
        den += yt_y_bar_square

    ry = []
    # numerator
    for l in range(0, lags + 1):
        num = 0
        for j in range(l, len(y)):
            yt_y_bar_yt_lags_y_bar = (y[j] - y_bar) * (y[j - l] - y_bar)
            num += yt_y_bar_yt_lags_y_bar
        est_autocorr = num / den
        ry.append(est_autocorr)

    # list of estimate autocorrelation
    ryy = ry[::-1]
    Ry = ryy[:-1] + ry
    return Ry

# Define GPAC function
def Cal_GPAC(Ry,j,k,label):
    mid = int((len(Ry)-1)/2)               # Find the the position of Ry(0) in Ry list
    GPAC = pd.DataFrame(np.zeros(shape=(j+1,k+1))).drop([0],axis=1)       # Create a GPAC Table with the size of j+1 * k

    for K in range(1,k+1):
        col = []
        for J in range(j+1):
            if K == 1:                      # Input the value into the first column
                num = Ry[mid+J+K]           # numerator
                den = Ry[mid+J]             # denominator
                res = num/den
                GPAC.iloc[J,K-1] = res
            else:                           # Input the value into the rest of columns
                den_list = []
                for i in range(K):
                    den = Ry[mid-J-i:mid-J+K-i]
                    den_list.append(den)
                den_array = np.array(den_list)            # denominator matrix
                num_array = den_array.copy()
                num_array[:,-1] = Ry[mid+J+1:mid+J+K+1]   # numerator matrix, which is same as denominator except the last column of matrix
                num_det = np.linalg.det(num_array)
                den_det = np.linalg.det(den_array)
                if den_det == 0:                          # If denominator is 0, we can't do division and the result is NaN
                    res = np.nan
                else:
                    res = num_det/den_det
                    if abs(res) < 0.001:                  # If the result is close to 0, set the result to 0
                        res = 0
                GPAC.iloc[J,K-1] = res
    # Print Table
    print(GPAC)

    # Plot the Heatmap of Table
    fig,ax = plt.subplots(figsize=(k-2,j-3))
    ax = sns.heatmap(GPAC,annot=True,cmap='YlGnBu')
    plt.title('Generalized Partial Autocorrelation (GPAC) Table of\n{}'.format(label))
    plt.xlabel('$n_a$')
    plt.ylabel('$n_b$')
    plt.show()

# Define GPAC function (only table dataframe)
def Cal_GPAC_DataFrame(Ry,j,k):
    mid = int((len(Ry)-1)/2)               # Find the the position of Ry(0) in Ry list
    GPAC = pd.DataFrame(np.zeros(shape=(j+1,k+1))).drop([0],axis=1)       # Create a GPAC Table with the size of j+1 * k

    for K in range(1,k+1):
        col = []
        for J in range(j+1):
            if K == 1:                      # Input the value into the first column
                num = Ry[mid+J+K]           # numerator
                den = Ry[mid+J]             # denominator
                res = num/den
                GPAC.iloc[J,K-1] = res
            else:                           # Input the value into the rest of columns
                den_list = []
                for i in range(K):
                    den = Ry[mid-J-i:mid-J+K-i]
                    den_list.append(den)
                den_array = np.array(den_list)            # denominator matrix
                num_array = den_array.copy()
                num_array[:,-1] = Ry[mid+J+1:mid+J+K+1]   # numerator matrix, which is same as denominator except the last column of matrix
                num_det = np.linalg.det(num_array)
                den_det = np.linalg.det(den_array)
                if den_det == 0:                          # If denominator is 0, we can't do division and the result is NaN
                    res = np.nan
                else:
                    res = num_det/den_det
                    if abs(res) < 0.001:                  # If the result is close to 0, set the result to 0
                        res = 0
                GPAC.iloc[J,K-1] = res
    # Print Table
    print(GPAC)
    return GPAC



# Define differencing function
def difference(dataset,interval=1):
    diff = []
    for i in range(interval,len(dataset)):
        value = dataset[i]-dataset[i-interval]
        diff.append(value)
    return diff

# Define ADF-test function
def ADF_Cal(x, label):
    print('The ADF test of {}:'.format(label))
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

# Define ACF and PACF plot function
def ACF_PACF_Plot(y,lags,label):
    plt.figure(figsize=(10,8))
    plt.subplot(211)
    plot_acf(y, ax=plt.gca(), lags=lags, title='The ACF and PACF of {}\nAutocorrelation'.format(label))
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    plt.show()

# Define Rolling mean and variance plot function
def Rolling_Mean_Var_Plot(y,label):
    rolling_mean = []
    rolling_var = []
    for i in range(1, len(y)+1):
        df = y[:i]
        rolling_mean.append(np.mean(df))
        rolling_var.append(np.var(df))

    # Plot rolling mean and variance versus time.
    plt.figure(figsize=(10,10))
    # Rolling Mean
    plt.subplot(211)
    plt.plot(rolling_mean, label='Rolling Mean')
    plt.legend()
    plt.grid()
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Means of {}'.format(label))
    # Rolling Variance
    plt.subplot(212)
    plt.plot(rolling_var, label='Rolling Variance')
    plt.legend()
    plt.grid()
    plt.xlabel('Samples')
    plt.ylabel('Magnitude')
    plt.title('Rolling Variances of {}'.format(label))
    plt.show()

# Define plot raw dataset function
def Plot_Raw_Data(y,label):
    plt.figure()
    plt.plot(y,label='Original Data')
    plt.legend()
    plt.xlabel('Number of Samples')
    plt.ylabel('Magnitude')
    plt.title('Original Dataset of {}'.format(label))
    plt.show()

# Define plot raw dataset versus differenced dataset function
def Plot_Raw_Differenced_Data(raw,diff,label):
    plt.figure()
    plt.plot(raw,label='Original Data')
    plt.plot(diff, label='Differenced Data')
    plt.legend()
    plt.xlabel('Number of Samples')
    plt.ylabel('Magnitude')
    plt.title('Original Dataset versus Differenced Dataset {}'.format(label))
    plt.show()

def ARIMA():
    T = int(input('Enter the number of data samples [1000]:'))
    mean_e = int(input('Enter the mean of the white noise [0]:'))
    var_e = int(input('Enter the variance of the white noise [1]:'))
    na = int(input('Enter AR order:'))
    nb = int(input('Enter MA order:'))

    max = np.maximum(na,nb)
    an,bn = [],[]
    if na == 0:
        an = [0]*nb
    else:
        for i in range(1,na+1):
            coefficient_ar = float(input('Enter the coefficient of AR a{}:'.format(str(i))))
            an.append(coefficient_ar)
        an = an + [0]*(max-na)

    if nb == 0:
        bn = [0]*na
    else:
        for j in range(1,nb+1):
            coefficient_ma = float(input('Enter the coefficient of MA b{}:'.format(str(j))))
            bn.append(coefficient_ma)
        bn = bn + [0] * (max - nb)

    arparams = np.array(an)
    maparams = np.array(bn)
    ar = np.r_[1, arparams]
    ma = np.r_[1, maparams]

    # Construct ARMA Process
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    print('Is this a stationary process:', arma_process.isstationary)

    # Generate the white noise
    e = np.random.normal(mean_e, np.sqrt(var_e), size=T)

    system = (ma, ar, 1)
    tout, y_new = signal.dlsim(system,e)
    y = np.ndarray.flatten(y_new)               # np.ndarray.flatten() Return a copy of the array collapsed into one dimension
    return y

# Define a Q value function
# A test on actual autocorrelation values to check how the derived model perform
# If autocorrelation is close to zero, then Q will be small, which means the model is good
# If autocorrelation is large, the Q will be large, which means the model is not good enough
def Q_value(y, lags):
    # y_bar
    y_bar = np.mean(y)

    # denominator
    den = 0
    for i in range(len(y)):
        yt_y_bar_square = (y[i] - y_bar) ** 2
        den += yt_y_bar_square

    ry = []
    # numerator
    for l in range(0, lags + 1):
        num = 0
        for j in range(l, len(y)):
            yt_y_bar_yt_lags_y_bar = (y[j] - y_bar) * (y[j - l] - y_bar)
            num += yt_y_bar_yt_lags_y_bar
        est_autocorr = num / den
        ry.append(est_autocorr)

    # Q value
    ry_square_sum = 0
    for i in range(1, lags + 1):
        ry_square_sum += (ry[i] ** 2)

    Q = len(y) * ry_square_sum
    return Q

# Define Levenberg Marquardt algorithm
def LMA(y, na, nb):
    # Calculate e function
    def e_Cal(theta):
        if na != 0:
            arparams = theta[:na]
        else:
            arparams = np.zeros(nb)
        if nb != 0:
            maparams = theta[na:]
        else:
            maparams = np.zeros(na)

        ar = np.r_[1, arparams]
        ma = np.r_[1, maparams]

        if len(ar) > len(ma):
            ma = np.append(ma,np.zeros(len(ar)-len(ma)))
        if len(ma) > len(ar):
            ar = np.append(ar,np.zeros(len(ma)-len(ar)))

        # generate e(theta1,theta2,...,thetan)
        system_rev = (ar, ma, 1)
        _, e_theta = signal.dlsim(system_rev, y)
        return e_theta

    # Step 1
    def Step_1(theta):
        if na != 0:
            arparams = theta[:na]
        else:
            arparams = np.zeros(nb)
        if nb != 0:
            maparams = theta[na:]
        else:
            maparams = np.zeros(na)

        ar = np.r_[1, arparams]
        ma = np.r_[1, maparams]

        if len(ar) > len(ma):
            ma = np.append(ma,np.zeros(len(ar)-len(ma)))
        if len(ma) > len(ar):
            ar = np.append(ar,np.zeros(len(ma)-len(ar)))

        # generate e(theta1, theta2, ..., thetan)
        system_rev = (ar, ma, 1)
        _, e_theta = signal.dlsim(system_rev, y)

        theta_delta = theta + delta                   # [theta1+delta,theta2+delta,...,thetan+delta]

        for j in range(n):
            if j < na:                                # na>0 or nb=0
                ar_update = ar.copy()
                ar_update[1 + j] = theta_delta[j]     # update ar to [1,theta1,thetai+delta,...,thetan]
                system_rev = (ar_update, ma, 1)
                _, e_theta_delta = signal.dlsim(system_rev, y)  # e(theta1,...,thetai+delta,...,thetan)
                x = (e_theta - e_theta_delta) / delta
                if j == 0:                            # x1
                    X = np.asmatrix(x)
                else:                                 # x2,..,xn
                    X = np.hstack((X, x))
                j += 1
            else:                                     # na=0 or nb>0
                ma_update = ma.copy()
                ma_update[1 + j - na] = theta_delta[j] # update ma to [1,theta1,thetai+delta,...,thetan]
                system_rev = (ar, ma_update, 1)
                _, e_theta_delta = signal.dlsim(system_rev, y)  # e(theta1,...,thetai+delta,...,thetan)
                x = (e_theta - e_theta_delta) / delta
                if j == 0:
                    X = np.asmatrix(x)
                else:
                    X = np.hstack((X, x))
                j += 1

        A = X.T.dot(X)
        g = X.T.dot(e_theta)
        return e_theta, A, g

    # Step 2
    def Step_2(theta, A, g):
        I = np.identity(n)
        theta_diff = np.linalg.inv(A + mu * I) * g
        theta_new = theta.reshape(-1, 1) + theta_diff
        theta_new = np.array(theta_new).flatten()
        return theta_diff, theta_new

    N = len(y)
    n = na + nb
    mu = 0.01
    delta = 10e-6
    iterations = 100
    eps = 10e-3
    mu_max = 10e100
    i = 0
    SSE_list = []

    # first iteration
    theta_0 = np.zeros(n)                                   # [0]*na+[0]*nb

    # step 1
    e_theta, A, g = Step_1(theta_0)
    SSE_theta = np.dot(e_theta.T,e_theta)                   # SSE_theta
    SSE_list.append(SSE_theta.flatten()[0])
    # step 2
    theta_diff, theta_new = Step_2(theta_0, A, g)
    e_theta_new = e_Cal(theta_new)
    SSE_theta_new = np.dot(e_theta_new.T,e_theta_new)       # SSE_theta_new
    SSE_list.append(SSE_theta_new.flatten()[0])

    for i in range(iterations):
        if SSE_theta_new < SSE_theta:
            if np.sqrt(np.dot(theta_diff.T,theta_diff)) < eps:
                est_theta = theta_new
                var_e = SSE_theta_new / (N - n)
                var_e = var_e.flatten()[0]
                if A.shape == (1, 1):
                    inv_A = 1 / A
                else:
                    inv_A = np.linalg.inv(A)
                covar_matrix = var_e * inv_A
                return SSE_theta_new, var_e, covar_matrix, est_theta, SSE_list
            else:
                theta = theta_new
                mu = mu / 10
        while SSE_theta_new >= SSE_theta:
            mu = mu * 10
            if mu > mu_max:
                print('SSE_theta_new did not go down!')
                return theta_new
            # Step 2
            theta_diff, theta_new = Step_2(theta_new, A, g)
            e_theta_new = e_Cal(theta_new)
            SSE_theta_new = np.dot(e_theta_new.T, e_theta_new)
        i += 1
        if i > iterations:
            print('Somthing Wrong!')
            return theta_new
        theta = theta_new
        # Step 1
        e_theta, A, g = Step_1(theta)
        # SSE_theta = np.dot(e_theta.T,e_theta)
        # step 2
        theta_diff, theta_new = Step_2(theta, A, g)
        e_theta_new = e_Cal(theta_new)
        SSE_theta_new = np.dot(e_theta_new.T,e_theta_new)
        SSE_list.append(SSE_theta_new.flatten()[0])


def A_Cal(theta, na, nb, delta, y):
    if na != 0:
        arparams = theta[:na]
    else:
        arparams = np.zeros(nb)
    if nb != 0:
        maparams = theta[na:]
    else:
        maparams = np.zeros(na)

    ar = np.r_[1, arparams]
    ma = np.r_[1, maparams]

    if len(ar) > len(ma):
        ma = np.append(ma,np.zeros(len(ar)-len(ma)))
    if len(ma) > len(ar):
        ar = np.append(ar,np.zeros(len(ma)-len(ar)))

    # generate e(theta1, theta2, ..., thetan)
    system_rev = (ar, ma, 1)
    _, e_theta = signal.dlsim(system_rev, y)

    theta_delta = theta + delta                   # [theta1+delta,theta2+delta,...,thetan+delta]

    for j in range(na+nb):
        if j < na:                                # na>0 or nb=0
            ar_update = ar.copy()
            ar_update[1 + j] = theta_delta[j]     # update ar to [1,theta1,thetai+delta,...,thetan]
            system_rev = (ar_update, ma, 1)
            _, e_theta_delta = signal.dlsim(system_rev, y)  # e(theta1,...,thetai+delta,...,thetan)
            x = (e_theta - e_theta_delta) / delta
            if j == 0:                            # x1
                X = np.asmatrix(x)
            else:                                 # x2,..,xn
                X = np.hstack((X, x))
            j += 1
        else:                                     # na=0 or nb>0
            ma_update = ma.copy()
            ma_update[1 + j - na] = theta_delta[j] # update ma to [1,theta1,thetai+delta,...,thetan]
            system_rev = (ar, ma_update, 1)
            _, e_theta_delta = signal.dlsim(system_rev, y)  # e(theta1,...,thetai+delta,...,thetan)
            x = (e_theta - e_theta_delta) / delta
            if j == 0:
                X = np.asmatrix(x)
            else:
                X = np.hstack((X, x))
            j += 1

    A = X.T.dot(X)
    return A
