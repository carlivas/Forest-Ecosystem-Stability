import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde, cauchy, norm
from scipy.optimize import curve_fit
import cv2

# https://github.com/earlywarningtoolbox/spatial_warnings/blob/master/rspec_ews.R


# initialization fcts
@njit
def init(mu_, shape, mean=1):
    M, N = shape
    p = 0.5*(1 + mu_ / mean)
    x_ = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            if p > np.random.random():
                x_[i, j] = np.random.normal(mean, 0.05)
            else:
                x_[i, j] = - np.random.normal(mean, 0.05)
    return x_


@njit
def init_split(mu_, shape, mean=1.2):
    M, N = shape
    p = 0.5*(1 + mu_ / mean)
    sig_low = 0.1
    sig_high = 0.1

    if p > 0.5:
        sig_low = 0.5
    else:
        sig_high = 0.5

    x_ = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            if p > np.random.random():
                x_[i, j] = np.random.normal(mean,
                                            sig_high)
            else:
                x_[i, j] = - np.random.normal(mean, sig_low)
    return x_


# plotting
def plot_dens(x, y, ax, label=None, cmap=" Greens ", c=None):
    xy = np.vstack([x, y])
    dens = gaussian_kde(xy)(xy)
    idens = dens.argsort()
    x_, y_, dens = x[idens], y[idens], dens[idens]
    i0 = int(idens.size / 2)
    ax.scatter(x_[i0], y_[i0], s=10, c=c, label=label)
    ax.scatter(x_, y_, c=dens, s=10, cmap=cmap)


def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.
        line : Line2D object
    position : x- position of the arrow.If None, mean of xdata is taken
    direction : ’left ' or ’right '
    size : size of the arrow in fontsize points
    color : if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()

    # find closest index
    start_ind = np.argmin(np.abs(xdata - position))

    if direction == 'right ':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1
    line.axes.annotate('', xytext=(xdata[start_ind], ydata[start_ind]), xy=(
        xdata[end_ind], ydata[end_ind]), arrowprops=dict(arrowstyle=" ->", color=color), size=size)


def plot_2d_fft(state, figax=None):
    M, N = state.shape
    if figax == None:
        figax = plt.subplots()
    fftcen = np.fft.fftshift(np.abs(np.fft.fft2(state)))
    fftcen[M // 2, N // 2] = 0
    fftcen = fftcen / np.sum(fftcen)
    fftcen = fftcen[M // 2:, N // 2:]
    im2 = figax[1].imshow(fftcen, norm=LogNorm(
        vmin=None, vmax=None), cmap=" hot", origin=" lower ")
    figax[0].colorbar(im2, ax=figax[1], label=" power (a.u.)")


def plot_hysteris(Rvals, mus, ax_=None, colors=("k", "r"), label=None):
    # fairly bad
    if ax_ == None:
        fig_, ax_ = plt.subplots(1, 1, dpi=200, figsize=(4, 3))

    N_5 = int(Rvals.size / 5)
    add_arrow(ax_.plot(Rvals[: N_5], mus[: N_5], colors[0], label=label)[0])
    add_arrow(ax_.plot(Rvals[N_5:3 * N_5], mus[N_5:3 * N_5], colors[1])[0])
    add_arrow(ax_.plot(Rvals[3 * N_5:], mus[3 * N_5:], colors[0])[0])

    ax_.axhline(0, ls=" --", c="k", alpha=0.2)
    ax_.axvline(0, ls=" --", c="k", alpha=0.2)
    ax_.set(xlabel="R", ylabel=r"$\mu$ ")

 # ecosystem evolution


@ njit
def time_evo_dif2(x, lamb, dif, dif2, R):
    M, N = x.shape
    return np.asarray([[(lamb - 1) * x[i, j] - x[i, j]**3 + R
                        - 2 * dif * (-4 * x[i, j] + x[i - 1, j] + x[(i + 1) % M, j]
                                     + x[i, j - 1] + x[i, (j + 1) % N])
                        - dif2 * (20 * x[i, j]
                                  - 8*(x[i - 1, j] + x[(i + 1) % M, j] +
                                       x[i, j - 1] + x[i, (j + 1) % N])
                                  + 2*(x[i - 1, j - 1] + x[(i + 1) % M, j - 1] +
                                       x[i - 1, (j + 1) % N] + x[(i + 1) % M, (j + 1) % N])
                                  + x[(i + 2) % M, j] + x[i, (j + 2) % N] + x[i - 2, j] + x[i, j - 2])
                        for j in range(N)] for i in range(M)])


@ njit
def equi(state, lamb, dif, dif2, R_, sigma, dt=0.01, numt=5000):
    M, N = state.shape
    xt = np.copy(state)
    mean_arr = np.empty(numt)
    for l in range(numt):
        noise = np.random.normal(loc=0.0, scale=1.0, size=(M, N))
        mean_arr[l] = np.mean(xt)
        xt += dt * time_evo_dif2(xt, lamb, dif, dif2, R_) + sigma * noise
    return (np.mean(mean_arr[-1000:]), xt)


@ njit
def equi_var(state, lamb, dif, dif2, R_, sigma, dt=0.01, numt=40_000, eps=1e-8):
    M, N = state.shape
    xt = np.copy(state)
    mean_arr = np.empty(numt)
    Flag = True
    for l in range(numt):
        noise = np.random.normal(loc=0.0, scale=1.0, size=(M, N))
        mean_arr[l] = np.mean(xt)
        xt += dt * time_evo_dif2(xt, lamb, dif, dif2, R_ + sigma * noise)

    while np.std(mean_arr[-10_000:]) > eps:
        noise = np.random.normal(0.0, 1.0, size=(M, N))
        xt += dt * time_evo_dif2(xt, lamb, dif, dif2, R_ + sigma * noise)
        mean_arr = np.append(mean_arr, np.mean(xt))
        numt += 1
        if numt > 200_000:
            print(R_, " equilibrium not reached ")
            Flag = False
            break
    print(numt)
    return mean_arr, xt, Flag, numt


@ njit
def equi_var2(state, R_, lamb, dif, dif2, sigma, dt=0.01, numt=40_000, eps=1e-8):
    M, N = state.shape
    xt = np.copy(state)
    mean_arr = np.empty(numt)
    Flag = True

    for l in range(numt):
        noise = np.random.normal(loc=0.0, scale=1.0, size=(M, N))
        mean_arr[l] = np.mean(xt)
        xt += dt * time_evo_dif2(xt, lamb, dif, dif2, R_) + sigma * noise

    while np.std(mean_arr[-10_000:]) > eps:
        noise = np.random.normal(loc=0.0, scale=1.0, size=(M, N))
        xt += dt * time_evo_dif2(xt, lamb, dif, dif2, R_) + sigma * noise
        mean_arr = np.append(mean_arr, np.mean(xt))
        numt += 1
        if numt > 200_000:
            print(R_, " equilibrium not reached ")
            Flag = False
            break
    if numt > 150_000:
        print(R_, numt)
    return np.mean(mean_arr[-1000:]), xt, Flag


@ njit
def calc_hysterisis2(initial_state, R_max, lamb, dif, dif2, sigma, Npts=125):
    M, N = initial_state.shape
    dR = 5 * R_max / Npts
    R_arr = np.empty(Npts)
    R_arr[: int(Npts / 5)] = np.linspace(0., R_max - dR, int(Npts / 5))
    R_arr[int(Npts / 5):int(2 * Npts / 5)
          ] = np.flip(np.linspace(dR, R_max, int(Npts / 5)))
    R_arr[int(2 * Npts / 5):int(4 * Npts / 5)] = - \
        np.flip(R_arr[: int(2 * Npts / 5)])
    R_arr[int(4 * Npts / 5):] = np.linspace(0, R_max - dR, int(Npts / 5))

    mu = np.empty(Npts)
    moran = np.empty(Npts)  # mainnu = np.empty (Npts)
    special_var = np.empty(Npts)
    mask = (mu > -100)
    x = np.copy(initial_state)
    # x = np.empty ((M, N, Npts +1))
    # nur = np.fft.fftfreq ((M //2+1) *2) [:(M //2+1) ]

    for i, R in enumerate(R_arr):
        mu[i], x, mask[i] = equi_var2(x, R, lamb, dif, dif2, sigma)
        moran[i] = moranI(x)
        special_var[i] = np.var(x)
        # mainnu [i] = nur[np.argmax (rspec_ews (x) [1]) ]
    return (R_arr, mask, mu, moran, special_var)


def calc_hysterisis3(initial_state, R_max, lamb, dif, dif2, sigma, R_start=0, dir="+", Npts=100):
    M, N = initial_state.shape
    dR = 4 * R_max / Npts
    R_arr = np.empty(Npts)
    R_arr[: int(Npts / 2)] = np.flip(np.linspace(- R_max +
                                                 dR, R_max, int(Npts / 2)))
    R_arr[int(Npts / 2):] = np.linspace(- R_max, R_max - dR, int(Npts / 2))
    index = np.argwhere(np.round(R_arr[: int(Npts / 2)], 4) == R_start)[0, 0]
    R_arr = np.roll(R_arr, index)
    if dir == "-":
        R_arr = np.flip(R_arr)
        R_arr = np.roll(R_arr, 1)

    mu = np.empty(Npts)
    moran = np.empty(Npts)
    special_var = np.empty(Npts)
    r_sign = np.empty(Npts)
    th_sign = np.empty(Npts)
    Nspots = np.empty(Npts)
    Ndots = np.empty(Npts)

    mask = (mu > -100)
    x = np.copy(initial_state)

    for i, R in enumerate(R_arr):
        mu[i], x, mask[i] = equi_var2(x, R, lamb, dif, dif2, sigma)
        moran[i] = moranI(x)
        special_var[i] = 235
        Nspots[i] = patchnumber(x, "+") - 1
        Ndots[i] = patchnumber(x, "-") - 1


return (R_arr, mask, mu, moran, special_var, r_sign, th_sign, Nspots, Ndots)


@ njit
def evolve(state, numt, Rs, lamb, dif, dif2, sigma, dt=0.01):
    xt = np.copy(state)
    for t in range(numt):
        noise = np.random.normal(loc=0.0, scale=1.0, size=(M, N))
        xt += dt * time_evo_dif2(xt, lamb, dif, dif2, Rs[t]) + sigma * noise
    return xt


def run_down(state, R_range, lamb, dif, dif2, sigma, dt=0.01, numt=1e6, Nvals=200):
    Nvals = int(Nvals)
    numt = int(numt)
    dR = np.abs((R_range[0] - R_range[1]) / numt)

    if R_range[0] < R_range[1]:
        R_t = np.linspace(R_range[0] + dR, R_range[1], numt)
    else:
        R_t = np.linspace(R_range[1], R_range[0] - dR, numt)
        R_t = np.flip(R_t)

    R0 = np.empty(Nvals + 1)
    properties = np.empty((8, Nvals + 1))

    xt = np.copy(state)
    evo_time = int(numt / (Nvals))

    rsign, thsign = peak_sign(xt)
    Nlow, As_low = patchsizes(xt, "-")
    Nhigh, As_high = patchsizes(xt, "+")
    properties[:, 0] = np.array([R_range[0], np.mean(
        xt), moranI(xt), np.var(xt), rsign, thsign, Nlow, Nhigh])
    areasL = np.empty(0)
    areasL = np.append(areasL, As_low)
    areasH = np.empty(0)
    areasH = np.append(areasH, As_high)
    # evolve for some time
    for k in range(0, Nvals):
        R_intv = R_t[k * evo_time:(k + 1) * evo_time]
        xt = evolve(xt, evo_time, R_intv, lam, dif, dif2, sigma, dt=dt)
        rsign, thsign = peak_sign(xt)
        Nlow, As_low = patchsizes(xt, "-")
        Nhigh, As_high = patchsizes(xt, "+")
        properties[:, k + 1] = np.array([R_t[(k + 1) * evo_time - 1], np.mean(
            xt), moranI(xt), np.var(xt), rsign, thsign, Nlow, Nhigh])
    areasL = np.append(areasL, As_low)
    areasH = np.append(areasH,
                       As_high)
    return properties, xt, areasL, areasH


# spatial early warnig signals
@ njit
def moranI(state):
    mean = np.mean(state)
    M, N = state.shape
    moranI = 0
    for i in range(M):
        for j in range(N):
            moranI += (state[i, j] - mean) * (state[i, j - 1] + state[i, (j + 1) %
                                                                      N] + state[i - 1, j] + state[(i + 1) % M, j] - 4 * mean)
    if np.var(state) != 0:
        moranI /= (4 * np.var(state) * N * M)
    else:
        moranI = 1
    return moranI


def rspec_ews(state):
    M, N = state.shape
    n0x = N // 2 + 1
    n0y = M // 2 + 1

    # Create distance and angle matrices
    x = np.tile(np.arange(1, N + 1), (M, 1)) - n0x
    y = np.tile(np.arange(1, M + 1), (N, 1)).T - n0y
    DIST = np.sqrt(x ** 2 + y ** 2)
    ANGLE = np.arctan2(-y, x) * 180 / np.pi

    # Calculate DFT
    mi = 1
    ma = min(n0x, n0y)
    DISTMASK = (DIST >= mi) & (DIST <= ma)
    tmp = np.fft.fftshift(np.fft.fft2(state))
    tmp[n0y - 1, n0x - 1] = 0
    aspectr2D = np.abs(tmp) / (n0x * n0y) ** 2
    aspectr2D /= np.sum(aspectr2D[DISTMASK])

    # Now calculate r- spectrum
    STEP = 1
    ray = np.arange(mi, ma + 1, STEP)
    rspectr = np.zeros(len(ray))
    for i in range(len(ray)):
        m = (DIST >= ray[i] - STEP / 2) & (DIST < ray[i] + STEP / 2)
        rspectr[i] = np.mean(aspectr2D[m])

    # Now calculate theta - spectrum
    STEP = 5  # increments of 5 degrees
    anglebin = np.arange(STEP, 181, STEP)
    tspectr = np.zeros(len(anglebin))
    for i in range(len(tspectr) - 1):
        m = np.where(DISTMASK & (
            ANGLE >= anglebin[i] - STEP) & (ANGLE < anglebin[i]))
        tspectr[i] = np.sum(aspectr2D[m]) / len(m[0])
    m = np.where(DISTMASK & (
        ANGLE >= anglebin[-1] - STEP) & (ANGLE <= anglebin[-1]))
    tspectr[-1] = np.sum(aspectr2D[m]) / len(m[0])
    return tspectr, rspectr


def patchsizes(im, sign="-"):
    """
    Parameters
    ----------
    im : np.ndarray
    patterned state centered around 0.
    sign : str, optional
    sign of the background.The default is " -".
    Returns
    -------
    numL : float
    number of patches.
    areas : np.ndarray
    areas of the patches (unordered).
    labels : np.ndarray
    greyscale image with labeled patches.
    """
    M, N = im.shape
    if sign == "+":
        grey_im = np.zeros((M, N))
        grey_im[im > 0] = 1
    elif sign == "-":
        grey_im = np.zeros((M, N))
        grey_im[im < 0] = 1
    numL, labels = cv2.connectedComponents(np.uint8(grey_im))
    lab_list = np.arange(numL)
    # taking care of periodic boundary conditions
    for i in range(M):
        label_l = labels[i, 0]
        label_r = labels[i, -1]
        # rewrite labels
        if (label_l != 0) and (label_r != 0) and (label_l != label_r):
            labels[label_r == labels] = label_l
            lab_list = lab_list[lab_list != label_r]
            numL -= 1

    for j in range(N):
        label_l = labels[0, j]
        label_r = labels[-1, j]
        # rewrite labels
        if (label_l != 0) and (label_r != 0) and (label_l != label_r):
            labels[label_r == labels] = label_l
            lab_list = lab_list[lab_list != label_r]
            numL -= 1

    areas = np.empty(numL, dtype=int)
    for l in range(numL):
        areas[l] = labels[labels == lab_list[l]].size
    # print (areas, lab_list)
    return numL, areas, labels


def null_model(state, Nex=200):
    M, N = state.shape
    red = np.random.normal(np.mean(state), np.std(state), (Nex, M, N))
    spec = np.empty((Nex, int(M / 2+1)))
    for i, noise in enumerate(red):
        spec[i] = rspec_ews(noise)[1] / np.sum(rspec_ews(noise)[1])
    low = np.percentile(spec, 5, axis=0)
    high = np.percentile(spec, 95, axis=0)
    return low, high, np.std(spec, axis=0, ddof=1)


def peak_test(spec):
    p = 2 * norm.sf((np.max(spec) - np.mean(spec)) /
                    np.std(spec, ddof=1), 0, 1)
    # if p < 1e -5: print (f" significant peak.p = {p *100:.2 g} %")
    return p, np.argmax(spec)


def peak_sign(state):
    M, N = state.shape
    k = np.arange(1, int(M / 2+2))
    thspec, rspec = rspec_ews(state)[1]
    rspec /= np.sum(rspec)
    thspec /= np.sum(thspec)

    try:
        parsr, _ = curve_fit(lambda x, p, b, mu, sig: ((1 - p) * 1 / b * np.e ** (-(
            x - 1) / b) + p * cauchy.pdf(x, mu, sig)), k, rspec, p0=[0.7, 1, 5, 0.5])
    except:
        parsr = np.array([-10, 0, 0, 0])

    p = 2 * norm.sf((np.max(thspec) - np.mean(thspec)) /
                    np.std(thspec, ddof=1), 0, 1)
    return parsr[0], p
