import numpy as np

alp = 0
alp1 = np.linspace(0, 160, 500) # !!!!!!! сделать разные диапазоны


def secant(f, x0, eps: float = 1e-7, kmax: int = 1e3) -> float:
    """
    solves f(x) = 0 by secant method with precision eps
    :param f: f
    :param x0: starting point
    :param eps: precision wanted
    :return: root of f(x) = 0
    """
    x, x_prev, i = x0, x0 + 2 * eps, 0

    while abs(x - x_prev) >= eps and i < kmax:
        x, x_prev, i = x - f(x) / (f(x) - f(x_prev)) * (x - x_prev), x, i + 1

    return x


def find_alp(f):
    alp_real = []
    alp_im = []
    real_roots = []
    im_roots = []
    f_real = [f(alp1[i]) for i in range(len(alp1))]
    f_im = [f(alp1[i] * 1j) for i in range(len(alp1))]
    for i in range(1, len(alp1)):
        if f_real[i - 1] * f_real[i] < 0:
            alp_real.append(alp1[i])
        if f_im[i - 1] * f_im[i] < 0:
            alp_im.append(alp1[i] * 1j)

    for i in range(len(alp_real)):
        real_roots.append(secant(f, alp_real[i]).real)

    for i in range(len(alp_im)):
        im_roots.append(secant(f, alp_im[i]).imag * 1j)

    return real_roots + im_roots
