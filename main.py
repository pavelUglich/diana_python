import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import functions


#x2[n]=1.0

kappa = 2
p = 1
alp = 0
alp1 = np.linspace(0, 160, 500)

# alp = 1
x = functions.x2



def U2(alp, x2):  # x2 - индекс
    y0 = [0 + 0 * 1j, p + 0 * 1j]
    return functions.shoot(functions.shorter_system, kappa, rho, mu, y0, x, alp)[0][x2]


def SIGMA2(alp):
    y0 = [0 + 0 * 1j, p + 0 * 1j]
    return functions.shoot(functions.shorter_system, kappa, functions.rho, functions.mu, y0, x, alp)[1][-1]


def U2_toch(alp, x2):  # x2 - индекс
    y0 = [0 + 0 * 1j, p + 0 * 1j]
    return functions.shoot(functions.shorter_system, kappa, rho_toch, mu, y0, x, alp)[0][x2]


def SIGMA2_toch(alp):  # x2 - индекс
    y0 = [0 + 0 * 1j, p + 0 * 1j]
    return functions.shoot(functions.shorter_system, kappa, rho_toch, mu, y0, x, alp)[1][-1]


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


alp_n = find_alp(SIGMA2)
alp_n_toch = find_alp(SIGMA2_toch)
print(alp_n, len(alp_n))
print(alp_n_toch, len(alp_n_toch))


def sys4(x, y0, kappa, rho, mu, alp):
    u, du, dsigma, sigma = y0
    dF_dx = [sigma / mu(x), dsigma / mu(x), 2 * alp * mu(x) * u + (alp ** 2 * mu(x) - kappa ** 2 * rho(x)) * du,
             (alp ** 2 * mu(x) - kappa ** 2 * rho(x)) * u]
    return dF_dx


def dSIGMA2(alp):
    y0 = [0 + 0 * 1j, 0 + 0 * 1j, 0 + 0 * 1j, p + 0 * 1j]
    return shoot(sys4, kappa, rho, mu, y0, x, alp)[2][-1]  # !!!


def dSIGMA2_toch(alp):
    y0 = [0 + 0 * 1j, 0 + 0 * 1j, 0 + 0 * 1j, p + 0 * 1j]
    return shoot(sys4, kappa, rho_toch, mu, y0, x, alp)[2][-1]  # !!!


def u(x1, x2):
    return 1j * sum(U2(alp_n[i], x2) / dSIGMA2(alp_n[i]) * np.exp(1j * alp_n[i] * x1) for i in range(len(alp_n)))


def u_toch(x1, x2):
    return 1j * sum(U2_toch(alp_n_toch[i], x2) / dSIGMA2_toch(alp_n_toch[i]) * np.exp(1j * alp_n_toch[i] * x1) for i in
                    range(len(alp_n_toch)))


# print(u(0, 0))

def sys6(x, y0, kappa, rho, mu, alp):
    u, du, ddu, dsigma, ddsigma, sigma = y0
    dF_dx = [sigma / mu(x), dsigma / mu(x), ddsigma / mu(x),
             2 * alp * mu(x) * u + (alp ** 2 * mu(x) - kappa ** 2 * rho(x)) * du,
             2 * mu(x) * u + 4 * alp * mu(x) * du + (alp ** 2 * mu(x) - rho(x) * kappa ** 2) * ddu,
             (alp ** 2 * mu(x) - kappa ** 2 * rho(x)) * u]
    return dF_dx


def sol_sys6(alp):
    y0 = [0 + 0 * 1j, 0 + 0 * 1j, 0 + 0 * 1j, 0 + 0 * 1j, 0 + 0 * 1j, p + 0 * 1j]
    return functions.shoot(sys6, kappa, rho, mu, y0, x, alp)  # !!!


##
##def a0(alp, ksi): #ksi индекс
##    return sol_sys6(alp)[0][ksi]
##
##
##def a1(alp, ksi): #ksi индекс
##    return sol_sys6(alp)[1][ksi]
##
##def b1(alp): #ksi индекс
##    return sol_sys6(alp)[3][-1]
##
##def b2(alp): #ksi индекс
##    return 0.5 * sol_sys6(alp)[4][-1]
##
##def I(x1, ksi):
##    return p * 1j * sum( (b1(alp_n[i])**(-2) * (2 * a0(alp_n[i], ksi) * (a1(alp_n[i], ksi) - a0(alp_n[i], ksi) * b1(alp_n[i])**(-1) * b2(alp_n[i])) - 1j * x1 * a0(alp_n[i], ksi)**2))
##                         * np.exp(1j * alp_n[i] * x1) for i in range(len(alp_n)))
def I(x1, ksi):  # ksi индекс
    s = 0
    for i in range(len(alp_n)):
        sol = sol_sys6(alp_n[i])
        a0 = sol[0][ksi]
        a1 = sol[1][ksi]
        b1 = sol[3][-1]
        b2 = 0.5 * sol[4][-1]
        s += b1 ** (-2) * (2 * a0 * (a1 - a0 * b1 ** (-1) * b2) - 1j * x1 * a0 ** 2) * np.exp(1j * alp_n[i] * x1)
    return 1j * s


def f():
    f = [u_toch(x1[i], -1).real - u(x1[i], -1).real for i in range(n)]
    return f


def A1():
    A = np.zeros((functions.n, functions.n), dtype=complex)
    for i in range(functions.n):
        for j in range(functions.n):
            A[i][j] = kappa ** 2 * I(functions.x1[i], j).real
            print("A = ", A[i][j])  #!!!!
    return A


A = A1()


def metod_Tihonova(A, f, a, b, c, d):
    N = m = n

    h_s = (b - a) / (n - 1)  # x2
    h_x = (d - c) / (m - 1)  # x1

    x = [c + (i - 1) * h_x for i in range(1, m + 1)]
    s = [a + (j - 1) * h_s for j in range(1, n + 1)]

    u = f()

    B = np.zeros((N, N), dtype=complex)

    for j in range(n):
        for k in range(n):
            B[j][k] = h_x * h_s * sum(A[i][k] * A[i][j] for i in range(m))

    F = [h_x * sum(u[i] * A[i][j] for i in range(m)) for j in range(n)]

    # Строим матрицу C1
    C1 = [[1 / (h_s) ** 2], [-1 / (h_s) ** 2, 2 / (h_s) ** 2]]
    for i in range(N - 2):
        c1 = [-1 / h_s ** 2, 2 / h_s ** 2]
        for j in range(i + 1):
            c1.insert(0, 0)
        C1.extend([c1])
    df = pd.DataFrame(C1)
    C1 = df.combine_first(df.T)
    C1[N - 1][N - 1] /= 2
    C = np.eye(N) + C1

    # print(iters, f())

    ##################
    delta = 1e-5
    for i in range(N):
        for j in range(N):
            A[i][j] *= 1
            # print(x, s, A)
    h = 1e-3
    alpha = 1e-1
    alpha1 = alpha / 2
    epsilon = 1e-5
    Nit = 0
    um = []
    um1 = []
    um2 = []
    for i in range(n):
        um.append(0)
        um1.append(0)
        um2.append(0)

    while (Nit < 50) and (abs(alpha - alpha1) > epsilon):

        Nit += 1
        B_alpha = B + alpha * C
        # print(B_alpha, F)
        Ualpha = np.linalg.solve(B_alpha, F)
        B_alpha = B + alpha1 * C
        Ualpha1 = np.linalg.solve(B_alpha, F)
        for i in range(n):
            um[i] = Ualpha[i]
            um1[i] = Ualpha1[i]
        roi_temp = 0
        for j0 in range(m):
            roi_temp += (sum((h_s * A[j0][i0] * um[i0]) for i0 in range(n)) - u[j0]) ** 2
        roi = h_x * roi_temp - (delta + h * (sum(um[i] ** 2 for i in range(n))) ** 0.5) ** 2
        roi = roi.real
        roi_temp = 0
        for j1 in range(m):
            roi_temp += (sum(h_s * A[j1][i1] * um1[i1] for i1 in range(n)) - u[j1]) ** 2
        roi1 = h_x * roi_temp - (delta + h * (sum(um1[i] ** 2 for i in range(n))) ** 0.5) ** 2
        roi1 = roi1.real
        alpha2 = alpha / (1 - (alpha1 - alpha) * roi / (alpha1 * (roi - roi1)))
        alpha2 = alpha2.real
        B_alpha = B + alpha2 * C
        Ualpha2 = np.linalg.solve(B_alpha, F)
        for i in range(n):
            um2[i] = Ualpha2[i]
        roi_temp = 0
        for j2 in range(m):
            roi_temp += (sum(h_s * A[j2][i2] * um2[i2] for i2 in range(n)) - u[j2]) ** 2
        roi2 = h_x * roi_temp - (delta + h * (sum(um2[i] ** 2 for i in range(n))) ** 0.5) ** 2
        roi2 = roi2.real
        if ((roi * roi1) > 0):
            alpha = alpha1
            alpha1 = alpha2

        if (roi * roi1) < 0:
            if (roi * roi2) < 0:
                alpha1 = alpha2
        elif (roi1 * roi2) < 0:
            alpha = alpha2
    ##################

    B_alpha = B + alpha * C

    return (np.linalg.solve(B_alpha, F))


# print((f()))
rho1 = metod_Tihonova(A, f, a, b, c, d)
rho3 = UnivariateSpline(x, rho1, k=5)
print(rho1)
##
plt.plot(x, rho_toch(x))
plt.plot(x, rho3(x) + rho(x))
plt.show()

# итер проц
##
for i in range(1, 10):
    print("-------------------------", i)
    alp_n = find_alp(SIGMA2)
    print('alp_n', alp_n, len(alp_n))
    A = A1()
    rho2 = metod_Tihonova(A, f, a, b, c, d)
    for j in range(n):
        rho1[j] += rho2[j].real

    print(rho1)

rho11 = UnivariateSpline(x, rho1, k=5)
plt.plot(x, rho_toch(x))
plt.plot(x, rho11(x) + rho(x))
plt.show()

##
##intergal = []
##for i in range(n):
##    h = 1/(n - 1)
##    s = 0
##    for j in range(0, n):
##        s += h * A[i][j] * rho1[j]
##    intergal.append(s)
##print('intergal', intergal)
##print('f', f())
