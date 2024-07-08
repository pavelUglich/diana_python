from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import functions
from secant import find_roots

#p = 1

# 1. Частота колебаний
kappa = 2

# 2. Количество точек на отрезках
n_x: int = 20
n_y: int = 20
a = 0
b = 1
c = 0
d = 1

h_x = 1 / n_x  # Длина каждого подотрезка
h_y = 1 / n_y  # Длина каждого подотрезка

x1 = np.zeros(n_x)  # Создание массива из N элементов
x2 = np.zeros(n_y)  # Создание массива из N элементов

for i in range(n_x):
    # Вычисление значения, находящегося в середине каждого подотрезка
    x1[i] = (i + 0.5) * h_x

for i in range(n_y):
    x2[i] = (i + 0.5) * h_y

# Восстанавливаемая плотность
rho1 = np.zeros(n_y)


def rho(x):
    """
    функция распределения плотности (восстанавиваемая)
    :param x: поперечная координата
    :return: значение плотности
    """
    spl = UnivariateSpline(x2, rho1, k=5)
    return 1 + spl(x)


# построение двух дисперсионных уравнений
equation1 = lambda xx: functions.dispersion_equation(xx, rho, functions.mu, kappa)
equation2 = lambda xx: functions.dispersion_equation(xx, functions.rho_toch, functions.mu, kappa)

# отыскание корней в неоднородном и однородном случаях
roots = find_roots(equation1)
roots_toch = find_roots(equation2)

# отыскание вычетов в неоднородном и однородном случаях
residues = functions.find_residues(roots, rho, functions.mu, kappa)
residues_toch = functions.find_residues(roots, functions.rho_toch, functions.mu, kappa)


# построение правой части
wavefield: list[int] = []
wavefield_toch: list[int] = []
for i in range(n_x):
    wavefield.append(functions.displacement(x1[i], roots, residues))
    wavefield_toch.append(functions.displacement(x1[i], roots_toch, residues_toch))
rp = functions.array_difference(wavefield, wavefield_toch)

# print(roots, len(roots))
# print(roots_toch, len(roots_toch))

# построение матрицы
# отыскание b_1, b_2
b_1, b_2 = functions.evaluate_b1_b2(roots, rho, functions.mu, kappa)
a_0, a_1 = functions.evaluate_a0_a1(roots, rho, functions.mu, kappa, x2)
A = functions.matrix_rho(roots, x1, a_0, a_1, b_1, b_2, n_x, n_y)
A = functions.process_matrix(A, kappa, h_y)




"""
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

    # РЎС‚СЂРѕРёРј РјР°С‚СЂРёС†Сѓ C1
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

# РёС‚РµСЂ РїСЂРѕС†
##
for i in range(1, 10):
    print("-------------------------", i)
    roots = find_roots(SIGMA2)
    print('alp_n', roots, len(roots))
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
"""