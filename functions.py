import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline

kappa = 2

n: int = 20
a = 0
b = 1
c = 0
d = 1

subsegment_length = 1 / n  # Длина каждого подотрезка

x1 = np.zeros(n)  # Создание массива из N элементов
x2 = np.zeros(n)  # Создание массива из N элементов

for i in range(n):
    # Вычисление значения, находящегося в середине каждого подотрезка
    x1[i] = (i + 0.5) * subsegment_length
    x2[i] = (i + 0.5) * subsegment_length

rho1 = np.zeros(n)


def shorter_system(x, y0, kappa, rho, mu, alpha):
    """
    малая система дифференциальных уравнений для решения системы в трансформантах
    :param x: поперечная координата
    :param y0: ???
    :param kappa: частота
    :param rho: функция распределения плотности
    :param mu: функция распределения модуля сдвига
    :param alpha: параметр преобразования Фурье
    :return: набор правых частей системы уравнений
    """
    u, sigma = y0
    return [sigma / mu(x), (alpha ** 2 * mu(x) - kappa ** 2 * rho(x)) * u]


def bigger_system(x, y0, kappa, rho, mu, alpha):
    """
    система, содержащая первые производные по alpha
    :param x: поперечная координата
    :param y0:
    :param kappa: частота колебаниц
    :param rho: распределение плотности
    :param mu: распределение модуля сдвига
    :param alpha:
    :return:
    """
    u, du, dsigma, sigma = y0
    return [sigma / mu(x), dsigma / mu(x), 2 * alpha * mu(x) * u + (alpha ** 2 * mu(x) - kappa ** 2 * rho(x)) * du,
             (alpha ** 2 * mu(x) - kappa ** 2 * rho(x)) * u]


def shoot(sys, kappa, rho, mu, y0, x, alp):
    """
    метод пристрелки
    :param sys: система уравнений
    :param kappa: частота колебаний
    :param rho: распределение плотности
    :param mu: функция распределения модуля сдвига
    :param y0: начальные условия?
    :param x: поперечная координата (параметр не используется)
    :param alp: параметр преобразования Фурье
    :return: решение в точке b
    """
    sol = solve_ivp(sys, [0, 1], y0, args=(kappa, rho, mu, alp), t_eval=x)
    return sol.y


def rho(x):
    """
    функция распределения плотности (восстанавиваемая)
    :param x: поперечная координата
    :return: значение плотности
    """
    #eta = 0.1
    spl = UnivariateSpline(x2, rho1, k=5)
    return 1 + spl(x)


def rho_toch(x):
    """
    точное распределение плотности
    :param x: поперечная координата
    :return: значение плотности
    """
    eta = 0.1
    return 1 + eta * np.sin(np.pi * x)


def mu(x):
    """
    модуль сдвига
    :param x: поперечная координата
    :return: значение
    """
    return 1


def U2(alp, idx):  # x2 - индекс
    """
    :param alp:
    :param idx:
    :return:
    """
    y0 = [0 + 0 * 1j, 1 + 0 * 1j]
    return shoot(shorter_system, kappa, rho, mu, y0, x2, alp)[0][idx]


def dispersion_equation(alpha):
    y0 = [0 + 0 * 1j, 1 + 0 * 1j]
    return shoot(shorter_system, kappa, rho, mu, y0, [0, 1], alpha)[1][-1]


def displacement(x1, alpha, rho, mu):
    result = 0
    y0 = [0 + 0 * 1j, 0 + 0 * 1j, 0 + 0 * 1j, 1 + 0 * 1j]
    for i in range(len(alpha)):
        # def shoot(sys, kappa, rho, mu, y0, x, alp):
        solution = shoot(bigger_system, kappa, rho, mu, y0, [0, 1], alpha[i])
        term = solution[0][-1] / solution[2][-1]
        result += 1j * term * np.exp(1j * alpha[i] * x1)
    return result


# print(u(0, 0))

def sys6(x, y0, kappa, rho, mu, alp):
    u, du, ddu, dsigma, ddsigma, sigma = y0
    return [sigma / mu(x), dsigma / mu(x), ddsigma / mu(x),
             2 * alp * mu(x) * u + (alp ** 2 * mu(x) - kappa ** 2 * rho(x)) * du,
             2 * mu(x) * u + 4 * alp * mu(x) * du + (alp ** 2 * mu(x) - rho(x) * kappa ** 2) * ddu,
             (alp ** 2 * mu(x) - kappa ** 2 * rho(x)) * u]


def sol_sys6(alp):
    y0 = [0 + 0 * 1j, 0 + 0 * 1j, 0 + 0 * 1j, 0 + 0 * 1j, 0 + 0 * 1j, 1 + 0 * 1j]
    return shoot(sys6, kappa, rho, mu, y0, x2, alp)  # !!!


def I(x1, idx, alpha):  # ksi индекс
    s = 0
    for i in range(len(alpha)):
        sol = sol_sys6(alpha[i])
        a0 = sol[0][idx]
        a1 = sol[1][idx]
        b1 = sol[3][-1]
        b2 = 0.5 * sol[4][-1]
        s += b1 ** (-2) * (2 * a0 * (a1 - a0 * b1 ** (-1) * b2) - 1j * x1 * a0 ** 2) * np.exp(1j * alpha[i] * x1)
    return 1j * s


def right_part():
    f = [u_toch(x1[i], -1).real - u(x1[i], -1).real for i in range(n)]
    return f


def A1(n, alpha):
    A = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            A[i][j] = kappa ** 2 * I(x1[i], j, alpha).real
            print("A[", i, ",", j, "] = ", A[i][j])  #!!!!
    return A

