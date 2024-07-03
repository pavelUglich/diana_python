import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline


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
    sol = solve_ivp(sys, [0, 1], y0, args=(kappa, rho, mu, alp), t_eval=x, rtol=1e-6)
    return sol.y


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


def dispersion_equation(alpha, rho, mu, kappa):
    y0 = [0 + 0 * 1j, 1 + 0 * 1j]
    return shoot(shorter_system, kappa, rho, mu, y0, [0, 1], alpha)[1][-1]


# отыскание вычетов первого порядка для отыскания перемещений
def find_residues(roots, rho, mu, kappa):
    result = []
    y0 = [0 + 0 * 1j, 0 + 0 * 1j, 0 + 0 * 1j, 1 + 0 * 1j]
    for i in range(len(roots)):
        solution = shoot(bigger_system, kappa, rho, mu, y0, [1], roots[i])
        result.append(solution[0] / solution[2])
    return result


def displacement(x1, roots, residues):
    result = 0
    for i in range(len(roots)):
        result += 1j * residues[i] * np.exp(1j * roots[i] * x1)
    return result


def array_difference(arr1, arr2):
    if len(arr1) != len(arr2):
        raise IndexError("массивы имеют разную длину")

    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] - arr2[i])

    return result


def the_very_biggest(x, y0, kappa, rho, mu, alp):
    u, du, ddu, dsigma, ddsigma, sigma = y0
    return [sigma / mu(x), dsigma / mu(x), ddsigma / mu(x),
             2 * alp * mu(x) * u + (alp ** 2 * mu(x) - kappa ** 2 * rho(x)) * du,
             2 * mu(x) * u + 4 * alp * mu(x) * du + (alp ** 2 * mu(x) - rho(x) * kappa ** 2) * ddu,
             (alp ** 2 * mu(x) - kappa ** 2 * rho(x)) * u]


def sol_sys6(alpha, x2, rho, mu, kappa):
    y0 = [0 + 0 * 1j, 0 + 0 * 1j, 0 + 0 * 1j, 0 + 0 * 1j, 0 + 0 * 1j, 1 + 0 * 1j]
    return shoot(the_very_biggest, kappa, rho, mu, y0, x2, alpha)  # !!!


def evaluate_b1_b2(roots, rho, mu, kappa):
    b1 = []
    b2 = []
    for i in range(len(roots)):
        solution = sol_sys6(roots[i], [1], rho, mu, kappa)
        b1.append(solution[3])
        b2.append(solution[4])
    return b1, b2


def evaluate_a0_a1(roots, rho, mu, kappa, x2):
    a0 = []
    a1 = []
    y0 = [0 + 0 * 1j, 0 + 0 * 1j, 0 + 0 * 1j, 1 + 0 * 1j]
    for i in range(len(roots)):
        solution = shoot(bigger_system, kappa, rho, mu, y0, x2, roots[i])
        a0.append(solution[0])
        a1.append(solution[1])
    return a0, a1


def matrix_rho(roots, x1, a_0, a_1, b_1, b_2, rows, columns):
    result = np.zeros((rows, columns), dtype=complex)
    for i in range(len(roots)):
        for ii in range(rows):
            for iii in range(columns):
                multiplier = np.exp(x1[iii] * 1j * roots[i])
                item = 1j * (2 * a_0[i][ii] * (a_1[i][ii] - a_0[i][ii] * 0.5 * b_2[i]/b_1[i])
                 + 1j * a_0[i][ii] * a_0[i][ii]*x1[iii]) / b_1[i] / b_1[i]
                result[iii][ii] += item * multiplier
    return result


def process_matrix(matrix, kappa, h_y):
    return matrix * kappa * h_y