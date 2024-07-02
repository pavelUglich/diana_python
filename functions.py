import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline

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

def shorter_system(x, y0, kappa, rho, mu, alp):
    """
    малая система дифференциальных уравнений для решения системы в трансформантах
    :param x: поперечная координата
    :param y0: ???
    :param kappa: частота
    :param rho: функция распределения плотности
    :param mu: функция распределения модуля сдвига
    :param alp: параметр преобразования Фурье
    :return: набор правых частей системы уравнений
    """
    u, sigma = y0
    dF_dx = [sigma / mu(x), (alp ** 2 * mu(x) - kappa ** 2 * rho(x)) * u]
    return dF_dx


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
