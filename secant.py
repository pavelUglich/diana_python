import numpy as np

alp = 0
alp1 = np.linspace(0, 90, 45) # !!!!!!! сделать разные диапазоны



def secant_method(f, x0, x1, tol=1e-7, max_iter: int = 20):
    """
    Функция для нахождения корня уравнения f(x)=0 с помощью метода секущих.

    Аргументы:
    f : функция
        Функция, корень уравнения f(x)=0 которой мы ищем.
    x0, x1 : float
        Начальные точки для метода секущих.
    tol : float
        Точность вычисления (критерий останова).
    max_iter : int
        Максимальное число итераций.

    Возвращает:
    x : float
        Найденное приближение корня уравнения f(x)=0.
    """

    iter_count = 0
    while iter_count < max_iter:
        # Вычисляем новую точку методом секущих
        x_new = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))

        # Проверяем критерий останова (точность)
        if abs(x_new - x1) < tol:
            return x_new

        # Переходим к следующей итерации
        x0 = x1
        x1 = x_new
        iter_count += 1

    # Если не достигли необходимой точности за заданное число итераций, выводим ошибку
    raise ValueError(f"Метод секущих не сошелся к заданной точности {tol} после {max_iter} итераций.")


def find_roots(f):
    real_roots = []
    im_roots = []
    f_real = [f(alp1[i]) for i in range(len(alp1))]
    f_im = [f(alp1[i] * 1j) for i in range(len(alp1))]
    for i in range(1, len(alp1)):
        if f_real[i - 1] * f_real[i] < 0:
            real_roots.append(secant_method(f, alp1[i-1], alp1[i]).real)
        if f_im[i - 1] * f_im[i] < 0:
            im_roots.append(secant_method(f, alp1[i-1]*1j, alp1[i]*1j).imag * 1j)
    return real_roots + im_roots
