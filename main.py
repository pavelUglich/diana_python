import numpy as np
from scipy.interpolate import UnivariateSpline
import functions
from Stabilizer import BoundaryCondition
from secant import find_roots
from voyevodin import VoyevodinMethod

#p = 1

# 1. Частота колебаний
kappa = 2

# 2. Количество точек на отрезках
n_x: int = 5
n_y: int = 5
a = 0
b = 2
c = 0
d = 1

h_x = 2 / n_x  # Длина каждого подотрезка
h_y = 1 / n_y  # Длина каждого подотрезка

x1 = np.zeros(n_x)  # Создание массива из N элементов
x2 = np.zeros(n_y)  # Создание массива из N элементов

for i in range(n_x):
    # Вычисление значения, находящегося в середине каждого подотрезка
    x1[i] = (i + 0.5) * h_x

for i in range(n_y):
    x2[i] = (i + 0.5) * h_y

"""
# Восстанавливаемая плотность
rho1 = np.zeros(n_y)


#def rho(x):
#    """
#    функция распределения плотности (восстанавиваемая)
#    :param x: поперечная координата
#    :return: значение плотности
#    """
#    spl = UnivariateSpline(x2, rho1, k=5)
#    return 1 + spl(x)

"""
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

rp = np.zeros(n_x, dtype=complex)
for i in range(n_x):
    y = (i+0.5)*h_x
    rp[i] = np.sin(3.1415926538*y)

matrix = []
for i in range(n_x):
    row = np.zeros(n_y, dtype=complex)
    for ii in range(n_y):
        x = h_x*(i+0.5)
        y = h_y*(ii+0.5)
        row[ii] = h_x/(1+10*(x-y)**2)
    matrix.append(row)

vm = VoyevodinMethod(matrix, rp, h_x, BoundaryCondition.DIRICHLE, BoundaryCondition.DIRICHLE).solution
print(vm)



# print((f()))

"""
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
#"""