import numpy as np

from Stabilizer import Stabilizer


class MatrixSystem:
    # ms = MatrixSystem(matrix, right_part, step, h, delta, eps)

    def __init__(self, matrix, right_part, step, h, p, left, right):
        self.__matrix = matrix
        self.__right_part = right_part
        self.__step = step
        self.__h = h
        self.__rows = len(matrix)
        self.__columns = len(matrix[0])
        self.__stabilizer = Stabilizer(self.__columns, step, p, left, right)
        self.__multipy_asinv()
        self.__multiply_transpose_au()
        self.__qpr()
        self.__multiply_rx()

    def __multipy_asinv(self) -> None:
        diagonal = self.__stabilizer.diagonal
        up_diagonal = self.__stabilizer.up_diagonal
        for i in range(len(self.__matrix)):
            self.__matrix[i][0] /= diagonal[i]
        for i in range(1, len(self.__matrix[0])):
            for ii in range(len(self.__matrix)):
                self.__matrix[ii][i] -= up_diagonal[i - 1] * self.__matrix[ii][i - 1]
                self.__matrix[ii][i] /= diagonal[i]
        return

    def __multiply_transpose_au(self) -> None:
        v = np.zeros(self.__columns, dtype=complex)
        for i in range(self.__columns):
            for ii in range(self.__rows):
                v[i] += self.__matrix[ii][i].conjugate() * self.__right_part[ii]
        self.__right_part = v

    def __qpr(self) -> None:
        size = self.__rows if self.__rows < self.__columns else self.__columns
        self.__p1 = []
        self.__p2 = []
        for i in range(size):
            self.__del_col(i)
            self.__del_row(i)

    def __del_col(self, k) -> None:
        if k >= self.__columns or k >= self.__rows:
            return
        l: int = self.__rows - k
        av = np.zeros(l, dtype=complex)
        for i in range(l):
            av[i] = self.__matrix[i + k][k]
        av[0] -= np.linalg.norm(av) * av[0] / abs(av[0])
        av /= np.linalg.norm(av)
        vv = np.zeros(l, dtype=complex)
        for i in range(l):
            vv[i] = self.__matrix[i + k][k]
        sc = np.dot(vv, av)
        pp = self.__matrix[k][k] - 2.0 * av[0] * sc
        for i in range(k + 1, self.__rows):
            for ii in range(l):
                vv[ii] = self.__matrix[ii + k][i]
            sc = np.dot(av, vv)
            for ii in range(k, self.__rows):
                self.__matrix[ii][i] -= 2.0 * av[ii - k] * sc
        for i in range(l):
            self.__matrix[i + k][k] = av[i]
        self.__p1.append(pp)

    def __del_row(self, k) -> None:
        if k >= self.__columns - 1 or k >= self.__rows:
            return
        l: int = self.__columns - k - 1
        av = np.zeros(l, dtype=complex)
        for i in range(l):
            av[i] = self.__matrix[k][i + k + 1]
        av[0] -= np.linalg.norm(av) * av[0] / abs(av[0])
        av /= np.linalg.norm(av)
        vv = np.zeros(l, dtype=complex)
        for i in range(l):
            vv[i] = self.__matrix[k][i + k + 1]
        sc = np.dot(av, vv)
        pp = self.__matrix[k][k + 1] - 2 * av[0] * sc
        for i in range(k + 1, self.__rows):
            for ii in range(l):
                vv[ii] = self.__matrix[i][ii + k + 1]
            sc = np.dot(av, vv)
            for ii in range(k + 1, self.__columns):
                self.__matrix[i][ii] -= 2 * av[ii - k - 1] * sc
        for i in range(l):
            self.__matrix[k][i + k + 1] = av[i]
        self.__p2.append(pp)

    def __multiply_rx(self) -> None:
        for i in range(self.__rows - 1):
            av = np.zeros(self.__columns, dtype=complex)
            for ii in range(i + 1, self.__columns):
                av[ii] = self.__matrix[i][ii]
            sc = 0
            for ii in range(i + 1, self.__columns):
                sc += av[ii] * self.__right_part[ii]
            for ii in range(i + 1, self.__columns):
                self.__right_part[ii] -= 2 * av[ii].conjugate() * sc
