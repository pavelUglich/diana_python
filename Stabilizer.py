import numpy as np
from enum import Enum


class BoundaryCondition(Enum):
    NEUMANN = 1
    DIRICHLE = 2


class Stabilizer:
    def __init__(self, n, step, p, left, right):
        self.__size = n
        self.__diagonal = np.zeros(n)
        self.__up_diagonal = np.zeros(n - 1)
        h_stab = p / step / step
        for i in range(n):
            self.__diagonal[i] = 1 + 2 * h_stab
        for i in range(n - 1):
            self.__up_diagonal[i] = -h_stab
        if left == BoundaryCondition.DIRICHLE:
            self.__diagonal[0] = 1 + 3 * h_stab
        else:
            self.__diagonal[0] = 1 + h_stab
        if right == BoundaryCondition.DIRICHLE:
            self.__diagonal[n - 1] = 1 + 3 * h_stab
        else:
            self.__diagonal[n - 1] = 1 + h_stab
        self.__squareroot()

    def __squareroot(self) -> None:
        self.__diagonal[0] = np.sqrt(self.__diagonal[0])
        self.__up_diagonal[0] /= self.__diagonal[0]
        for i in range(1, self.__size - 1):
            self.__diagonal[i] = np.sqrt(self.__diagonal[i] - self.__up_diagonal[i - 1] * self.__up_diagonal[i - 1])
            self.__up_diagonal[i] /= self.__diagonal[i]
        self.__diagonal[self.__size - 1] = np.sqrt(
            self.__diagonal[self.__size - 1] - self.__up_diagonal[self.__size - 2] * self.__up_diagonal[
                self.__size - 2])

    @property
    def diagonal(self):
        return self.__diagonal

    @property
    def up_diagonal(self):
        return self.__up_diagonal

