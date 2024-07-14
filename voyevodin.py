from IterativeProcess import IterativeProcess
from MatrixSystem import MatrixSystem
from Stabilizer import BoundaryCondition


class VoyevodinMethod:
    # конструктор
    def __init__(self, matrix, right_part, step,
                 left=BoundaryCondition.NEUMANN, right=BoundaryCondition.NEUMANN,
                 p=1.0, alpha_initial_value=0.1e-1,
                 h=0, delta=0, eps=1e-4):
        #self.__matrix = matrix
        #self.__right_part = right_part,
        #self.__left = left
        #self.__right = right
        #self.__p = p
        #self.__alpha_initial_value = alpha_initial_value
        #self.__h = h
        #self.__eps = eps
        # 1. Создаём систему и приводим её к двухдиагональному виду
        ms = MatrixSystem(matrix, right_part, step, h, p, left, right)
        ip = IterativeProcess(ms.diagonal, ms.up_diagonal, ms.right_part, ms.multiply_qtu(right_part),
                              alpha_initial_value, step, h, delta, eps)
        self.__solution = ip.solution
        ms.__multiply_rtx(self.__solution)
        ms.__multiply_sinv(self.__solution)
