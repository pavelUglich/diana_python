class VoyevodinMethod:
    # конструктор
    def __init__(self, matrix, right_part, step,
                 left=NEUMANN, right=NEUMANN,
                 p=1.0, alpha_initial_value=0.1e-1,
                 h=0, delta=0, eps=1e-4):
        self.__matrix = matrix
        self.__right_part = right_part,
        self.__left = left
        self.__right = right
        self.__p = p
        self.__alpha_initial_value = alpha_initial_value
        self.__h = h
        self.__eps = eps
        # 1. Создаём систему и приводим её к двухдиагональному виду
        ms = MatrixSystem(matrix, right_part, step, h, left, right)

