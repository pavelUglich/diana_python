
class MatrixSystem:
    # ms = MatrixSystem(matrix, right_part, step, h, delta, eps)

    def __init__(self, matrix, right_part, step, h, left, right):
        self.__matrix = matrix
        self.__right_part = right_part
        self.__step = step
        self.__h = h
        self.__rows = len(matrix)
        self.__columns = len(matrix[0])
        self.__stabilizer = Stabilizer(self.__columns, step, p, left, right)
        self.__multipy_asinv()

    def __multipy_asinv(self) -> None:
