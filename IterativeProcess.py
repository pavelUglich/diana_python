import numpy as np


class IterativeProcess:

    def __init__(self, p1, p2, right_part, qtu, alpha, step, h, delta, eps,
                 iterations=50):
        self.__p1 = p1
        self.__p2 = p2
        self.__qtu = qtu
        self.__right_part = right_part
        self.__alpha = alpha
        self.__step = step
        self.__h = h
        self.__delta = delta
        self.__iterations = iterations
        self.__size = len(right_part)
        self.__eps = eps
        self.__tridiag()
        self.__iterations_run()

    def __tridiag(self):
        self.__a = np.zeros(self.__size, dtype=complex)
        self.__b = np.zeros(self.__size, dtype=complex)
        self.__c = np.zeros(self.__size, dtype=complex)
        size = len(self.__p1)
        self.__a[0] = np.abs(self.__p1[0]) ** 2
        if len(self.__p1) == len(self.__p2):
            self.__a[size - 1] = np.abs(self.__p2[size - 1]) ** 2
        else:
            self.__a[size - 1] = np.abs(self.__p2[size - 1]) ** 2 + np.abs(self.__p1[size - 1]) ** 2
        self.__b[0] = self.__p1[0].conjugate()* self.__p2[0]
        self.__b[size-1] = 0
        for i in range(1, self.__size - 1):
            self.__a[i] = np.abs(self.__p2[i-1]) ** 2 + np.abs(self.__p1[i]) ** 2
            self.__b[i] = self.__p1[i].conjugate() * self.__p2[i]
        self.__c[0] = 0
        for i in range(1, size):
            self.__c[i] = self.__b[i-1].conjugate()

    def __iterations_run(self):
        alpha_s = self.__alpha
        alpha_n = 0.5*alpha_s
        ss = self.__residual(alpha_s)
        sn = self.__residual(alpha_n)
        for i in range(self.__iterations):
            alpha = alpha_n/(1-(1/alpha_s)*(alpha_s-alpha_n)*sn/(sn-ss))
            ss = sn
            sn = self.__residual(alpha)
            alpha_s = alpha_n
            alpha_n = alpha
            if np.abs(alpha_n-alpha_s):
                break



    def __residual(self, alpha):
        a = np.zeros(self.__size, dtype=complex)
        for i in range(self.__size):
            a[i] = -self.__a[i] - alpha
        self.__solution = self.__marching(a)
        nz = np.linalg.norm(self.__solution)
        pz = np.zeros(len(self.__p1), dtype=complex)
        for i in range(len(self.__p2) - 1):
            pz[i] = self.__p1[i] * self.__solution[i] + self.__p2[i] * self.__solution[i+1]
        pz[len(self.__p1) - 1] = self.__p1[len(self.__p1) - 1] * self.__solution[len(self.__p1) - 1]
        for i in range(len(pz)):
            pz[i] -= self.__qtu[i]
        npz = np.linalg.norm(pz)
        return self.__step**2 * npz**2 - (self.__delta+self.__h*nz)**2

    def __marching(self, a):
        x = np.zeros(self.__size)
        xi = np.zeros(self.__size + 1)
        eta = np.zeros(self.__size + 1)
        xi[0] = 0
        eta[0] = 0
        for i in range(self.__size):
            denum = a[i] - self.__c[i] * xi[i]
            xi[i+1] = self.__b[i] / denum
            eta[i+1] = (self.__c[i]*eta[i]-self.__right_part[i])/denum
        x[self.__size-1] = eta[self.__size]
        for i in range(1, self.__size):
            x[self.__size - i - 1] = xi[self.__size - i] * x[self.__size - i] + eta[self.__size-i]
        return x

    @property
    def solution(self):
        return self.__solution
