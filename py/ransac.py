# -*- coding: utf-8 -*-
"""
Created on 2018-04-21 21:04:04
@Author: ZHAO Lingfeng
@Version : 0.0.1
"""
import random

import numpy as np
from scipy import linalg


class Ransac:

    def __init__(self, data1: np.ndarray, data2: np.ndarray, max_iter_times=1000):
        """利用Ransac算法求得变换矩阵

        Args:
            data1 (np.ndarray): n*2形状的点数组
            data2 (np.ndarray): n*2形状的点数组
            max_iter_times (int, optional): Defaults to 1000. 最大迭代次数

        Raises:
            ValueError: 输入数组形状不同
        """

        self.data1 = data1
        self.data2 = data2
        if(self.data1.shape != self.data2.shape):
            raise ValueError("Argument shape not equal")

        self.points_length = self.data1.shape[0]
        self.good_points = 0
        self.max_iter_times = max_iter_times

        # 0 for not chosen
        self.mask = np.zeros(self.points_length, dtype=np.bool)

    def random_calculate(self, max_try_times=1000)-> np.ndarray:
        """随机取四个点对，计算其变换矩阵
            max_try_times (int, optional): Defaults to 1000. 选取尝试次数

        Returns:
            np.ndarray: 变换矩阵M
        """

        if self.points_length - np.count_nonzero(self.mask) < 4:
            return False

        try_times = 0
        rand_point = random.sample(range(self.points_length), 4)
        while np.any(self.mask[rand_point]):
            try_times += 1
            rand_point = random.sample(range(self.points_length), 4)
            if try_times > max_try_times:
                return False
        self.mask[rand_point] = True
        M = self.get_perspective_transform(self.data1[rand_point], self.data2[rand_point])
        return M

    @staticmethod
    def get_perspective_transform(src: np.ndarray, dst: np.ndarray)-> np.ndarray:
        """获取透视变换矩阵

        Args:
            src (np.ndarray): 2*4形状
            dst (np.ndarray): 2*4形状

        Returns:
            np.ndarray: 变换矩阵M
        """
        X = np.array((8, 1), np.float)
        A = np.zeros((8, 8), np.float)
        B = np.zeros((8), np.float)

        for i in range(4):
            A[i][0] = A[i + 4][3] = src[i][0]
            A[i][1] = A[i + 4][4] = src[i][1]
            A[i][2] = A[i + 4][5] = 1
            A[i][3] = A[i][4] = A[i][5] = A[i + 4][0] = A[i + 4][1] = A[i + 4][2] = 0
            A[i][6] = -src[i][0] * dst[i][0]
            A[i][7] = -src[i][1] * dst[i][0]
            A[i + 4][6] = -src[i][0] * dst[i][1]
            A[i + 4][7] = -src[i][1] * dst[i][1]
            B[i] = dst[i][0]
            B[i + 4] = dst[i][1]

        X = linalg.solve(A, B).copy()
        X.resize((3, 3))
        X[2][2] = 1
        return X

    @staticmethod
    def perspective_transform(points: np.ndarray, M: np.ndarray)->np.ndarray:
        """求得在M变换矩阵情况下points数组里每个点的新变换坐标

        Args:
            points (np.ndarray): n*2形状的数组
            M (np.ndarray): 3*3的透视变换矩阵

        Returns:
            np.ndarray: 变换后的点
        """

        array_i = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype))).T
        x = np.dot(M, array_i)
        result = np.vstack((x[0] / x[2], x[1] / x[2])).T
        return result

    # TODO: get a better threshold or dynamic change it
    @classmethod
    def get_good_points(cls, points1: np.ndarray, points2: np.ndarray, M: np.ndarray, threshold=3)-> np.ndarray:
        """求得在给定变换矩阵下，计算将points1变换后与points2之间的距离

        Args:
            points1 (np.ndarray): n*2形状数组
            points2 (np.ndarray): n*2形状数组
            M (np.ndarray): 3*3透视变换矩阵
            threshold (int, optional): Defaults to 3. 距离阈值

        Returns:
            np.ndarray: Bool数组，是否是优秀点
        """

        transformed = cls.perspective_transform(points1, M)
        dis = np.sum((transformed - points2) * (transformed - points2), axis=1)
        good = dis < threshold * threshold
        return good

    @staticmethod
    def get_itereration_time(proportion: float, p=0.995, points=4)->int:
        """更新迭代次数

        Args:
            proportion (float): 优秀点比例
            p (float, optional): Defaults to 0.995. 确信值
            points (int, optional): Defaults to 4. 四个点

        Returns:
            int: 更新后的所需迭代次数
        """

        proportion = max(min(proportion, 1), 0)
        p = max(min(p, 1), 0)
        # print(p,proportion)
        k = np.log(1 - p) / np.log(1 - np.power(proportion, 4))
        return int(k)

    def run(self)->np.ndarray:
        """进行计算

        Returns:
            计算的透视变换矩阵
        """

        iter_times = 0
        best_M = None
        while iter_times < self.max_iter_times:
            M = self.random_calculate()
            good = self.get_good_points(self.data1, self.data2, M)
            good_nums = np.sum(good)
            if good_nums > self.good_points:
                self.good_points = good_nums
                best_M = M
                self.max_iter_times = min(self.max_iter_times,
                                          self.get_itereration_time(good_nums / self.points_length))
            iter_times += 1

        return best_M


def main():
    pass
    test()


def test():
    data_point1 = np.array([[1, 2], [3, 3], [5, 5], [6, 8]])
    data_point2 = np.array([[4, 2], [5, 3], [12, 5], [64, 8]])
    ransac = Ransac(data_point1, data_point2)
    M = ransac.get_perspective_transform(data_point1, data_point2)
    print(M)
    print("Supposed to be:\n",
          "[[-0.76850095,  1.15180266,  1.77229602]\n",
          "[ 0.20777989, -0.31593928,  2.07779886],\n",
          "[-0.10388994, -0.03462998,  1.        ]]")
    print(ransac.get_good_points(data_point1, data_point1, M))

    test_data = np.random.rand(20, 2) * 30
    dst_data = Ransac.perspective_transform(test_data, M)
    dst_data[0] = [21312, 213123]
    ransac = Ransac(test_data, dst_data)
    result_M = ransac.run()
    print("Result M:")
    print(result_M)
    print("Max itereration times")
    print(ransac.max_iter_times)


if __name__ == "__main__":
    main()
