# -*- coding: utf-8 -*-
"""
Created on 2018-03-13 15:29:29
@Author: Ben
@Version : 0.0.1
"""

from enum import Enum
from typing import List, Tuple, Union
import unittest
import os
import random

import cv2
import numpy as np

import k_means
from ransac import *
import blend


def show_image(image: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).show()


class Method(Enum):

    SURF = cv2.xfeatures2d.SURF_create
    SIFT = cv2.xfeatures2d.SIFT_create
    ORB = cv2.ORB_create


colors = ((123, 234, 12), (23, 44, 240), (224, 120, 34), (21, 234, 190),
          (80, 160, 200), (243, 12, 100), (25, 90, 12), (123, 10, 140))


class Area:

    def __init__(self, *points):

        self.points = list(points)

    def is_inside(self, x: Union[float, Tuple[float, float]], y: float=None) -> bool:
        if isinstance(x, tuple):
            x, y = x
        raise NotImplementedError()


class Matcher():

    def __init__(self, image1: np.ndarray, image2: np.ndarray, method: Enum=Method.SURF, threshold=800) -> None:
        """输入两幅图像，计算其特征值
        此类用于输入两幅图像，计算其特征值，输入两幅图像分别为numpy数组格式的图像，其中的method参数要求输入SURF、SIFT或者ORB，threshold参数为特征值检测所需的阈值。

        Args:
            image1 (np.ndarray): 图像一
            image2 (np.ndarray): 图像二
            method (Enum, optional): Defaults to Method.SURF. 特征值检测方法
            threshold (int, optional): Defaults to 800. 特征值阈值

        """

        self.image1 = image1
        self.image2 = image2
        self.method = method
        self.threshold = threshold

        self._keypoints1: List[cv2.KeyPoint] = None
        self._descriptors1: np.ndarray = None
        self._keypoints2: List[cv2.KeyPoint] = None
        self._descriptors2: np.ndarray = None

        if self.method == Method.ORB:
            # error if not set this
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self.matcher = cv2.BFMatcher(crossCheck=True)
            # self.matcher = cv2.FlannBasedMatcher()

        self.match_points = []

        self.image_points1 = np.array([])
        self.image_points2 = np.array([])

    def compute_keypoint(self) -> None:
        """计算特征点
        利用给出的特征值检测方法对图像进行特征值检测。

        Args:
            image (np.ndarray): 图像
        """
        feature = self.method.value(self.threshold)
        self._keypoints1, self._descriptors1 = feature.detectAndCompute(
            self.image1, None)
        self._keypoints2, self._descriptors2 = feature.detectAndCompute(
            self.image2, None)

    def match(self, max_match_lenth=20, threshold=0.04, show_match=False):
        """对两幅图片计算得出的特征值进行匹配，对ORB来说使用OpenCV的BFMatcher算法，而对于其他特征检测方法则使用FlannBasedMatcher算法。

            max_match_lenth (int, optional): Defaults to 20. 最大匹配点数量
            threshold (float, optional): Defaults to 0.04. 默认最大匹配距离差
            show_match (bool, optional): Defaults to False. 是否展示匹配结果
        """

        self.compute_keypoint()

        '''计算两张图片中的配对点，并至多取其中最优的`max_match_lenth`个'''
        self.match_points = sorted(self.matcher.match(
            self._descriptors1, self._descriptors2), key=lambda x: x.distance)

        match_len = min(len(self.match_points), max_match_lenth)

        # if self.method == Method.ORB:
        #     threshold = 20
        # elif self.method == Method.SIFT:
        #     threshold = 20
        # max_distance = max(2 * self.match_points[0].distance, threshold)

        # in case distance is 0
        max_distance = max(2 * self.match_points[0].distance, 20)

        for i in range(match_len):
            if self.match_points[i].distance > max_distance:
                match_len = i
                break
        print('max distance: ', self.match_points[match_len].distance)
        print("Min distance: ", self.match_points[0].distance)
        print('match_len: ', match_len)
        assert(match_len >= 4)
        self.match_points = self.match_points[:match_len]

        if show_match:
            img3 = cv2.drawMatches(self.image1, self._keypoints1, self.image2, self._keypoints2,
                                   self.match_points, None, flags=0)
            show_image(img3)

        '''由最佳匹配取得匹配点对，并进行形变拼接'''
        image_points1, image_points2 = [], []
        for i in self.match_points:
            image_points1.append(self._keypoints1[i.queryIdx].pt)
            image_points2.append(self._keypoints2[i.trainIdx].pt)

        self.image_points1 = np.float32(image_points1)
        self.image_points2 = np.float32(image_points2)

        # print(image_points1)


def get_weighted_points(image_points: np.ndarray):

    # print(k_means.k_means(image_points))
    # exit(0)

    average = np.average(image_points, axis=0)

    max_index = np.argmax(np.linalg.norm((image_points - average), axis=1))
    return np.append(image_points, np.array([image_points[max_index]]), axis=0)


class Stitcher:

    def __init__(self, image1: np.ndarray, image2: np.ndarray, method: Enum=Method.SURF, use_kmeans=False):
        """输入图像和匹配，对图像进行拼接
        目前采用简单矩阵匹配和平均值拼合

        Args:
            image1 (np.ndarray): 图像一
            image2 (np.ndarray): 图像二
            matcher (Matcher): 匹配结果
            use_kmeans (bool): 是否使用kmeans 优化点选择
        """

        self.image1 = image1
        self.image2 = image2
        self.method = method
        self.use_kmeans = use_kmeans
        self.matcher = Matcher(image1, image2, method=method)
        self.M = np.eye(3)

        self.image = None

    def stich(self, show_result=True, max_match_lenth=40, show_match_point=True, use_partial=False):
        """对图片进行拼合

            show_result (bool, optional): Defaults to True. 是否展示拼合图像
            show_match_point (bool, optional): Defaults to True. 是否展示拼合点
        """
        self.matcher.match(max_match_lenth=max_match_lenth,
                           show_match=show_match_point)

        if self.use_kmeans:
            self.image_points1, self.image_points2 = k_means.get_group_center(
                self.matcher.image_points1, self.matcher.image_points2)
        else:
            self.image_points1, self.image_points2 = (
                self.matcher.image_points1, self.matcher.image_points2)

        self.M, _ = cv2.findHomography(
            self.image_points1, self.image_points2, cv2.RANSAC)

        # ransac = Ransac(self.image_points1, self.image_points2)
        # self.M = ransac.run()
        # print("RANSAC Good Points: ", ransac.good_points)
        # print("RANSAC Iteration times: ", ransac.max_iter_times)
        # # print("RANSAC Totall Points: ", ransac.points_length)

        left, right, top, bottom = self.get_transformed_size()
        # print(self.get_transformed_size())
        width = int(max(right, self.image2.shape[1]) - min(left, 0))
        height = int(max(bottom, self.image2.shape[0]) - min(top, 0))
        print(width, height)
        width, height = min(width, 10000), min(height, 10000)
        if use_partial:
            self.partial_transform()

        # 移动矩阵
        adjustM = np.array(
            [[1, 0, max(-left, 0)],  # 横向
             [0, 1, max(-top, 0)],  # 纵向
             [0, 0, 1]
             ], dtype=np.float64)
        # print('adjustM: ', adjustM)
        self.M = np.dot(adjustM, self.M)
        transformed_1 = cv2.warpPerspective(
            self.image1, self.M, (width, height))
        transformed_2 = cv2.warpPerspective(
            self.image2, adjustM, (width, height))

        self.image = self.blend(transformed_1, transformed_2)

        if show_match_point:
            for point1, point2 in zip(self.image_points1, self.image_points2):
                point1 = self.get_transformed_position(tuple(point1))
                point1 = tuple(map(int, point1))
                point2 = self.get_transformed_position(tuple(point2), M=adjustM)
                point2 = tuple(map(int, point2))

                cv2.line(self.image, point1, point2, random.choice(colors), 3)
                cv2.circle(self.image, point1, 10, (20, 20, 255), 5)
                cv2.circle(self.image, point2, 8, (20, 200, 20), 5)
        if show_result:
            show_image(self.image)

    def partial_transform(self):
        def distance(p1, p2):
            return np.sqrt(
                (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
        width = self.image1.shape[0]
        height = self.image1.shape[1]
        offset_x = np.min(self.image_points1[:, 0])
        offset_y = np.min(self.image_points1[:, 1])

        # width = np.max(self.image_points1[:, 0]) - offset_x
        # height = np.max(self.image_points1[:, 1]) - offset_y
        x_mid = int((np.max(self.image_points1[:, 0]) + offset_x) / 2)
        y_mid = int((np.max(self.image_points1[:, 1]) + offset_y) / 2)

        center = [0, 0]
        up = x_mid
        down = width - x_mid
        left = y_mid
        right = height - y_mid

        ne, se, sw, nw = [], [], [], []
        transform_acer = [[center, [up, 0], [up, right]],
                          [center, [down, 0], [0, right]],
                          [center, [down, left], [0, left]],
                          [[up, 0], [up, left], [up, left]]]
        transform_acer = [[center, [0, up], [right, up]],
                          [center, [0, down], [right, 0]],
                          [center, [left, down], [left, 0]],
                          [[0, up], [left, up], [left, up]]]
        # transform_acer = [[center, up, right],
        #                   [center, down, right],
        #                   [center, down, left],
        #                   [center, up, left]]

        # 对点的位置进行分类
        for index in range(self.image_points1.shape[0]):
            point = self.image_points1[index]
            if point[0] > y_mid:
                if point[1] > x_mid:
                    se.append(index)
                else:
                    ne.append(index)
            else:
                if point[1] > x_mid:
                    sw.append(index)
                else:
                    nw.append(index)

        # 求点最少处位置，排除零
        minmum = np.argmin(
            list(map(lambda x: len(x) if len(x) > 0 else 65536, [ne, se, sw, nw])))
        # 当足够少时
        min_part = (ne, se, sw, nw)[minmum]

        # debug:
        print("minum part: ", minmum, "point len: ", len(
            min_part), "|", list(map(len, (ne, se, sw, nw))))
        for index in min_part:
            point = self.image_points1[index]
            cv2.circle(self.image1, tuple(
                map(int, point)), 20, (0, 255, 255), 5)

        # cv2.circle(self.image1, tuple(map(int, (y_mid, x_mid))),
        #            25, (255, 100, 60), 7)
        # end debug
        if len(min_part) < len(self.image_points1) / 8:
            for index in min_part:
                point = self.image_points1[index].tolist()
                print("Point: ", point)
                # maybe can try other value?
                if distance(self.get_transformed_position(tuple(point)),
                            self.image_points2[index]) > 10:
                    def relevtive_point(p):
                        return (p[0] - y_mid if p[0] > y_mid else p[0],
                                p[1] - x_mid if p[1] > x_mid else p[1])
                    cv2.circle(self.image1, tuple(map(int, point)),
                               40, (255, 0, 0), 10)
                    src_point = transform_acer[minmum].copy()
                    src_point.append(relevtive_point(point))
                    other_point = self.get_transformed_position(
                        tuple(self.image_points2[index]), M=np.linalg.inv(self.M))
                    dest_point = transform_acer[minmum].copy()
                    dest_point.append(relevtive_point(other_point))

                    def a(x): return np.array(x, dtype=np.float32)
                    print(src_point, dest_point)
                    partial_M = cv2.getPerspectiveTransform(
                        a(src_point), a(dest_point))

                    if minmum == 1 or minmum == 2:
                        boder_0, boder_1 = x_mid, width
                    else:
                        boder_0, boder_1 = 0, x_mid
                    if minmum == 2 or minmum == 3:
                        boder_2, boder_3 = 0, y_mid
                    else:
                        boder_2, boder_3 = y_mid, height

                    print("Changed:",
                          "\nM: ", partial_M,
                          "\npart: ", minmum,
                          "\ndistance: ", distance(self.get_transformed_position(tuple(point)),
                                                   self.image_points2[index])
                          )
                    part = self.image1[boder_0:boder_1, boder_2:boder_3]

                    print(boder_0, boder_1, boder_2, boder_3)
                    for point in transform_acer[minmum]:
                        print(point)
                        cv2.circle(part, tuple(
                            map(int, point)), 40, (220, 200, 200), 10)
                    for point in src_point:
                        print(point)
                        cv2.circle(part, tuple(
                            map(int, point)), 22, (226, 43, 138), 8)
                    # for point in dest_point:
                    #     print(point)
                    #     cv2.circle(part, tuple(
                    #         map(int, point)), 25, (20, 97, 199), 8)

                    # cv2.circle(part, tuple(map(int, relevtive_point(point))),
                    #            40, (255, 0, 0), 10)
                    # show_image(part)

                    part = cv2.warpPerspective(
                        part, partial_M, (part.shape[1], part.shape[0]))
                    cv2.circle(part, tuple(map(int, relevtive_point(other_point))),
                               40, (20, 97, 199), 6)
                    # show_image(part)
                    self.image1[boder_0:boder_1, boder_2:boder_3] = part
                    return

    def blend(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """对图像进行融合

        Args:
            image1 (np.ndarray): 图像一
            image2 (np.ndarray): 图像二

        Returns:
            np.ndarray: 融合结果
        """

        # result = cv2.addWeighted(image1, 0.5, image2, 0.5, 1)
        # result = blend.average_blend(image1, image2)
        mask = self.generate_mask(image1, image2)
        print("Blending")
        result = blend.gaussian_blend(image1, image2, mask)

        return result

    def generate_mask(self, image1: np.ndarray, image2: np.ndarray):
        """生成供融合使用的遮罩，由变换后图像的垂直平分线来构成分界线

        Args:
            shape (tuple): 遮罩大小

        Returns:
            np.ndarray: 01数组
        """
        print("Generating mask")
        center1 = self.image1.shape[0] / 2, self.image1.shape[1] / 2
        center2 = self.image2.shape[0] / 2, self.image2.shape[1] / 2
        center2 = self.get_transformed_position(center2)
        # 垂直平分线 y=-(x2-x1)/(y2-y1)* [x-(x1+x2)/2]+(y1+y2)/2
        x1, y1 = center1
        x2, y2 = center2
        print(x1, y1, x2, y2)
        print(-(x2 - x1) / (y2 - y1), ' * (x -', (x1 + x2) / 2, ') + ', (y1 + y2) / 2)

        def y(x):
            return -((x2 - x1) / (y2 - y1)) * (x - (x1 + x2) / 2) + (y1 + y2) / 2

        if y2 > y1:
            def function(i, j, *k): return j > y(i)
        else:
            def function(i, j, *k): return j < y(i)

        mask = np.fromfunction(function, image1.shape)
        # mask = mask&_i2+mask&i1+i1&_i2
        mask = np.logical_and(mask, np.logical_not(image2)) \
            + np.logical_and(mask, image1)\
            + np.logical_and(image1, np.logical_not(image2))
        return mask

    def get_transformed_size(self) ->Tuple[int, int, int, int]:
        """计算形变后的边界
        计算形变后的边界，从而对图片进行相应的位移，保证全部图像都出现在屏幕上。

        Returns:
            Tuple[int, int, int, int]: 分别为左右上下边界
        """

        conner_0 = (0, 0)  # x, y
        conner_1 = (self.image1.shape[1], 0)
        conner_2 = (self.image1.shape[1], self.image1.shape[0])
        conner_3 = (0, self.image1.shape[0])
        points = [conner_0, conner_1, conner_2, conner_3]

        # top, bottom: y, left, right: x
        top = min(map(lambda x: self.get_transformed_position(x)[1], points))
        bottom = max(
            map(lambda x: self.get_transformed_position(x)[1], points))
        left = min(map(lambda x: self.get_transformed_position(x)[0], points))
        right = max(map(lambda x: self.get_transformed_position(x)[0], points))

        return left, right, top, bottom

    def get_transformed_position(self, x: Union[float, Tuple[float, float]], y: float=None, M=None) -> Tuple[float, float]:
        """求得某点在变换矩阵（self.M）下的新坐标

        Args:
            x (Union[float, Tuple[float, float]]): x坐标或(x,y)坐标
            y (float, optional): Defaults to None. y坐标，可无
            M (np.ndarray, optional): Defaults to None. 利用M进行坐标变换运算

        Returns:
            Tuple[float, float]:  新坐标
        """

        if isinstance(x, tuple):
            x, y = x
        p = np.array([x, y, 1])[np.newaxis].T
        if M is not None:
            M = M
        else:
            M = self.M
        pa = np.dot(M, p)
        return pa[0, 0] / pa[2, 0], pa[1, 0] / pa[2, 0]


class Test(unittest.TestCase):

    def _test_matcher(self):
        image1 = np.random.randint(100, 256, size=(400, 400, 3), dtype='uint8')
        # np.random.randint(256, size=(400, 400, 3), dtype='uint8')
        image2 = np.copy(image1)
        for method in Method:
            matcher = Matcher(image1, image2, method)

            matcher.match(show_match=True)

    def test_transform_coord(self):
        stitcher = Stitcher(None, None, None, None)
        self.assertEqual((0, 0), stitcher.get_transformed_position(0, 0))
        self.assertEqual((10, 20), stitcher.get_transformed_position(10, 20))

        stitcher.M[0, 2] = 20
        stitcher.M[1, 2] = 10
        self.assertEqual((20, 10), stitcher.get_transformed_position(0, 0))
        self.assertEqual((30, 30), stitcher.get_transformed_position(10, 20))

        stitcher.M = np.eye(3)
        stitcher.M[0, 1] = 2
        stitcher.M[1, 0] = 4
        self.assertEqual((0, 0), stitcher.get_transformed_position(0, 0))
        self.assertEqual((50, 60), stitcher.get_transformed_position(10, 20))

    def test_get_transformed_size(self):
        image1 = np.empty((500, 400, 3), dtype='uint8')
        image1[:, :] = 255, 150, 100
        image1[:, 399] = 10, 20, 200

        # show_image(image1)
        image2 = np.empty((400, 400, 3), dtype='uint8')
        image2[:, :] = 50, 150, 255

        stitcher = Stitcher(image1, image2, None, None)
        stitcher.M[0, 2] = -20
        stitcher.M[1, 2] = 10
        stitcher.M[0, 1] = .2
        stitcher.M[1, 0] = .1
        left, right, top, bottom = stitcher.get_transformed_size()
        print(stitcher.get_transformed_size())
        width = int(max(right, image2.shape[1]) - min(left, 0))
        height = int(max(bottom, image2.shape[0]) - min(top, 0))
        print(width, height)
        show_image(cv2.warpPerspective(image1, stitcher.M, (width, height)))

    def test_stich(self):
        image1 = np.empty((500, 400, 3), dtype='uint8')
        image1[:, :] = 255, 150, 100
        image1[:, 399] = 10, 20, 200

        # show_image(image1)
        image2 = np.empty((400, 400, 3), dtype='uint8')
        image2[:, :] = 50, 150, 255

        points = np.float32([[0, 0], [20, 20], [12, 12], [40, 20]])
        stitcher = Stitcher(image1, image2, points, points)
        stitcher.M[0, 2] = 20
        stitcher.M[1, 2] = 10
        stitcher.M[0, 1] = .2
        stitcher.M[1, 0] = .1
        stitcher.stich()


def main():
    unittest.main()


if __name__ == "__main__":
    import time
    # main()
    os.chdir(os.path.dirname(__file__))

    start_time = time.time()
    img2 = cv2.imread("../resource/20-left.jpg")
    img1 = cv2.imread("../resource/20-right.jpg")
    # matcher = Matcher(img1, img2, Method.ORB)
    # matcher.match(max_match_lenth=20, show_match=True,)
    stitcher = Stitcher(img1, img2, Method.ORB, False)
    stitcher.stich(max_match_lenth=40, use_partial=False)

    # cv2.imwrite('../resource/20-sift-ransac.jpg', stitcher.image)

    print("Time: ", time.time() - start_time)
    print("M: ", stitcher.M)
