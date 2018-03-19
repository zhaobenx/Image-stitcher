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


import cv2
import numpy as np


def show_image(image: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).show()


class Method(Enum):

    SURF = cv2.xfeatures2d.SURF_create
    SIFT = cv2.xfeatures2d.SIFT_create
    ORB = cv2.ORB_create
    # FAST = cv2.FastFeatureDetector_create

    # @classmethod
    # def __getattribute__(cls, attribute: str):
    #     if hasattr(cls, attribute):
    #         return getattr(cls, attribute)


class Area:

    def __init__(self, *points):

        self.points = list(points)

    def is_inside(self, x: Union[float, Tuple[float, float]], y: float=None) -> bool:
        if isinstance(x, tuple):
            x, y = x
        raise NotImplementedError()


class Matcher():
    """docstring for Matcher."""

    def __init__(self, image1: np.ndarray, image2: np.ndarray, method: Enum=Method.SURF, threshold=800) -> None:

        self.image1 = image1
        self.image2 = image2
        self.method = method
        self.threshold = threshold

        self._keypoints1: List[cv2.KeyPoint] = None
        self._descriptors1: np.ndraary = None
        self._keypoints2: List[cv2.KeyPoint] = None
        self._descriptors2: np.ndraary = None

        if self.method == Method.ORB:
            # error if not set this
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self.matcher = cv2.FlannBasedMatcher()

        self.match_points = []

        self.image_points1 = np.array([])
        self.image_points2 = np.array([])

    def compute_keypoint(self) -> None:
        """计算特征点

        Args:
            image (np.ndarray): 图像

        Returns:
            Tuple[List[cv2.KeyPoint], np.ndarray[np.ndarray]]: 特征点和描述值
        """
        feature = self.method.value(self.threshold)
        self._keypoints1, self._descriptors1 = feature.detectAndCompute(
            self.image1, None)
        self._keypoints2, self._descriptors2 = feature.detectAndCompute(
            self.image2, None)

    def match(self, max_match_lenth=20, threshold=0.04, show_match=False):
        self.compute_keypoint()

        '''计算两张图片中的配对点，并至多取其中最优的`max_match_lenth`个'''
        self.match_points = sorted(self.matcher.match(
            self._descriptors1, self._descriptors2), key=lambda x: x.distance)

        match_len = min(len(self.match_points), max_match_lenth)
        min_distance = max(2 * self.match_points[0].distance, threshold)

        for i in range(match_len):
            if self.match_points[i].distance > min_distance:
                match_len = i
                break
        print('min distance: ', min_distance)
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


class Sticher:
    """docstring for Sticher."""

    def __init__(self, image1: np.ndarray, image2: np.ndarray, matcher: Matcher):
        self.image1 = image1
        self.image2 = image2
        self.image_points1 = matcher.image_points1
        self.image_points2 = matcher.image_points2

        self.M = np.eye(3)

        self.image = None

    def stich(self, show_result=True, show_match_point=True):
        self.M, _ = cv2.findHomography(
            self.image_points1, self.image_points2, cv2.RANSAC)

        left, right, top, bottom = self.get_transformed_size()
        print(self.get_transformed_size())
        width = int(max(right, self.image2.shape[1]) - min(left, 0))
        height = int(max(bottom, self.image2.shape[0]) - min(top, 0))
        print(width, height)

        # 移动矩阵
        adjustM = np.array(
            [[1, 0, max(-left, 0)],  # 横向
             [0, 1, max(-top, 0)],  # 纵向
             [0, 0, 1]
             ], dtype=np.float64)
        print('adjustM: ', adjustM)
        self.M = np.dot(adjustM, self.M)
        transformed_1 = cv2.warpPerspective(
            self.image1, self.M, (width, height))
        transformed_2 = cv2.warpPerspective(
            self.image2, adjustM, (width, height))

        self.image = self.blend(transformed_1, transformed_2)

        if show_match_point:
            for point in self.image_points1:
                point = self.get_transformed_position(tuple(point))
                point = tuple(map(int, point))
                cv2.circle(self.image, point, 10, (20, 20, 255))
            for point in self.image_points2:
                point = self.get_transformed_position(tuple(point), M=adjustM)
                point = tuple(map(int, point))
                cv2.circle(self.image, point, 8, (20, 200, 20))
        if show_result:
            show_image(self.image)

    def blend(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        # result = np.zeros(image1.shape, dtype='uint8')
        # result[0:image2.shape[0], 0:self.image2.shape[1]] = self.image2
        # result = np.maximum(transformed, result)

        # result = cv2.addWeighted(transformed, 0.5, result, 0.5, 1)
        result = self.average(image1, image2)

        return result

    def average(self, image1: np.ndarray, image2: np.ndarray):
        assert(image1.shape == image2.shape)
        result = np.zeros(image1.shape, dtype='uint8')

        # image1 != 0 && image2 !=0:
        overlap = np.logical_and(
            np.all(np.not_equal(image1, [0, 0, 0]), axis=2),
            np.all(np.not_equal(image2, [0, 0, 0]), axis=2),
        )
        result[overlap] = np.average(
            np.array([image1[overlap], image2[overlap]]), axis=0) .astype(np.uint8)

        not_overlap = np.logical_not(overlap)
        result[not_overlap] = np.maximum(
            image1[not_overlap], image2[not_overlap])

        return result
        # for i in range(image1.shape[0]):
        #     for j in range(image1.shape[1]):
        #         if

    def get_transformed_size(self):
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
        sticher = Sticher(None, None, None, None)
        self.assertEqual((0, 0), sticher.get_transformed_position(0, 0))
        self.assertEqual((10, 20), sticher.get_transformed_position(10, 20))

        sticher.M[0, 2] = 20
        sticher.M[1, 2] = 10
        self.assertEqual((20, 10), sticher.get_transformed_position(0, 0))
        self.assertEqual((30, 30), sticher.get_transformed_position(10, 20))

        sticher.M = np.eye(3)
        sticher.M[0, 1] = 2
        sticher.M[1, 0] = 4
        self.assertEqual((0, 0), sticher.get_transformed_position(0, 0))
        self.assertEqual((50, 60), sticher.get_transformed_position(10, 20))

    def test_get_transformed_size(self):
        image1 = np.empty((500, 400, 3), dtype='uint8')
        image1[:, :] = 255, 150, 100
        image1[:, 399] = 10, 20, 200

        # show_image(image1)
        image2 = np.empty((400, 400, 3), dtype='uint8')
        image2[:, :] = 50, 150, 255

        sticher = Sticher(image1, image2, None, None)
        sticher.M[0, 2] = -20
        sticher.M[1, 2] = 10
        sticher.M[0, 1] = .2
        sticher.M[1, 0] = .1
        left, right, top, bottom = sticher.get_transformed_size()
        print(sticher.get_transformed_size())
        width = int(max(right, image2.shape[1]) - min(left, 0))
        height = int(max(bottom, image2.shape[0]) - min(top, 0))
        print(width, height)
        show_image(cv2.warpPerspective(image1, sticher.M, (width, height)))

    def test_stich(self):
        image1 = np.empty((500, 400, 3), dtype='uint8')
        image1[:, :] = 255, 150, 100
        image1[:, 399] = 10, 20, 200

        # show_image(image1)
        image2 = np.empty((400, 400, 3), dtype='uint8')
        image2[:, :] = 50, 150, 255

        points = np.float32([[0, 0], [20, 20], [12, 12], [40, 20]])
        sticher = Sticher(image1, image2, points, points)
        sticher.M[0, 2] = 20
        sticher.M[1, 2] = 10
        sticher.M[0, 1] = .2
        sticher.M[1, 0] = .1
        sticher.stich()


def main():
    unittest.main()


if __name__ == "__main__":
    # main()
    os.chdir(os.path.dirname(__file__))

    img1 = cv2.imread("../resource/5-down.jpg")
    img2 = cv2.imread("../resource/5-up.jpg")
    matcher = Matcher(img1, img2, Method.ORB)
    matcher.match(show_match=True)
    sticher = Sticher(img1, img2, matcher)
    sticher.stich()
    print("M: ", sticher.M)
