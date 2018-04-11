# -*- coding: utf-8 -*-
"""
Created on 2018-04-02 22:12:12
@Author: ZHAO Lingfeng
@Version : 0.0.1
"""
from typing import Tuple

import cv2
import numpy as np


def difference(array: np.ndarray):
    x = []
    for i in range(len(array) - 1):
        x.append(array[i + 1] - array[i])

    return np.array(x)


def find_peek(array: np.ndarray):
    peek = difference(difference(array))
    # print(peek)
    peek_pos = np.argmax(peek) + 2
    return peek_pos


def k_means(points: np.ndarray):
    """返回一个数组经kmeans分类后的k值以及标签，k值由计算拐点给出

    Args:
        points (np.ndarray): 需分类数据

    Returns:
        Tuple[int, np.ndarry]: k值以及标签数组
    """

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS
    length = []
    max_k = min(10, points.shape[0])
    for k in range(2, max_k + 1):
        avg = 0
        for i in range(5):
            compactness, _, _ = cv2.kmeans(
                points, k, None, criteria, 10, flags)
            avg += compactness
        avg /= 5
        length.append(avg)

    peek_pos = find_peek(length)
    k = peek_pos + 2
    # print(k)
    return k, cv2.kmeans(points, k, None, criteria, 10, flags)[1]  # labels


def get_group_center(points1: np.ndarray, points2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """输入两个相对应的点对数组，返回经kmeans优化后的两个数组

    Args:
        points1 (np.ndarray): 数组一
        points2 (np.ndarray): 数组二

    Returns:
        Tuple[np.ndarray, np.ndarray]: 两数组
    """

    k, labels = k_means(points1)
    labels = labels.flatten()
    selected_centers1 = []
    selected_centers2 = []
    for i in range(k):
        center1 = np.mean(points1[labels == i], axis=0)
        center2 = np.mean(points2[labels == i], axis=0)
        # center1 = points1[labels == i][0]
        # center2 = points2[labels == i][0]

        selected_centers1.append(center1)
        selected_centers2.append(center2)

    selected_centers1, selected_centers2 = np.array(
        selected_centers1), np.array(selected_centers2)

    # return selected_centers1, selected_centers2
    # return np.append(selected_centers1, points1, axis=0), np.append(selected_centers2, points2, axis=0)
    return points1, points2


def main():
    x = np.array([[1, 1], [1, 2], [2, 2], [3, 3]], dtype=np.float32)
    y = np.array([[1, 1], [1, 2], [2, 2], [3, 3]], dtype=np.float32)
    print(get_group_center(x, y))
    pass


if __name__ == "__main__":
    main()
