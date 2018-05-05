# -*- coding: utf-8 -*-
"""
Created on 2018-05-04 19:47:47
@Author: ZHAO Lingfeng
@Version : 0.0.1
"""
import os

import numpy as np
from scipy.ndimage.filters import gaussian_filter

#
import cv2


class Blend:
    pass


class GaussianBlend(Blend):

    LEVEL = 6

    def __init__(self, image1: np.ndarray, image2: np.ndarray, mask: np.ndarray):
        self.image1 = image1
        self.image2 = image2
        if np.issubdtype(mask.dtype, np.integer):
            self.mask = mask / 255
        else:
            self.mask = mask

    def blend(self):
        la1 = self.get_laplacian_pyramid(self.image1)
        la2 = self.get_laplacian_pyramid(self.image2)

        gm = self.get_gaussian_pyramid(self.mask)
        new_la = la1 * gm + la2 * (1.0 - gm)
        return self.rebuild_image(new_la)

    @classmethod
    def get_laplacian_pyramid(cls, image: np.ndarray):
        output = []
        last = image

        for i in range(cls.LEVEL - 1):
            this = gaussian_filter(last, (1, 1, 0))
            laplace = cls.subtract(last, this)
            output.append(laplace)
            last = this
        output.append(last)
        return np.array(output)

    @staticmethod
    def rebuild_image(laplacian_pyramid: np.ndarray):
        result = np.sum(laplacian_pyramid, axis=0)
        return np.clip(result, 0, 255).astype('uint8')

    @classmethod
    def get_gaussian_pyramid(cls, image: np.ndarray):
        G = []
        tmp = image
        for i in range(cls.LEVEL):
            G.append(tmp)
            tmp = gaussian_filter(tmp, (1, 1, 0))

        return np.array(G)

    @staticmethod
    def subtract(array1: np.ndarray, array2: np.ndarray):
        """give non minus subtract

        Args:
            array1 (np.ndarray): array1
            array2 (np.ndarray): array2

        Returns:
            np.ndarray: (array1 - array2)>0?(array1 - array2):0
        """
        array1 = array1.astype(int)
        array2 = array2.astype(int)
        result = array1 - array2
        # result[np.where(result < 0)] = 0

        return result  # .astype(np.uint8)


def gaussian_blend(image1: np.ndarray, image2: np.ndarray, mask: np.ndarray):
    return GaussianBlend(image1, image2, mask).blend()


def test():
    os.chdir(os.path.dirname(__file__))

    def show_image(image: np.ndarray) -> None:
        from PIL import Image
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).show()

    image1 = cv2.imread("../resource/1-left.jpg")
    image2 = cv2.imread("../resource/1-right.jpg")
    show_image(np.concatenate((image1, image2), axis=0))
    mask = np.zeros(image1.shape, 'uint8')
    mask[:600] = 255
    mask = gaussian_filter(mask, (5,5,0))
    show_image(mask)
    show_image(gaussian_blend(image1, image2, mask))


def main():
    test()


if __name__ == "__main__":
    main()
