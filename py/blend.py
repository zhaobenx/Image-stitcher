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


def show_image(image: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).show()


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
        print("Calculating pyramid")
        la1 = self.get_laplacian_pyramid(self.image1)
        la2 = self.get_laplacian_pyramid(self.image2)

        gm = self.get_gaussian_pyramid(self.mask)

        result = np.zeros(self.image1.shape, int)
        # new_la = []
        for i in range(self.LEVEL):
            mask = next(gm)
            # new_la.append(next(la1) * mask + next(la2) * (1.0 - mask))

            result += (next(la1) * mask + next(la2) * (1.0 - mask)).astype(int)
            del mask
            print(i, " level blended")
        return np.clip(result, 0, 255).astype('uint8')
        
        # print("Rebuilding ")
        # return self.rebuild_image(new_la)

    @classmethod
    def get_laplacian_pyramid(cls, image: np.ndarray):
        # output = []
        last = image

        for i in range(cls.LEVEL - 1):
            this = gaussian_filter(last, (1, 1, 0))
            laplace = cls.subtract(last, this)
            # output.append(laplace)
            yield laplace
            last = this
        # output.append(last)
        yield last
        # return output

    @classmethod
    def get_gaussian_pyramid(cls, image: np.ndarray):
        # G = []
        tmp = image
        for i in range(cls.LEVEL):
            # G.append(tmp)
            yield tmp
            tmp = gaussian_filter(tmp, (1, 1, 0))

        # return G

    @staticmethod
    def rebuild_image(laplacian_pyramid: np.ndarray):
        result = np.sum(laplacian_pyramid, axis=0)
        return np.clip(result, 0, 255).astype('uint8')

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


def average_blend(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """平均算法拼合

    Args:
        image1 (np.ndarray): 图片一
        image2 (np.ndarray): 图片二

    Returns:
        np.ndarray: 拼合后图像
    """

    assert(image1.shape == image2.shape)
    result = np.zeros(image1.shape, dtype='uint8')

    # image1 != 0 && image2 !=0:
    overlap = np.logical_and(
        np.all(np.not_equal(image1, [0, 0, 0]), axis=2),
        np.all(np.not_equal(image2, [0, 0, 0]), axis=2),
    )
    # 重叠处用平均值
    result[overlap] = np.average(
        np.array([image1[overlap], image2[overlap]]), axis=0
    ) .astype(np.uint8)
    # 非重叠处采选最大值
    not_overlap = np.logical_not(overlap)
    result[not_overlap] = np.maximum(
        image1[not_overlap], image2[not_overlap])

    return result


def gaussian_blend(image1: np.ndarray, image2: np.ndarray, mask: np.ndarray, mask_blend=3):
    if mask_blend:
        mask = gaussian_filter(mask.astype(float), (mask_blend, mask_blend, 0))
    # show_image((mask * 255).astype('uint8'))
    # show_image(image1)
    # show_image(image2)

    return GaussianBlend(image1, image2, mask).blend()


def direct_blend(image1: np.ndarray, image2: np.ndarray, mask: np.ndarray, mask_blend=0):
    if mask_blend:
        mask = gaussian_filter(mask.astype(float), (mask_blend, mask_blend, 0))
    if np.issubdtype(mask.dtype, np.integer):
        mask = mask / 255
    # show_image((mask * 255).astype('uint8'))
    # show_image(image1)
    # show_image(image2)

    return (image1 * mask + image2 * (1 - mask)).astype('uint8')


def test():
    os.chdir(os.path.dirname(__file__))

    image1 = cv2.imread("../example/3-left.jpg")
    image2 = cv2.imread("../example/3-right.jpg")
    show_image(np.concatenate((image1, image2), axis=0))
    mask = np.zeros(image1.shape)
    mask[:600] = 1.0
    mask = gaussian_filter(mask, (5, 5, 0))
    show_image(gaussian_blend(image1, image2, mask))


def main():
    test()


if __name__ == "__main__":
    main()
