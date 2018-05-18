# -*- coding: utf-8 -*-
"""
Created on 2018-05-18 18:12:12
@Author: ZHAO Lingfeng
@Version : 0.0.1
"""
import time
import os

# import numpy as np
import cv2

import stitch

show_image = stitch.show_image


def main():
    os.chdir(os.path.dirname(__file__))

    image = [22]
    # image = [3, 19, 20, 22, 23, 24]
    for number in image:
        try:
            img1 = cv2.imread("../resource/{}-left.jpg".format(number))
            img2 = cv2.imread("../resource/{}-right.jpg".format(number))
            for method in (stitch.Method.SIFT, stitch.Method.ORB):
                for use_genetic in (True, False):
                    try:
                        print("Image {} start stitching, using {}".format(number, method))
                        if use_genetic:
                            print("Using the genetic method")

                        start_time = time.time()
                        stitcher = stitch.Stitcher(img1, img2, method, False)
                        stitcher.stich(max_match_lenth=40, use_partial=False,
                                       use_new_match_method=use_genetic, show_match_point=True, show_result=True, use_gauss_blend=False)
                        if use_genetic:
                            cv2.imwrite('../resource/thesis/{}-{}-genetic.jpg'.format(number, method), stitcher.image)
                        else:
                            cv2.imwrite('../resource/thesis/{}-{}.jpg'.format(number, method), stitcher.image)

                        print("Time: ", time.time() - start_time)
                        print("M: ", stitcher.M)
                        print("#===================================================#")

                    except Exception as e:
                        print("Error happens: ", e)
        except Exception as e:
            print("Error happens: ", e)


if __name__ == "__main__":
    main()
