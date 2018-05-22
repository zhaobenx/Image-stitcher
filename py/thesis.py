# -*- coding: utf-8 -*-
"""
Created on 2018-05-18 18:12:12
@Author: ZHAO Lingfeng
@Version : 0.0.1
"""
import time
import os

import numpy as np
import cv2

import stitch

show_image = stitch.show_image


def main():
    os.chdir(os.path.dirname(__file__))
    base_path = "../../论文/md/img/comparison"
    # os.chdir(base_path)
    # image = [3]
    # image = [3, 19, 20, 22, 23, 24, 27, 29]
    image = ("Road", "Lake", "Tree", "Building", "School", "Grass", "Palace", "NewHarbor")
    for name in image:
        try:
            # img1 = cv2.imread(("/{}-left.jpg".format(name)))
            # img2 = cv2.imread(("/{}-right.jpg".format(name)).encode('gbk').decode())
            img1 = cv2.imdecode(np.fromfile(base_path + "/{}-left.jpg".format(name), dtype='uint8'), -1)
            img2 = cv2.imdecode(np.fromfile(base_path + "/{}-right.jpg".format(name), dtype='uint8'), -1)
            for method in (stitch.Method.SIFT, stitch.Method.ORB):
                for use_genetic in (True, False):
                    try:
                        print("Image {} start stitching, using {}".format(name, method))
                        if use_genetic:
                            print("Using the genetic method")

                        start_time = time.time()
                        stitcher = stitch.Stitcher(img1, img2, method, False)
                        stitcher.stich(max_match_lenth=40, use_partial=False,
                                       use_new_match_method=use_genetic, show_match_point=False, show_result=False, use_gauss_blend=False)
                        if use_genetic:
                            # cv2.imwrite('/result/{}-{}-genetic.jpg'.format(name,
                                                                        #    method), stitcher.image)
                            cv2.imencode('.jpg', stitcher.image)[1].tofile(base_path +
                                                                           '/result/{}-{}-genetic.jpg'.format(name, method))
                        else:
                            # cv2.imwrite('/result/{}-{}.jpg'.format(name, method), stitcher.image)
                            cv2.imencode('.jpg', stitcher.image)[1].tofile(base_path +
                                                                           '/result/{}-{}.jpg'.format(name, method))

                        print("Time: ", time.time() - start_time)
                        # print("M: ", stitcher.M)
                        print("#===================================================#")

                    except Exception as e:
                        print("Error happens: ", e)
        except Exception as e:
            print("Error happens: ", e)


if __name__ == "__main__":
    main()
