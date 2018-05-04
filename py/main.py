import os
import sys


import cv2

from stich import Stitcher, Matcher, Method

# os.chdir(os.path.dirname(__file__))
# os.chdir(os.getcwd())
# debug:


def log(*args):
    import logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s:\n %(message)s')
    logging.debug(''.join([str(x) for x in args]))


def read_image(*images):
    res = []
    for i in images:
        res.append(cv2.imread(i))
    return tuple(res)


def show_help():
    print("""
Usage:
    {} path_to_image1 path_to_image2
-----
    To create stiched image from that two
    """.format(os.path.basename(__file__))

          )
    exit(1)


def main():
    if len(sys.argv) < 2:
        show_help()

    img1, img2 = read_image(*sys.argv[1:])
    if img1 is None or img2 is None:
        print('File not exist')
        sys.exit(1)
    matcher = Matcher(img1, img2, Method.SIFT)
    matcher.match(show_match=True)
    stitcher = Stitcher(img1, img2, matcher)
    stitcher.stich()
    cv2.imwrite(sys.argv[1] + 'pano.jpg', stitcher.image)
    print("M: ", stitcher.M)


if __name__ == "__main__":
    main()
