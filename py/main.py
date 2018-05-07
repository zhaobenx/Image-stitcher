import os
import sys


import cv2

from stitch import Stitcher, Method

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
    {} path_to_image1 path_to_image2 ... path_to_imagen
-----
    To create stitched image
    """.format(os.path.basename(__file__)))
    exit(1)


def main():
    if len(sys.argv) < 2:
        show_help()

    images = read_image(*sys.argv[1:])
    image_last = images[0]
    for index, image in enumerate(images):
        if index == 0:
            continue
        if image is None:
            print('File not exist')
            sys.exit(1)
        stitcher = Stitcher(image_last, image, Method.SIFT, False)
        stitcher.stich(show_match_point=0)
        image_last = stitcher.image
        cv2.imwrite(sys.argv[1] + 'tmp-{}.jpg'.format(index), image_last)

    cv2.imwrite(sys.argv[1] + 'pano.jpg', image_last)
    print("Done")


if __name__ == "__main__":
    main()
