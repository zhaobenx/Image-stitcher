import os

import cv2
import numpy as np
import matplotlib.pylab as plt
from PIL import Image

os.chdir(os.path.dirname(__file__))


# debug:


def log(*args):
    import logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s:\n %(message)s')
    logging.debug(''.join([str(x) for x in args]))


PIL_show = True


def show_image(img, *args, **kwargs):
    if PIL_show:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img.show()
    else:
        plt.axis("off")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.show(img)
        plt.show()


def read_image():
    img1 = cv2.imread("../resource/5-down.jpg")
    img2 = cv2.imread("../resource/5-up.jpg")
    # img1 = cv2.imread("../resource/4-right.jpg")
    # img2 = cv2.imread("../resource/4-left.jpg")
    # img1 = cv2.imread("../resource/3-right.jpg")
    # img2 = cv2.imread("../resource/3-left.jpg")
    # img1 = cv2.imread("../resource/1-right.jpg")
    # img2 = cv2.imread("../resource/1-left.jpg")
    return img1, img2


def compute_keypoint(image):
    surf = cv2.xfeatures2d.SURF_create(800)
    kps, des = surf.detectAndCompute(image, None)
    return kps, des


def main():
    img1, img2 = read_image()

    ''' Calculate keypoint and draw them '''
    kps1, des1 = compute_keypoint(img1)
    # img1 = cv2.drawKeypoints(img1, kps1, None, flags=0)
    kps2, des2 = compute_keypoint(img2)
    # img2 = cv2.drawKeypoints(img2, kps2, None, flags=0)

    '''匹配关键点'''
    matcher = cv2.FlannBasedMatcher()
    # matcher.add(des1)
    # matcher.train()

    match_points = sorted(matcher.match(des1, des2), key=lambda x: x.distance)
    log(type(match_points[0]))
    # log(match_points[-1].distance)
    # log(match_points[-10].distance)
    # log(match_points[1].distance)
    # log(match_points[10].distance)

    N = 20
    match_len = min(len(match_points), N)
    min_distance = max(2 * match_points[0].distance, 0.02)

    log('min distance: ', min_distance)

    for i in range(match_len):
        if match_points[i].distance > min_distance:
            match_len = i
            break

    match_points = match_points[:match_len]
    assert(match_len >= 4)

    log('match_points: %d' % match_len)

    '''形变'''
    img3 = cv2.drawMatches(img1, kps1, img2, kps2,
                           match_points, None, flags=0)

    image_points1, image_points2 = [], []
    for i in match_points:
        image_points1.append(kps1[i.queryIdx].pt)
        image_points2.append(kps2[i.trainIdx].pt)

    image_points1, image_points2 = np.array(
        image_points1), np.array(image_points2)
    # #img1 = cv2.drawKeypoints(img1, image_points1, None, flags=0)
    log(type(image_points1[0]))

    H, mask = cv2.findHomography(image_points1, image_points2, cv2.RANSAC)

    log("H size: %s" % str(H.shape))
    log("H: ", H)
    # res = cv2.warpAffine(
    #     img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    tmp1 = cv2.warpPerspective(
        img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    # res[0:img2.shape[0], 0:img2.shape[1]] = img2

    '''融合'''
    log('tmp1 type: %s' % tmp1.dtype)
    tmp2 = np.zeros(tmp1.shape, dtype='uint8')
    tmp2[0:img2.shape[0], 0:img2.shape[1]] = img2
    res = cv2.addWeighted(tmp1, 0.5, tmp2, 0.5, 1)
    # res = np.maximum(tmp1, tmp2)
    # res = tmp1

    show_image(img3)
    show_image(res)

    # h, w, _ = img1.shape
    # log("%d %d %d" % (h, w,  _))

    # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
    #                   [w - 1, 0]]).reshape(-1, 1, 2)

    # M = cv2.perspectiveTransform(pts, M)

    # dst = cv2.warpPerspective(img1, M, (300, 300))
    # # log("type of dst %s, shape: %s" % (type(dst), str(dst.shape)))
    # show_image(dst)


if __name__ == "__main__":
    # main()

    from stich import Sticher, Matcher, Method
    img1, img2 = read_image()
    matcher = Matcher(img1, img2, Method.ORB)
    matcher.match(show_match=True)
    sticher = Sticher(img1, img2, matcher)
    sticher.stich()
    print("M: ", sticher.M)
