---
title: TMGA
---

# A Method of Using Genetic Algorithm in Image Stitching

>   Lingfeng Zhao, Yujie Huang,Minge Jing,Xiaoyang Zeng, Yibo Fan

>   State Key Laboratory of ASIC & System, Fudan University, Shanghai, China

## Abstract

Image stitching is an important part of computer vision, and how to do it more efficiently with highquality is a heated topic. In this paper, the authors propose a new method called TMGA for image stitching to get an improved performance in calculating Transform Matrix by using Genetic Algorithm. The proposed TMGA not only counts the number of interior points, but also takes standard error and degree of dispersion into consideration compared the traditional methods. The results demonstrate that the proposed algorithm can gain a high-quality transform matrix and improves the result of the stitching. 

## Images

### Original Image

| Image name | Image1                                      | Image2                                         |
| ---------- | ------------------------------------------- | ---------------------------------------------- |
| Road   | ![Road-left](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/orig/Road-left.JPG) | ![Road-right](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/orig/Road-right.JPG) |
| Lake   | ![Lake-left](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/orig/Lake-left.JPG) | ![Lake-right](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/orig/Lake-right.JPG) |
| Tree   | ![Tree-left](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/orig/Tree-left.JPG) | ![Tree-right](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/orig/Tree-right.JPG) |
| Building   | ![Building-left](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/orig/Building-left.JPG) | ![Building-right](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/orig/Building-right.JPG) |
| School   | ![School-left](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/orig/School-left.JPG) | ![School-right](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/orig/School-right.JPG) |
| Grass   | ![Grass-left](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/orig/Grass-left.JPG) | ![Grass-right](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/orig/Grass-right.JPG) |
| Palace   | ![Palace-left](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/orig/Palace-left.JPG) | ![Palace-right](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/orig/Palace-right.JPG) |
| NewHarbor   | ![NewHarbor-left](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/orig/NewHarbor-left.JPG) | ![NewHarbor-right](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/orig/NewHarbor-right.JPG) |

### Stitching Result

| Image name | ORB with RANSAC | ORB with TMGA                         | SIFT with RANSAC | SIFT with TMGA |
| ---------- | ------------------------------------------- | ---------------------------------------------- | ---------- | ---------- |
| Road | ![Road-ORB-RANSAC](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Road-ORB-RANSAC.jpg) | ![Road-ORB-TMGA](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Road-ORB-TMGA.jpg) | ![Road-SIFT-RANSAC](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Road-SIFT-RANSAC.jpg) | ![Road-SIFT-TMGA](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Road-SIFT-TMGA.jpg) |
| Lake | ![Lake-ORB-RANSAC](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Lake-ORB-RANSAC.jpg) | ![Lake-ORB-TMGA](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Lake-ORB-TMGA.jpg) | ![Lake-SIFT-RANSAC](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Lake-SIFT-RANSAC.jpg) | ![Lake-SIFT-TMGA](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Lake-SIFT-TMGA.jpg) |
| Tree | ![Tree-ORB-RANSAC](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Tree-ORB-RANSAC.jpg) | ![Tree-ORB-TMGA](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Tree-ORB-TMGA.jpg) | ![Tree-SIFT-RANSAC](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Tree-SIFT-RANSAC.jpg) | ![Tree-SIFT-TMGA](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Tree-SIFT-TMGA.jpg) |
| Building | ![Building-ORB-RANSAC](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Building-ORB-RANSAC.jpg) | ![Building-ORB-TMGA](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Building-ORB-TMGA.jpg) | ![Building-SIFT-RANSAC](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Building-SIFT-RANSAC.jpg) | ![Building-SIFT-TMGA](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Building-SIFT-TMGA.jpg) |
| School | ![School-ORB-RANSAC](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/School-ORB-RANSAC.jpg) | ![School-ORB-TMGA](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/School-ORB-TMGA.jpg) | ![School-SIFT-RANSAC](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/School-SIFT-RANSAC.jpg) | ![School-SIFT-TMGA](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/School-SIFT-TMGA.jpg) |
| Grass | ![Grass-ORB-RANSAC](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Grass-ORB-RANSAC.jpg) | ![Grass-ORB-TMGA](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Grass-ORB-TMGA.jpg) | ![Grass-SIFT-RANSAC](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Grass-SIFT-RANSAC.jpg) | ![Grass-SIFT-TMGA](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Grass-SIFT-TMGA.jpg) |
| Palace | ![Palace-ORB-RANSAC](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Palace-ORB-RANSAC.jpg) | ![Palace-ORB-TMGA](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Palace-ORB-TMGA.jpg) | ![Palace-SIFT-RANSAC](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Palace-SIFT-RANSAC.jpg) | ![Palace-SIFT-TMGA](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/Palace-SIFT-TMGA.jpg) |
| NewHarbor | ![NewHarbor-ORB-RANSAC](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/NewHarbor-ORB-RANSAC.jpg) | ![NewHarbor-ORB-TMGA](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/NewHarbor-ORB-TMGA.jpg) | ![NewHarbor-SIFT-RANSAC](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/NewHarbor-SIFT-RANSAC.jpg) | ![NewHarbor-SIFT-TMGA](https://github.com/zhaobenx/Image-stitcher/raw/master/docs/TMGA/img/result/NewHarbor-SIFT-TMGA.jpg) |

