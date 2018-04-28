# 图像拼接

## 介绍

此Repo是有关图像拼接的毕业设计的代码实现

## 进度

### 2018年3月24日：

利用Python的OpenCV库实现了简单的图像拼接，示例：

| 左图 | 右图 | 结果 |
| :--: | :--: | :--: |
|   ![3-left](example/3-left.JPG)   |  ![3-right](example/3-right.JPG)    |   ![3-surf](example/3-surf.jpg)   |
| ![19-left](example/19-left.JPG) | ![19-right](example/19-right.JPG) | ![19-SIFT](example/19-SIFT.jpg) |

### 2018年4月1日：

阅读OpenCV的ORB代码，并进行改动并编译。[67d825](../../commit/67d825b4d58d8a625effdb2d2688caaee8f32c34)

关于ORB的分析见[orb解析](./doc/orb解析/orb解析.md)。



### 2018年4月11日：

利用K-means算法进行特征点匹配的筛选。[fb4d88](../../commit/fb4d88449815402e2f2fdd0692478866eb20a1f0)

结果：不理想



### 2018年4月12日：

局部变换矩阵，在整体变换矩阵的基础上对与偏心过大的点进行单独的变换。[ca567e](../../commit/ca567e5bd39e4dd077962cfe29c08a43bc17d392)

结果：不理想



### 2018年4月24日：

实现python版本的RANSAC算法，并在其基础上对其进行改进。e36663

结果：实现了和OpenCV的cv2.findHomography函数功能基本一致的结果。



### 2018年4月26日：

利用遗传算法对RANSAC进行改进。ce3a8b

结果：能初步达成效果，但最终效果不及RANSAC



### 2018年4月27日：

试图使用变换矩阵作为遗传算法的基因。af87e9

结果：没有明显提高。



## TODO

*   改进ORB的旋转不变性
*   改进ORB的匹配或者特征点选取方式以使其基金SIFT的拼接效果
*   在RANSAC算法的基础上改进特征点选择方式使得ORB的特征选取更有代表性