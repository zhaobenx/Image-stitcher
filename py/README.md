# 图像拼接代码介绍

## 主入口

```python
matcher = Matcher(img1, img2, Method.SIFT)
matcher.match(show_match=True)
sticher = Sticher(img1, img2, matcher)
sticher.stich()
```

分为两部分，`Matcher`和`Sticher`，分别用作图像的内容识别及图像的拼接

## Matcher介绍

### 构造函数

```python
class Matcher():

    def __init__(self, image1: np.ndarray, image2: np.ndarray, method: Enum=Method.SURF, threshold=800) -> None:
        """输入两幅图像，计算其特征值

        Args:
            image1 (np.ndarray): 图像一
            image2 (np.ndarray): 图像二
            method (Enum, optional): Defaults to Method.SURF. 特征值检测方法
            threshold (int, optional): Defaults to 800. 特征值阈值

        """
        ...
```

此类用于输入两幅图像，计算其特征值，输入两幅图像分别为`numpy`数组格式的图像，其中的`method`参数要求输入SURF、SIFT或者ORB，`threshold`参数为特征值检测所需的阈值。

### 特征值计算

```python
    def compute_keypoint(self) -> None:
        """计算特征点

        Args:
            image (np.ndarray): 图像
        """
        ...
```

利用给出的特征值检测方法对图像进行特征值检测。

### 匹配

```python
    def match(self, max_match_lenth=20, threshold=0.04, show_match=False):
        """对图片进行匹配
            max_match_lenth (int, optional): Defaults to 20. 最大匹配点数量
            threshold (float, optional): Defaults to 0.04. 默认最大匹配距离差
            show_match (bool, optional): Defaults to False. 是否展示匹配结果
        """
        ...
```

对两幅图片计算得出的特征值进行匹配，对ORB来说使用OpenCV的`BFMatcher`算法，而对于其他特征检测方法则使用`FlannBasedMatcher`算法。

## Sticher介绍

### 构造函数

```python
class Sticher:

    def __init__(self, image1: np.ndarray, image2: np.ndarray, matcher: Matcher):
        """输入图像和匹配，对图像进行拼接
        目前采用简单矩阵匹配和平均值拼合

        Args:
            image1 (np.ndarray): 图像一
            image2 (np.ndarray): 图像二
            matcher (Matcher): 匹配结果
        """
        ...
```

输入图像和匹配，对图像进行拼接，目前采用简单矩阵匹配和平均值拼合。

### 拼合

```python
    def stich(self, show_result=True, show_match_point=True):
        """对图片进行拼合
            show_result (bool, optional): Defaults to True. 是否展示拼合图像
            show_match_point (bool, optional): Defaults to True. 是否展示拼合点
        """
        ...
```

对两幅图像进行拼合，采用透视变换矩阵，并利用平均值对图片进行无缝接合。

### 融合

```python
    def blend(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """对图像进行拼合

        Args:
            image1 (np.ndarray): 图像一
            image2 (np.ndarray): 图像二

        Returns:
            np.ndarray: 融合结果
        """
        ...
```

目前采用简单平均方式。

### 辅助函数

#### 平均值

```python
    def average(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarry:
        """平均算法拼合

        Args:
            image1 (np.ndarray): 图片一
            image2 (np.ndarray): 图片二

        Returns:
            np.ndarray: 拼合后图像
        """
        ...
```

返回两幅图片的平均值。

#### 边界计算

```python
    def get_transformed_size(self) ->Tuple[int, int, int, int]:
        """计算形变后的边界

        Returns:
            Tuple[int, int, int, int]: 分别为左右上下边界
        """
        ...
```

计算形变后的边界，从而对图片进行相应的位移，保证全部图像都出现在屏幕上。

#### 坐标变换

```python
    def get_transformed_position(self, x: Union[float, Tuple[float, float]], y: float=None, M=None) -> Tuple[float, float]:
        """求得某点在变换矩阵（self.M）下的新坐标

        Args:
            x (Union[float, Tuple[float, float]]): x坐标或(x,y)坐标
            y (float, optional): Defaults to None. y坐标，可无
            M (np.ndarry, optional): Defaults to None. 利用M进行坐标变换运算

        Returns:
            Tuple[float, float]:  新坐标
        """
        ...
```

求得某点在变换矩阵（self.M）下的新坐标，如有选参数`M`，则利用M进行坐标变换运算。