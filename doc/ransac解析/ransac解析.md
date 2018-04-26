# RANSAC算法与源码解析

>   来自OpenCV

## 算法介绍

 OpenCV中滤除误匹配对采用RANSAC算法寻找一个最佳单应性矩阵H，矩阵大小为3×3。RANSAC目的是找到最优的参数矩阵使得满足该矩阵的数据点个数最多，通常令$h_{33}$=1来归一化矩阵。由于单应性矩阵有8个未知参数，至少需要8个线性方程求解，对应到点位置信息上，一组点对可以列出两个方程，则至少包含4组匹配点对。



具体方法：

1.  随机从数据集中随机抽出4个样本数据 (此4个样本之间不能共线)，计算出变换矩阵H，记为模型M；
2.  计算数据集中所有数据与模型M的投影误差，若误差小于阈值，加入内点集 I ；
3.  如果当前内点集 $I$ 元素个数大于最优内点集 $I_{best} $, 则更新 $I_{best} = I$，同时更新迭代次数k ;
4.  如果迭代次数大于k,则退出 ; 否则迭代次数加1，并重复上述步骤；

  注：迭代次数k在不大于最大迭代次数的情况下，是在不断更新而不是固定的；

​                                $k = \frac{log(1-p)}{log(1-w^m)}$     

其中，p为置信度，一般取0.995；w为"内点"的比例 ; m为计算模型所需要的最少样本数=4；

其中代价函数为 :
$$
\sum ^{n}_{i=1}{[(x_i'-\frac{h_{11}x_i+h_{12}y_i+h_{13}}{h_{31}x_i+h_{32}y_i+h_{33}})^2 
+ (y_i'-\frac{h_{w1}x_i+h_{22}y_i+h_{23}}{h_{31}x_i+h_{32}y_i+h_{33}})^2
]}
$$

### 代码赏析

### 变换矩阵的求得

本项目中所用的变换矩阵函数，代码实现于`opencv/modules/calib3d/src/fundam.cpp`中的

```c++
cv::Mat cv::findHomography( InputArray _points1, InputArray _points2,
                            int method, double ransacReprojThreshold, OutputArray _mask,
                            const int maxIters, const double confidence)
```

从250行至433行。

### RANSAC筛选

其中调用的`createRANSACPointSetRegistrator`函数来进行RANSAC挑选，其实现于`opencv/modules/calib3d/src/ptsetreg.cpp`中的76至268行的`class RANSACPointSetRegistrator : public PointSetRegistrator`类中。

此类有四个方法，分别是：

`int findInliers( const Mat& m1, const Mat& m2, const Mat& model, Mat& err, Mat& mask, double thresh ) const` 检测给定点集、变换矩阵和阈值下内点的个数

`bool getSubset( const Mat& m1, const Mat& m2,Mat& ms1, Mat& ms2, RNG& rng,int maxAttempts=1000 ) const`返回是否有子集

`bool run(InputArray _m1, InputArray _m2, OutputArray _model, OutputArray _mask) const CV_OVERRIDE`

主函数

`void setCallback(const Ptr<PointSetRegistrator::Callback>& _cb) CV_OVERRIDE { cb = _cb; }`

其主要逻辑在于164行-258行的：

```c++
bool run(InputArray _m1, InputArray _m2, OutputArray _model, OutputArray _mask) const CV_OVERRIDE
```

关键代码：

```c++
        for( iter = 0; iter < niters; iter++ )//>迭代
        {
            int i, nmodels;
            if( count > modelPoints )
            {
                bool found = getSubset( m1, m2, ms1, ms2, rng, 10000 );//> rng为随机数， 10000为最大尝试次数， 即任选四组点对进行下一步计算
                if( !found )
                {
                    if( iter == 0 )
                        return false;//> 失败 点数不够
                    break;
                }
            }

            nmodels = cb->runKernel( ms1, ms2, model );//> 计算新的变换矩阵
            if( nmodels <= 0 )
                continue;
            CV_Assert( model.rows % nmodels == 0 );
            Size modelSize(model.cols, model.rows/nmodels);

            for( i = 0; i < nmodels; i++ )
            {
                Mat model_i = model.rowRange( i*modelSize.height, (i+1)*modelSize.height );
                int goodCount = findInliers( m1, m2, model_i, err, mask, threshold );//> 计算内点，err变量即为变量的偏差，返回为最佳匹配值

                if( goodCount > MAX(maxGoodCount, modelPoints-1) )//> 如果新得出的优秀匹配点比之前的多
                {
                    std::swap(mask, bestMask);//> 更新内点集，通过mask来标示
                    model_i.copyTo(bestModel);//> 新变换矩阵
                    maxGoodCount = goodCount;
                    niters = RANSACUpdateNumIters( confidence, (double)(count - goodCount)/count, modelPoints, niters );//> 更新阈值k
                }
            }
        }

```

105-162行，用以获取随机的四个点对：

```c++
bool getSubset( const Mat& m1, const Mat& m2,
                    Mat& ms1, Mat& ms2, RNG& rng,
                    int maxAttempts=1000 ) const
    {
        cv::AutoBuffer<int> _idx(modelPoints);
        int* idx = _idx;
        int i = 0, j, k, iters = 0;
        int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
        int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
        int esz1 = (int)m1.elemSize1()*d1, esz2 = (int)m2.elemSize1()*d2;
        int count = m1.checkVector(d1), count2 = m2.checkVector(d2);
        const int *m1ptr = m1.ptr<int>(), *m2ptr = m2.ptr<int>();

        ms1.create(modelPoints, 1, CV_MAKETYPE(m1.depth(), d1));
        ms2.create(modelPoints, 1, CV_MAKETYPE(m2.depth(), d2));

        int *ms1ptr = ms1.ptr<int>(), *ms2ptr = ms2.ptr<int>();

        CV_Assert( count >= modelPoints && count == count2 );
        CV_Assert( (esz1 % sizeof(int)) == 0 && (esz2 % sizeof(int)) == 0 );
        esz1 /= sizeof(int);
        esz2 /= sizeof(int);

        for(; iters < maxAttempts; iters++)
        {
            for( i = 0; i < modelPoints && iters < maxAttempts; )
            {
                int idx_i = 0;
                for(;;)//> 随机选取四个点
                {
                    idx_i = idx[i] = rng.uniform(0, count);
                    for( j = 0; j < i; j++ )
                        if( idx_i == idx[j] )
                            break;
                    if( j == i )
                        break;
                }
                for( k = 0; k < esz1; k++ )
                    ms1ptr[i*esz1 + k] = m1ptr[idx_i*esz1 + k];
                for( k = 0; k < esz2; k++ )
                    ms2ptr[i*esz2 + k] = m2ptr[idx_i*esz2 + k];
                if( checkPartialSubsets && !cb->checkSubset( ms1, ms2, i+1 ))//> 去除重复点
                {
                    // we may have selected some bad points;
                    // so, let's remove some of them randomly
                    i = rng.uniform(0, i+1);
                    iters++;
                    continue;
                }
                i++;
            }
            if( !checkPartialSubsets && i == modelPoints && !cb->checkSubset(ms1, ms2, i))
                continue;
            break;
        }

        return i == modelPoints && iters < maxAttempts;
    }

```



在53-73行，实现了迭代次数的判断  $k = \frac{log(1-p)}{log(1-w^m)}$   ：

```c++
int RANSACUpdateNumIters( double p, double ep, int modelPoints, int maxIters )
{
    if( modelPoints <= 0 )
        CV_Error( Error::StsOutOfRange, "the number of model points should be positive" );

    p = MAX(p, 0.);
    p = MIN(p, 1.);
    ep = MAX(ep, 0.);
    ep = MIN(ep, 1.);

    // avoid inf's & nan's
    double num = MAX(1. - p, DBL_MIN);
    double denom = 1. - std::pow(1. - ep, modelPoints);
    if( denom < DBL_MIN )
        return 0;

    num = std::log(num);
    denom = std::log(denom);

    return denom >= 0 || -num >= maxIters*(-denom) ? maxIters : cvRound(num/denom);//> 四舍五入
}
```

