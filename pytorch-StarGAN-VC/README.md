## [基于StarGAN的语音转换模型](https://github.com/hujinsen/pytorch-StarGAN-VC)

这是一个Pythorch实现的论文: [StarGAN VC：星型生成对抗网络下的非并行多对多语音转换](https://arxiv.org/abs/1806.02169).

**转换后的语音示例位于*samples*和*results_2019-06-10*目录中**

## [依赖](https://github.com/hujinsen/pytorch-StarGAN-VC)

- Python 3.6+
- pytorch 1.0
- librosa
- pyworld
- tensorboardX
- scikit-learn

## [使用方式](https://github.com/hujinsen/pytorch-StarGAN-VC)

### 下载数据集

将vcc 2016数据集下载到当前目录。

```shell
python download.py
```

下载的zip文件解压到`./data/vcc2016_training`和`./data/evaluation_all`两个目录。

1. **训练集：** 在本文中，作者从目录`./data/vcc2016_training`选用**四个说话人**。所以我们将对应的文件夹（比如SF1,SF2,TM1,TM2）到`./data/speakers`.
2. **测试集：** 在本文中，作者从目录`./data/evaluation_all`选用**四个说话人**。所以我们将对应的文件夹（比如SF1,SF2,TM1,TM2）到`./data/speakers_test`.

那么数据目录会变成这样：

```null
data
├── speakers  (训练集)
│   ├── SF1
│   ├── SF2
│   ├── TM1
│   └── TM2
├── speakers_test (测试集)
│   ├── SF1
│   ├── SF2
│   ├── TM1
│   └── TM2
├── vcc2016_training (vcc 2016训练集)
│   ├── ...
├── evaluation_all (vcc 2016评价集，作为测试集合)
│   ├── ...
```

### 预处理

从每个语音片段中提取特征（mcep、f0、ap）。这些特性存储为npy文件。我们还计算了每个说话人的统计特征。

```shell
python preprocess.py
```

这个预处理很可能花几分钟！

### 训练

```shell
python main.py
```

### 转换

```shell
python main.py --mode test --test_iters 200000 --src_speaker TM1 --trg_speaker "['TM1','SF1']"
```

## [网络结构](https://github.com/hujinsen/pytorch-StarGAN-VC)

![Snip20181102_2](https://github.com/hujinsen/StarGAN-Voice-Conversion/raw/master/imgs/Snip20181102_2.png)

注：我们的实现遵循了原论文的网络结构，而[pytorch-StarGAN](https://github.com/liusongxiang/StarGAN-Voice-Conversion)的VC代码使用StarGAN的网络。两者都有可以产生良好的音质。

## [参考](https://github.com/hujinsen/pytorch-StarGAN-VC)

[tensorflow StarGAN-VC代码](https://github.com/hujinsen/StarGAN-Voice-Conversion)

[StarGAN代码](https://github.com/taki0112/StarGAN-Tensorflow)

[CycleGAN-VC代码](https://github.com/leimao/Voice_Converter_CycleGAN)

[pytorch-StarGAN-VC代码](https://github.com/liusongxiang/StarGAN-Voice-Conversion)

[StarGAN-VC论文](https://arxiv.org/abs/1806.02169)

[StarGAN论文](https://arxiv.org/abs/1806.02169)

[CycleGAN论文](https://arxiv.org/abs/1703.10593v4)

## 更新于2019/06/10

原实现的网络结构是原论文的网络结构，但为了达到更好的转换效果，本次更新做了如下修改：

- 无训练问题的分类器改进
- 更新损失函数
- 将鉴别器激活函数修改为tanh（双曲正切函数）

---

如果你觉得这个回购是好的，请**点星**！

你的鼓励是我最大的动力！
