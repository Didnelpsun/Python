# 基于StarGAN的语音转换第二版

\** 转换过的样本即将推出  \**

一个基于: StarGAN-VC2: https://arxiv.org/pdf/1907.12279.pdf. 的pytorch实现。

* 最近的没有实现源目标对抗损失。
* 采用梯度惩罚。
* 没有在G中使用PS。

# 下载

**在Lunix虚拟机中使用Python3.6.2版本进行了测试**

建议使用linux环境——未针对mac或windows操作系统进行测试

## Python

* 使用Anaconda创建一个新环境

```shell
conda create -n stargan-vc python=3.6.2
```

* 下载conda的相关依赖

```shell
conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch
conda install pillow=5.4.1
conda install -c conda-forge librosa=0.6.1
conda install -c conda-forge tqdm=4.43.0
```

* 使用pip下载不能通过conda安装的依赖

```shell script
pip install pyworld=0.2.8
pip install mcd=0.4
```

**注释** 对于无法安装pyworld的mac用户，请参见：https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder

## 资源库

* 下载资源库
  * SoX: https://sourceforge.net/projects/sox/files/sox/14.4.2/ 
  * libsndfile: http://linuxfromscratch.org/blfs/view/svn/multimedia/libsndfile.html
  * yasm: http://www.linuxfromscratch.org/blfs/view/svn/general/yasm.html
  * ffmpeg: https://ffmpeg.org/download.html
  * libav: https://libav.org/download/

# 使用

## 下载数据集

* [VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)

```shell
mkdir ../data/VCTK-Data
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip?sequence=2&isAllowed=y
unzip VCTK-Corpus.zip -d ../data/VCTK-Data
```

如果下载的VCTK扩展名是tar.gz，运行此命令：

```shell
tar -xzvf VCTK-Corpus.tar.gz -C ../data/VCTK-Data
```

* VCC2016和2018尚未包括在内

## 预处理数据

我们将在这里使用梅尔倒谱系数（MCEPs）。

此示例脚本用于需要重新采样到16kHz的VCTK数据，该脚本允许您在不重采样的情况下对数据进行预处理。此脚本假定数据路径为`../data/VCTK-Data/`

```shell
# VCTK数据
python preprocess.py --perform_data_split y \
                     --resample_rate 16000 \
                     --origin_wavpath ../data/VCTK-Data/VCTK-Corpus/wav48 \
                     --target_wavpath ../data/VCTK-Data/VCTK-Corpus/wav16 \
                     --mc_dir_train ../data/VCTK-Data/mc/train \
                     --mc_dir_test ../data/VCTK-Data/mc/test \
                     --speaker_dirs p262 p272 p229 p232
```

# 训练

* 目前只测试了4个发音者之间的转换

* 尚未使用tensorboard进行测试

示例脚本：

```shell
# VCTK的示例
python main.py --train_data_dir ../data/VCTK-Data/mc/train \
               --test_data_dir ../data/VCTK-Data/mc/test \
               --use_tensorboard False \
               --wav_dir ../data/VCTK-Data/VCTK-Corpus/wav16 \
               --model_save_dir ../data/VCTK-Data/models \
               --sample_dir ../data/VCTK-Data/samples \
               --num_iters 200000 \
               --batch_size 8 \
               --speakers p262 p272 p229 p232 \
               --num_speakers 4
```

如果遇到以下错误：

```shell script
ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found
```

您可能需要导出用于导出的LD_LIBRARY_PATH：（查看[Stack Overflow](https://stackoverflow.com/questions/49875588/importerror-lib64-libstdc-so-6-version-cxxabi-1-3-9-not-found)）

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/<PATH>/<TO>/<YOUR>/.conda/envs/<ENV>/lib/
```

## 转换

例如：在步骤120000处恢复模型并指定发音者

```shell
# 使用VCTK作为示例
python convert.py --resume_model 120000 \
                  --sampling_rate 16000 \
                  --num_speakers 4 \
                  --speakers p262 p272 p229 p232 \
                  --train_data_dir ../data/VCTK-Data/mc/train/ \
                  --test_data_dir ../data/VCTK-Data/mc/test/ \
                  --wav_dir ../data/VCTK-Data/VCTK-Corpus/wav16 \
                  --model_save_dir ../data/VCTK-Data/models \
                  --convert_dir ../data/VCTK-Data/converted \
                  --num_converted_wavs 4
```

这会将转换后的文件保存到`../data/VCTK-Data/converted/120000/`

## 计算梅尔倒谱失真度

计算参考说话人（源发音者）与合成说话人的梅尔倒谱失真率。使用`--spk_to_spk`标记定义由转换脚本生成的多个发音者到发音者文件夹。

```shell
python mel_cep_distance.py --convert_dir ../data/VCTK-Data/converted/120000 \
                           --spk_to_spk p262_to_p272 \
                           --output_csv p262_to_p272.csv
```

# 代办事项

- [ ] 包括转换后的样本
- [ ] 包括MCD示例
- [ ] 包括如原始论文所谈到的源目标说你是

# StarGAN-Voice-Conversion-2

\** Converted samples coming soon  \**

A pytorch implementation based on: StarGAN-VC2: https://arxiv.org/pdf/1907.12279.pdf.

* Currently does not implement source-and-target adversarial loss.
* Makes use of gradient penalty.
* Doesnt make use of PS in G.

# Installation

**Tested on Python version 3.6.2 in a linux VM environment**

Recommended to use a linux environment - not tested for mac or windows OS 

## Python

* Create a new environment using Anaconda
```shell script
conda create -n stargan-vc python=3.6.2
```
* Install conda dependencies
```shell script
conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch
conda install pillow=5.4.1
conda install -c conda-forge librosa=0.6.1
conda install -c conda-forge tqdm=4.43.0
```

* Intall dependencies not available through conda using pip
```shell script
pip install pyworld=0.2.8
pip install mcd=0.4
```

**NB:** For mac users who cannot install pyworld see: https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder

## Libraries

* Install binaries
  * SoX: https://sourceforge.net/projects/sox/files/sox/14.4.2/ 
  * libsndfile: http://linuxfromscratch.org/blfs/view/svn/multimedia/libsndfile.html
  * yasm: http://www.linuxfromscratch.org/blfs/view/svn/general/yasm.html
  * ffmpeg: https://ffmpeg.org/download.html
  * libav: https://libav.org/download/

# Usage

## Download Dataset

* [VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)

```shell script
mkdir ../data/VCTK-Data
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip?sequence=2&isAllowed=y
unzip VCTK-Corpus.zip -d ../data/VCTK-Data
```

If the downloaded VCTK is in tar.gz, run this:

```shell script
tar -xzvf VCTK-Corpus.tar.gz -C ../data/VCTK-Data
```

* VCC2016 and 2018 are yet to be included

## Preprocessing data

We will use Mel-Cepstral coefficients(MCEPs) here.

This example script is for the VCTK data which needs resampling to 16kHz, the script allows you to preprocess the data without resampling either. This script assumes the data dir to be `../data/VCTK-Data/`

```shell script
# VCTK-Data
python preprocess.py --perform_data_split y \
                     --resample_rate 16000 \
                     --origin_wavpath ../data/VCTK-Data/VCTK-Corpus/wav48 \
                     --target_wavpath ../data/VCTK-Data/VCTK-Corpus/wav16 \
                     --mc_dir_train ../data/VCTK-Data/mc/train \
                     --mc_dir_test ../data/VCTK-Data/mc/test \
                     --speaker_dirs p262 p272 p229 p232
```

# Training

* Currently only tested with conversion between 4 speakers
* Not yet tested with use of tensorboard

Example script:
```shell script
# example with VCTK
python main.py --train_data_dir ../data/VCTK-Data/mc/train \
               --test_data_dir ../data/VCTK-Data/mc/test \
               --use_tensorboard False \
               --wav_dir ../data/VCTK-Data/VCTK-Corpus/wav16 \
               --model_save_dir ../data/VCTK-Data/models \
               --sample_dir ../data/VCTK-Data/samples \
               --num_iters 200000 \
               --batch_size 8 \
               --speakers p262 p272 p229 p232 \
               --num_speakers 4
```

If you encounter an error such as:

```shell script
ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found
```

You may need to export export LD_LIBRARY_PATH: (See [Stack Overflow](https://stackoverflow.com/questions/49875588/importerror-lib64-libstdc-so-6-version-cxxabi-1-3-9-not-found))

```shell script
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/<PATH>/<TO>/<YOUR>/.conda/envs/<ENV>/lib/
```

## Conversion

For example: restore model at step 120000 and specify the speakers

```shell script
# example with VCTK
python convert.py --resume_model 120000 \
                  --sampling_rate 16000 \
                  --num_speakers 4 \
                  --speakers p262 p272 p229 p232 \
                  --train_data_dir ../data/VCTK-Data/mc/train/ \
                  --test_data_dir ../data/VCTK-Data/mc/test/ \
                  --wav_dir ../data/VCTK-Data/VCTK-Corpus/wav16 \
                  --model_save_dir ../data/VCTK-Data/models \
                  --convert_dir ../data/VCTK-Data/converted \
                  --num_converted_wavs 4
```

This saves your converted flies to `../data/VCTK-Data/converted/120000/`

## Calculate Mel Cepstral Distortion

Calculate the Mel Cepstral Distortion of the reference speaker vs the synthesized speaker. Use `--spk_to_spk` tag to define multiple speaker to speaker folders generated with the convert script.

```shell script
python mel_cep_distance.py --convert_dir ../data/VCTK-Data/converted/120000 \
                           --spk_to_spk p262_to_p272 \
                           --output_csv p262_to_p272.csv
```

# TODO:
- [ ] Include converted samples
- [ ] Include MCD examples
- [ ] Include s-t loss like original paper
