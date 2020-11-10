# sys模块提供了一系列有关Python运行环境的变量和函数。
import sys
# argparse是python用于解析命令行参数和选项的标准模块
import argparse
# wave模块提供了一个处理WAV声音格式的便利接口。它不支持压缩/解压，但是支持单声道/立体声。
import wave
# multiprocessing包是Python中的多进程管理包。该进程可以运行在Python程序内部编写的函数。
from multiprocessing import cpu_count
#
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import glob
from os.path import join, basename
import subprocess


# 重采样函数，将对应的文件重采样处理到对应路径
def resample(spk_folder, sampling_rate, origin_wavpath, target_wavpath):
    """
    在x帧重采样文件并保存到对应的输出文件夹中
    参数：spk_folder: 发音者文件夹
    参数：sampling_rate: 重采样率
    参数：origin_wavpath: 重采样的根源路径
    参数：target_wavpath: 重采样后的目标路径
    返回值：None
    """
    # 将发音者文件夹何根源路径相连接得到总的路径，并获取wav格式的文件，得到所有wav格式文件的列表
    wavfiles = [i for i in os.listdir(join(origin_wavpath, spk_folder)) if i.endswith('.wav')]
    # 遍历所有的wav文件
    for wav in wavfiles:
        # 连接目标路径
        folder_to = join(target_wavpath, spk_folder)
        # 创建对应的目标路径
        os.makedirs(folder_to, exist_ok=True)
        # 连接对应的重采样目标文件路径和源文件路径
        wav_to = join(folder_to, wav)
        wav_from = join(origin_wavpath, spk_folder, wav)
        # subprocess.call用来执行一些命令行的代码，使用元组包含每个命令行元素，元素中不能含有空格
        # sox是一个跨平台的命令行实用程序，可以将各种格式的音频文件转换为需要的其他格式
        subprocess.call(['sox', wav_from, '-r', str(sampling_rate), wav_to])

    return None


def resample_to_xk(sampling_rate, origin_wavpath, target_wavpath, num_workers=1):
    """
    准备在x帧处重新映射的文件夹
    参数：sampling_rate: 帧重采样率
    参数：origin_wavpath: 重采样的根源路径
    参数：target_wavpath: 重采样后的目标路径
    参数：num_workers: cpu工作数量
    返回值：None
    """
    # 根据目标文件夹路径新建对应的文件
    os.makedirs(target_wavpath, exist_ok=True)
    # 获取源文件夹
    # os.listdir()方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    spk_folders = os.listdir(origin_wavpath)
    print(f'>使用了{num_workers}个CPU!')
    # 利用ProcessPoolExecutor构造线程池
    executor = ProcessPoolExecutor(max_workers=num_workers)

    futures = []
    # 建立一个进度条
    for spk_folder in tqdm(spk_folders):
        # partial为偏函数，即默认给对应的函数设置默认值
        # 这里等价于resample(spk_folder, sampling_rate, origin_wavpath, target_wavpath)
        # 通过线程池的submit函数提交执行的函数到线程池中
        futures.append(executor.submit(partial(resample, spk_folder, sampling_rate, origin_wavpath, target_wavpath)))

    result_list = [future.result() for future in tqdm(futures)]
    print('完成：')
    print(result_list)

    return None


# 获取wav文件的采样率
def get_sampling_rate(file_name):
    """
    获取一个wav文件的采样率
    参数：file_name: wav文件路径
    返回值：wav文件的采样率
    """
    # 使用wave.open打开对应数据文件，rb代表读取二进制文件
    with wave.open(file_name, 'rb') as wave_file:
        # getframerate获取采样率
        sample_rate = wave_file.getframerate()

    return sample_rate


def split_data(paths):
    """
    Split path data into train test split.
    参数：paths: all wav paths of a speaker dir.
    返回值：train wav paths, test wav paths
    """
    indices = np.arange(len(paths))
    test_size = 0.1
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=1234)
    train_paths = list(np.array(paths)[train_indices])
    test_paths = list(np.array(paths)[test_indices])

    return train_paths, test_paths


# 将wav数据直接转换为MCEP特征
def get_spk_world_feats(spk_name, spk_paths, output_dir, sample_rate):
    """
    将wav文件转换为MCEP特征
    参数：spk_name: 发音者文件夹的名字
    参数：spk_paths: 发音者所有数据的路径
    参数：output_dir: 保存输出MECP特征的路径
    参数：sample_rate: wav文件的采样率
    返回值：None
    """
    # 基频数组
    f0s = []
    # 包络数组
    coded_sps = []
    # 取出对应发音者文件夹中的所有wav文件
    for wav_file in spk_paths:
        # 调用自定义utils文件中的world_encode_wav函数获取一个文件的基频和包络
        f0, _, _, _, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        f0s.append(f0)
        coded_sps.append(coded_sp)

    log_f0s_mean, log_f0s_std = logf0_statistics(f0s)
    coded_sps_mean, coded_sps_std = coded_sp_statistics(coded_sps)

    np.savez(join(output_dir, spk_name + '_stats.npz'),
             log_f0s_mean=log_f0s_mean,
             log_f0s_std=log_f0s_std,
             coded_sps_mean=coded_sps_mean,
             coded_sps_std=coded_sps_std)

    for wav_file in tqdm(spk_paths):
        wav_name = basename(wav_file)
        _, _, _, _, coded_sp = world_encode_wav(wav_file, fs=sample_rate)
        normalised_coded_sp = (coded_sp - coded_sps_mean) / coded_sps_std
        np.save(os.path.join(output_dir, wav_name.replace('.wav', '.npy')),
                normalised_coded_sp,
                allow_pickle=False)

    return None


# 将wav文件处理为MCEP
def process_spk(spk_path, mc_dir):
    """
    将发音者的wavs格式文件转换为MCEP（梅尔倒谱系数）数据
    参数：spk_path: 发音者的wav文件路径
    参数：mc_dir: 输出发音者数据的路径
    返回值：None
    """
    # 连接路径字符串
    spk_paths = glob.glob(join(spk_path, '*.wav'))
    # 找到要转换的wav文件的原本采样率
    sample_rate = get_sampling_rate(spk_paths[0])
    # 获取文件的文件名
    spk_name = basename(spk_path)
    # 调用get_spk_world_feats方法处理
    get_spk_world_feats(spk_name, spk_paths, mc_dir, sample_rate)

    return None


def process_spk_with_split(spk_path, mc_dir_train, mc_dir_test):
    """
    Perform train test split on a speaker and process wavs to MCEPs.
    参数：spk_path: path to speaker wav dir
    参数：mc_dir_train: output dir for speaker train data
    参数：mc_dir_test: output dir for speaker test data
    返回值：None
    """
    spk_paths = glob.glob(join(spk_path, '*.wav'))

    # find the samplng rate of the wav files you are about to convert
    sample_rate = get_sampling_rate(spk_paths[0])

    spk_name = basename(spk_path)
    train_paths, test_paths = split_data(spk_paths)

    get_spk_world_feats(spk_name, train_paths, mc_dir_train, sample_rate)
    get_spk_world_feats(spk_name, test_paths, mc_dir_test, sample_rate)

    return None


if __name__ == '__main__':
    # 获取命令行参数
    parser = argparse.ArgumentParser()
    # 设置默认值为yes值的y
    perform_data_split_default = 'y'

    # 如果需要进行数据分割
    # 源wav文件路径
    origin_wavpath_default = "./data/VCTK-Corpus/wav48"
    target_wavpath_default = "./data/VCTK-Corpus/wav16"

    # 如果不需要进行数据分割
    # 源训练数据
    origin_wavpath_train_default = ''
    # 源评估数据
    origin_wavpath_eval_default = ''
    # 目标训练数据
    target_wavpath_train_default = './data/VCC2018-Corpus/wav22_train'
    # 目标评估数据
    target_wavpath_eval_default = './data/VCC2018-Corpus/wav22_eval'

    # 已处理过的mc文件的位置
    mc_dir_train_default = './data/mc/train'
    mc_dir_test_default = './data/mc/test'

    # 数据拆分
    parser.add_argument('--perform_data_split', choices=['y', 'n'], default=perform_data_split_default,
                        help='执行随机数据拆分')

    # 重新采样
    parser.add_argument('--resample_rate', type=int, default=0, help='重采样率')

    # 如果执行数据拆分：
    parser.add_argument('--origin_wavpath', type=str, default=origin_wavpath_default,
                        help='重采样文件源路径')
    parser.add_argument('--target_wavpath', type=str, default=target_wavpath_default,
                        help='重采样文件目标路径')

    # 如果不执行数据拆分：
    parser.add_argument('--origin_wavpath_train', type=str, default=origin_wavpath_train_default,
                        help='重采样训练文件源路径')
    parser.add_argument('--origin_wavpath_eval', type=str, default=origin_wavpath_eval_default,
                        help='重采样训练文件源路径')
    parser.add_argument('--target_wavpath_train', type=str, default=target_wavpath_train_default,
                        help='重采样训练文件目标路径')
    parser.add_argument('--target_wavpath_eval', type=str, default=target_wavpath_eval_default,
                        help='重采样评估文件目标路径')

    # MCEP预处理
    parser.add_argument('--mc_dir_train', type=str, default=mc_dir_train_default, help='训练数据特征文件夹')
    parser.add_argument('--mc_dir_test', type=str, default=mc_dir_test_default, help='测试数据特征文件夹')
    parser.add_argument('--speaker_dirs', type=str, nargs='+', required=True, help='被处理过的发音者')
    parser.add_argument('--num_workers', type=int, default=None, help='使用的CPU数量')

    argv = parser.parse_args()

    perform_data_split = argv.perform_data_split
    resample_rate = argv.resample_rate
    origin_wavpath = argv.origin_wavpath
    target_wavpath = argv.target_wavpath
    origin_wavpath_train = argv.origin_wavpath_train
    origin_wavpath_eval = argv.origin_wavpath_eval
    target_wavpath_train = argv.target_wavpath_train
    target_wavpath_eval = argv.target_wavpath_eval
    mc_dir_train = argv.mc_dir_train
    mc_dir_test = argv.mc_dir_test
    speaker_dirs = argv.speaker_dirs
    num_workers = argv.num_workers if argv.num_workers is not None else cpu_count()

    # 进行重新采样
    if perform_data_split == 'n':
        if resample_rate > 0:
            print(f'训练集重采样，采样率为{resample_rate}，将路径：[{origin_wavpath_train}]的发音者文件重采样到：[{target_wavpath_train}]')
            resample_to_xk(resample_rate, origin_wavpath_train, target_wavpath_train, num_workers)
            print(f'评估集重采样，采样率为{resample_rate}，将路径：[{origin_wavpath_eval}]的发音者文件重采样到：[{target_wavpath_eval}]')
            resample_to_xk(resample_rate, origin_wavpath_eval, target_wavpath_eval, num_workers)
    else:
        if resample_rate > 0:
            print(f'重采样，采样率为{resample_rate}，将路径：[{origin_wavpath}]的发音者文件重采样到路径：[{target_wavpath}]')
            resample_to_xk(resample_rate, origin_wavpath, target_wavpath, num_workers)

    print('正在为处理过的MCEP数据新建文件夹...')
    os.makedirs(mc_dir_train, exist_ok=True)
    os.makedirs(mc_dir_test, exist_ok=True)

    num_workers = len(speaker_dirs)
    print(f'发音者数量：{num_workers}')
    executer = ProcessPoolExecutor(max_workers=num_workers)

    futures = []
    if perform_data_split == 'n':
        # 训练wavs数据
        working_train_dir = target_wavpath_train
        for spk in tqdm(speaker_dirs):
            print(speaker_dirs)
            spk_dir = os.path.join(working_train_dir, spk)
            # 调用process_spk方法处理源数据为MECP
            futures.append(executer.submit(partial(process_spk, spk_dir, mc_dir_train)))

        # 评估wavs数据
        working_eval_dir = target_wavpath_eval
        for spk in tqdm(speaker_dirs):
            spk_dir = os.path.join(working_eval_dir, spk)
            futures.append(executer.submit(partial(process_spk, spk_dir, mc_dir_test)))
    else:
        # 当前正在使用所有的拆分数据
        working_dir = target_wavpath
        # tqdm是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器
        # 即这里会显示一个进度条来表示遍历的进度
        for spk in tqdm(speaker_dirs):
            # 获取文件完成路径
            spk_dir = os.path.join(working_dir, spk)
            futures.append(executer.submit(partial(process_spk_with_split, spk_dir, mc_dir_train, mc_dir_test)))

    result_list = [future.result() for future in tqdm(futures)]
    print('完成：')
    print(result_list)

    sys.exit(0)
