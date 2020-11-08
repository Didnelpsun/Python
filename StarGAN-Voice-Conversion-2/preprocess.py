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


def resample(spk_folder, sampling_rate, origin_wavpath, target_wavpath):
    """
    Resample files to x frames and save to output dir.
    :param spk_folder: speaker dir
    :param sampling_rate: frame rate to resample to
    :param origin_wavpath: root path of all speaker folders to resample
    :param target_wavpath: root path of resampled speakers to output to
    :return: None
    """
    wavfiles = [i for i in os.listdir(join(origin_wavpath, spk_folder)) if i.endswith('.wav')]
    for wav in wavfiles:
        folder_to = join(target_wavpath, spk_folder)
        os.makedirs(folder_to, exist_ok=True)
        wav_to = join(folder_to, wav)
        wav_from = join(origin_wavpath, spk_folder, wav)
        subprocess.call(['sox', wav_from, '-r', str(sampling_rate), wav_to])

    return None


def resample_to_xk(sampling_rate, origin_wavpath, target_wavpath, num_workers=1):
    """
    Prepare folders for resmapling at x frames.
    :param sampling_rate: frame rate to resample to
    :param origin_wavpath: root path of all speaker folders to resample
    :param target_wavpath: root path of resampled speakers to output to
    :param num_workers: cpu workers
    :return: None
    """
    os.makedirs(target_wavpath, exist_ok=True)
    spk_folders = os.listdir(origin_wavpath)
    print(f'> Using {num_workers} workers!')
    executor = ProcessPoolExecutor(max_workers=num_workers)

    futures = []
    for spk_folder in tqdm(spk_folders):
        futures.append(executor.submit(partial(resample, spk_folder, sampling_rate, origin_wavpath, target_wavpath)))

    result_list = [future.result() for future in tqdm(futures)]
    print('Completed:')
    print(result_list)

    return None


def get_sampling_rate(file_name):
    """
    Get the sampling rate of a wav file.
    :param file_name: wav file path
    :return: frame rate of wav file
    """
    with wave.open(file_name, 'rb') as wave_file:
        sample_rate = wave_file.getframerate()

    return sample_rate


def split_data(paths):
    """
    Split path data into train test split.
    :param paths: all wav paths of a speaker dir.
    :return: train wav paths, test wav paths
    """
    indices = np.arange(len(paths))
    test_size = 0.1
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=1234)
    train_paths = list(np.array(paths)[train_indices])
    test_paths = list(np.array(paths)[test_indices])

    return train_paths, test_paths


def get_spk_world_feats(spk_name, spk_paths, output_dir, sample_rate):
    """
    Convert wav files to there MCEP features.
    :param spk_name: name of speaker dir
    :param spk_paths: paths of all wavs in speaker dir
    :param output_dir: dir to output MCEPs to
    :param sample_rate: frame rate of wav files
    :return: None
    """
    f0s = []
    coded_sps = []
    for wav_file in spk_paths:
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


def process_spk(spk_path, mc_dir):
    """
    Prcoess speaker wavs to MCEPs
    :param spk_path: path to speaker wav dir
    :param mc_dir: output dir for speaker data
    :return: None
    """
    spk_paths = glob.glob(join(spk_path, '*.wav'))

    # find the sampling rate of teh wav files you are about to convert
    sample_rate = get_sampling_rate(spk_paths[0])

    spk_name = basename(spk_path)

    get_spk_world_feats(spk_name, spk_paths, mc_dir, sample_rate)

    return None


def process_spk_with_split(spk_path, mc_dir_train, mc_dir_test):
    """
    Perform train test split on a speaker and process wavs to MCEPs.
    :param spk_path: path to speaker wav dir
    :param mc_dir_train: output dir for speaker train data
    :param mc_dir_test: output dir for speaker test data
    :return: None
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
                        help='Perform random data split.')

    # 重新采样
    parser.add_argument('--resample_rate', type=int, default=0, help='Resampling rate.')

    # 如果执行数据拆分：
    parser.add_argument('--origin_wavpath', type=str, default=origin_wavpath_default,
                        help='Original wavpath for resampling.')
    parser.add_argument('--target_wavpath', type=str, default=target_wavpath_default,
                        help='Target wavpath for resampling.')

    # 如果不执行数据拆分：
    parser.add_argument('--origin_wavpath_train', type=str, default=origin_wavpath_train_default,
                        help='Original wavpath for resampling train files.')
    parser.add_argument('--origin_wavpath_eval', type=str, default=origin_wavpath_eval_default,
                        help='Original wavpath for resampling eval files.')
    parser.add_argument('--target_wavpath_train', type=str, default=target_wavpath_train_default,
                        help='Target wavpath for resampling train files.')
    parser.add_argument('--target_wavpath_eval', type=str, default=target_wavpath_eval_default,
                        help='Target wavpath for resampling eval files.')

    # MCEP预处理
    parser.add_argument('--mc_dir_train', type=str, default=mc_dir_train_default, help='Dir for training features.')
    parser.add_argument('--mc_dir_test', type=str, default=mc_dir_test_default, help='Dir for testing features.')
    parser.add_argument('--speaker_dirs', type=str, nargs='+', required=True, help='Speakers to be processed.')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of cpus to use.')

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
            print(f'Resampling speakers in {origin_wavpath_train} to {target_wavpath_train} at {resample_rate}')
            resample_to_xk(resample_rate, origin_wavpath_train, target_wavpath_train, num_workers)
            print(f'Resampling speakers in {origin_wavpath_eval} to {target_wavpath_eval} at {resample_rate}')
            resample_to_xk(resample_rate, origin_wavpath_eval, target_wavpath_eval, num_workers)
    else:
        if resample_rate > 0:
            print(f'Resampling speakers in {origin_wavpath} to {target_wavpath} at {resample_rate}')
            resample_to_xk(resample_rate, origin_wavpath, target_wavpath, num_workers)

    print('正在为处理过的MCEP数据新建文件夹...')
    os.makedirs(mc_dir_train, exist_ok=True)
    os.makedirs(mc_dir_test, exist_ok=True)

    num_workers = len(speaker_dirs)
    print(f'进程并行数量：{num_workers}')
    executer = ProcessPoolExecutor(max_workers=num_workers)

    futures = []
    if perform_data_split == 'n':
        # 训练wavs数据
        working_train_dir = target_wavpath_train
        for spk in tqdm(speaker_dirs):
            print(speaker_dirs)
            spk_dir = os.path.join(working_train_dir, spk)
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
            # 获取文件完成路由
            spk_dir = os.path.join(working_dir, spk)
            futures.append(executer.submit(partial(process_spk_with_split, spk_dir, mc_dir_train, mc_dir_test)))

    result_list = [future.result() for future in tqdm(futures)]
    print('完成：')
    print(result_list)

    sys.exit(0)
