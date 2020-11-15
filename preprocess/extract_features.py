import os, glob
import h5py
import numpy as np
from tqdm import tqdm
from utils import get_basename, mkdir
from tasks.audio import *
from tasks.vision import *
from tasks.text import *

def extract_features_h5(extract_func, get_input_func, utt_ids, save_path):
    h5f = h5py.File(save_path, 'w')
    for utt_id in tqdm(utt_ids):
        input_param = get_input_func(utt_id)
        feature = extract_func(input_param)
        h5f[utt_id] = feature
    h5f.close()

def get_utt_id_files(meta_dir, file_name):
    files = glob.glob(os.path.join(meta_dir, f'*/{file_name}.txt'))
    files = sorted(files)
    movie_names = list(map(lambda x: x.split('/')[-2], files))
    return files, movie_names

if __name__ == '__main__':
    import sys
    utt_file_name = sys.argv[1]
    part_no, total = eval(sys.argv[2]), eval(sys.argv[3])
    device = 0
    meta_dir = 'data/meta'
    feature_root = 'feature'
    face_dir = 'data/faces'
    frame_dir = 'data/frames'
    audio_dir = 'data/audio_clips'
    tmp_dir = 'data/.tmp'
    extract_comparE = ComParEExtractor(tmp_dir=tmp_dir)
    extract_vggish = VggishExtractor(device=device)
    extract_denseface = DensefaceExtractor(device=device)
    all_utt_files, movie_names = get_utt_id_files(meta_dir, utt_file_name)
    length = len(all_utt_files)
    start = int(part_no * length / total)
    end = int((part_no + 1) * length / total)
    all_utt_files = all_utt_files[start: end]
    print('[Main]: utt_id files found:', len(all_utt_files))
    print('[Main]: movies to be processed:')
    print('-------------------------------------------------')
    for i, movie_name in enumerate(movie_names):
        print(f'[{i}]\t{movie_name}')
    print('-------------------------------------------------')
    for utt_file, movie_name in zip(all_utt_files, movie_names):
        feature_dir = os.path.join(feature_root, movie_name)
        mkdir(feature_dir)
        utt_ids = open(utt_file).readlines()
        utt_ids = list(map(lambda x: x.strip(), utt_ids))
        # comparE
        print('[Main]: processing comparE')
        save_path = os.path.join(feature_dir, f'{utt_file_name}_comparE.h5')
        extract_features_h5(extract_comparE, lambda x: os.path.join(audio_dir, x+'.wav'), 
                    utt_ids, save_path)
        print(f'[ComparE]: {movie_name} saved in {save_path}')
        # vggish
        print('[Main]: processing vggish')
        save_path = os.path.join(feature_dir, f'{utt_file_name}_vggish.h5')
        extract_features_h5(extract_vggish, lambda x: os.path.join(audio_dir, x+'.wav'), 
                    utt_ids, save_path)
        print(f'[Vggish]: {movie_name} saved in {save_path}')
        # denseface
        save_path = os.path.join(feature_dir, f'{utt_file_name}_vggish.h5')
        extract_features_h5(extract_denseface, lambda x: os.path.join(audio_dir, x+'.wav'), 
                    utt_ids, save_path)


    
