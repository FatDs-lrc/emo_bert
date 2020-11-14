import os, glob
import shutil
import h5py
import numpy as np
import tools.istarmap
import multiprocessing
from tqdm import tqdm
from utils import get_basename, mkdir
from tasks.audio import *
from tasks.vision import *
from tasks.text import *
from tasks.common import *
from scheduler.gpu_scheduler import init_model_on_gpus
from scheduler.multiprocess_scheduler import simple_processer

def save_h5(feature, lengths, save_path):
    h5f = h5py.File(save_path, 'w')
    h5f['feature'] = feature
    h5f['lengths'] = lengths
    h5f.close()

def vision_process(paths):
    frame_dir = './data/frames'
    face_dir = './data/faces'
    get_frames = Video2Frame(save_root=frame_dir)
    get_faces = VideoFaceTracker(save_root=face_dir)
    get_activate_spk = ActiveSpeakerSelector()
    select_faces = FaceSelector()
    video_clip, audio_path = paths
    _frame_dir = get_frames(video_clip)
    _face_dir = get_faces(_frame_dir)
    face_paths = []
    active_spk_id = get_activate_spk(_face_dir, audio_path)
    if active_spk_id is None:
        count['no_active_spk'] += get_basename(_frame_dir)
    else:
        face_paths = select_faces(face_dir, active_spk_id)
        if len(face_paths) < 0.4 * len(glob.glob(os.path.join(frame_dir, '*.jpg'))):
            count['face_too_less'] += get_basename(_face_dir)
    
    return face_paths
    
if __name__ == '__main__':
    import sys
    print()
    print('----------------Feature Extracting Start---------------- ')
    print()
    device = int(sys.argv[1])
    num_worker = int(sys.argv[2])

    all_positive_clips = []
    with multiprocessing.Manager() as MG:

        transcripts_dir = './data/transcripts'
        video_clip_dir = './data/video_clips'
        audio_dir = './data/audio_clips'
        frame_dir = './data/frames'
        face_dir = './data/faces'
        comparE_tmp = './data/.tmp'
        feature_dir = './feature'

        # 流程
        extract_text = TranscriptExtractor(save_root=transcripts_dir)
        package_transcript = TranscriptPackager()
        detect_language = DetectLanguage()
        cut_video = VideoCutterOneClip(save_root=video_clip_dir) # VideoCutter(save_root=video_clip_dir)
        compile_features = CompileFeatures()
        filter_fun = FilterTranscrips(1)
        device = 0
        # 语音
        extract_audio = AudioSplitor(save_root=audio_dir)
        extract_comparE = ComParEExtractor()
        extract_vggish = VggishExtractor(device=device)

        # 文本
        bert_model = BertExtractor(device=device)
        
        # 视觉
        select_faces = FaceSelector()
        extract_denseface = DensefaceExtractor(device=device)

        all_count = {
            'No_transcripts': []
        }

        all_movies = []
        for _format in ['mkv', 'mp4', 'rmvb', 'avi', 'wmv', 'rm', 'ram']:
            all_movies += glob.glob(f'/data6/zjm/emobert/resources/raw_movies/*.{_format}')

        for i, movie in enumerate(all_movies):
            print('[Main]: Processing', movie)
            movie_name = get_basename(movie)
            movie_feature_dir = os.path.join(feature_dir, movie_name)
            mkdir(movie_feature_dir)
            count = MG.dict({
                'no_sentence': [],
                'too_short': [],
                'is_not_en': [],
                'no_active_spk': [],
                'face_too_less': [] 
            })

            pool = multiprocessing.Pool(num_worker)
            # 抽文本 # 存json/txt
            transcript_path = extract_text(movie)
            transcript_info = package_transcript(transcript_path)

            if transcript_info == None:
                all_count['No_transcripts'].append(movie)
                continue
        
            sentences = list(map(lambda  x: x['content'], transcript_info))

            # 检查是否句子长度>1
            _have_sentence = []
            for i, transcript in enumerate(transcript_info):
                if len(transcript['content']) > 0:
                    _have_sentence.append(transcript)
                else:
                    count['no_sentence'].append(transcripts['index'])

            transcript_info = _have_sentence
            
            # 过滤时间小于1s的句子
            print('[Main]: Start filtering Transcipts')
            is_long_enough = list(tqdm(pool.imap(filter_fun, transcript_info), total=len(transcript_info)))
            transcript_info = list(filter(lambda x: x[1], zip(transcript_info, is_long_enough)))
            count['too_short'] += [x for x,y in transcript_info if not y] 
            transcript_info = list(map(lambda x: x[0], transcript_info))
            print('[Main]: Remaining Transcipts: {}'.format(len(transcript_info)))
            
            # 保存句子
            # 抽bert
            # print('[Main]: Extract Bert features:')
            # text_contents = list(map(lambda x: x['content'], transcript_info))
            # bert_features = list(tqdm(map(lambda x: bert_model(x), text_contents), total=len(text_contents)))
            # bert_save_path = os.path.join(movie_feature_dir, 'text_bert.h5')
            # bert_features, bert_lengths = compile_features(bert_features)
            # print('[Main]: Bert feature:{} with length {}'.format(bert_features.shape, bert_lengths))
            # save_h5(bert_features, bert_lengths, bert_save_path)

            # 切视频
            print('[Main]: Start cutting video')
            video_clip_dir = list(tqdm(
                pool.istarmap(cut_video, [(movie, transcript) for transcript in transcript_info]), 
                    total=len(transcript_info)
                ))[0]
            all_video_clip = sorted(glob.glob(os.path.join(video_clip_dir, '*.mkv')), key=lambda x: int(x.split('/')[-1].split('.')[0]))
            print('[Main]: Total clips found:', len(all_video_clip))

            # 切语音:
            print('[Main]: Start extracting audio files')
            audio_paths = list(tqdm(pool.imap(extract_audio, all_video_clip), total=len(all_video_clip)))
            print('[Main]: Total wav found:', len(audio_paths))

            # 抽语音特征
            # print('[Main]: Extract comparE features:')
            # comparE_features = list(tqdm(pool.imap(extract_comparE, audio_paths), total=len(audio_paths)))
            # comparE_features, comparE_lengths = compile_features(comparE_features)
            # print('[Main]: ComparE feature:{} with length {}'.format(bert_features.shape, bert_lengths))
            # save_h5(comparE_features, comparE_lengths, os.path.join(movie_feature_dir, 'audio_comparE.h5'))

            # print('[Main]: Extract Vggish features:')
            # vggish_features = list(tqdm(map(lambda x: extract_vggish(x), audio_paths), total=len(audio_paths)))
            # vggish_save_path = os.path.join(movie_feature_dir, 'audio_vggish.h5')
            # vggish_features, vggish_lengths = compile_features(vggish_features)
            # print('[Main]: vggish feature:{} with length []'.format(vggish_features.shape, vggish_lengths))
            # save_h5(vggish_features, vggish_lengths, vggish_save_path)

            # 抽帧抽脸 先去除完全没有人脸的片段, 保存成一个json
            # 出现时间最长说话人占视频的20%(任何人脸) 保存成一个json
            # 删掉小于出现时长小于总时长20%的人脸再做检测
            # 说话人检测

            print('[Main]: start vision process:')
            input_param = list(zip(all_video_clip, audio_paths))
            face_paths = list(tqdm(pool.imap(vision_process, input_param, chunk_size=32), total=len(input_param)))
            # vision_rvision_process(input_param[0])
            print('[Main]: totally {} clips passed vision test'.format(len(face_paths)))
            positive_clips = list(filter(lambda  x: x[1], zip(all_video_clip, face_paths)))

            # # 抽denseface
            # print('[Main]: Extract Denseface features:')
            # denseface_features = []
            # for clip_name, face_path in tqdm(positive_clips):
            #     clip_denseface_feature = list(map(lambda x: extract_denseface(x), face_path))
            #     clip_denseface_feature = np.asarray(clip_denseface_feature)
            #     denseface_features.append(clip_denseface_feature)

            # denseface_features, denseface_lengths = compile_features(denseface_features)
            # print('[Main]: Denseface feature:{} with length {}'.format(denseface_features.shape, denseface_lengths))
            # save_h5(denseface_features, denseface_lengths, os.path.join(movie_feature_dir, 'vision_denseface.h5'))

            # 统计数据
            positive_clips = list(map(lambda x: x[0], positive_clips))
            all_positive_clips += positive_clips
            positive_clips = list(map(lambda x: '/'.join(x.split('/')[-2:]), positive_clips))
            positive_clips = list(map(lambda x: x[:x.rfind('.')] + '\n', positive_clips))
            mkdir("./data/check_data")
            save_path = os.path.join('./data/check_data', movie_name + '.txt')
            with open(save_path, 'w') as f:
                f.writelines(positive_clips)
            all_count[movie_name] = dict(count)
            json_path = os.path.join('./data/check_data', movie_name + '.json')
            json.dump(dict(count), open(json_path, 'w'), indent=4)

    # print(all_positive_clips)
    all_positive_clips = list(map(lambda x: '/'.join(x.split('/')[-2:]), all_positive_clips))
    all_positive_clips = list(map(lambda x: x[:x.rfind('.')]+'\n', all_positive_clips))
    print("Positive clips: {}".format(len(all_positive_clips)))
    with open('data/all_positive_clips.txt', 'w') as f:
        f.writelines(all_positive_clips)
    
    json_path = "data/negative_count.json"

    for key, value in all_count.items():
        if isinstance(value, multiprocessing.managers.DictProxy):
            all_count[key] = dict(value)
    
    json.dump(dict(all_count), open(json_path, 'w'), indent=4)
