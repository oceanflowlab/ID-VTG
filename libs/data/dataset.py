from collections import OrderedDict
from copy import deepcopy
from functools import partial
import json
import math
import os
import random
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
import torchvision.transforms as T

from .data_utils import trivial_batch_collator, worker_init_reset_seed
from .tokenizer import make_tokenizer

import time


datasets = dict()


def register_dataset(name):
    def decorator(module):
        datasets[name] = module
        return module
    return decorator


class BaseDataset(Dataset):

    def __init__(
        self,
        split,                  # data split, a tuple/list allowing concat of subsets
        is_training,            # whether in training mode

        anno_file,              # annotation json file
        vid_feat_dir,           # video feature directory or h5 file
        text_feat_dir,          # text feature directory or h5 file
        ext_score_dir,          # external score directory
        ext_image_dir,          # extra image information
        image_feat_dir,         # image features from clip
        tokenizer,              # tokenizer (optional)

        max_vid_len,            # max video length (#clips) in training
        max_text_len,           # max text length (#tokens) in training
        clip_size,              # number of frames per clip / feature
        clip_stride,            # temporal stride of clips (in frame)
        downsample_rate=1,      # down-sampling rate for video features
        to_fixed_len=False,     # whether to resize video features to max length

        normalize_vid=False,    # whether to normalize video features to unit length
        normalize_text=False,   # whether to normalize text features to unit length
        normalize_scores=True,  # whether to normalize external score using sigmoid
        temperature=1.0,        # sigmoid temperature for score normalization

        crop_ratio=(0.9, 1.0),  # random cropping of video features in training
        trunc_thresh=0.5,       # threshold for event truncation in training
        max_num_text=None,      # max number of text queries per video in training

        group_method="greedy",  # text grouping method ("greedy" | "random" | "all")
        num_epochs=1,           # number of epochs
        cur_level=None,
    ):
        super(BaseDataset, self).__init__()

        assert os.path.exists(anno_file)
        if not isinstance(split, (list, tuple)):
            split = (split,)
        if tokenizer is None:
            assert text_feat_dir is not None, (
                "text features must be given if tokenizer is not specified"
            )
        assert isinstance(downsample_rate, int) and downsample_rate >= 1
        if crop_ratio is not None:
            assert isinstance(crop_ratio, (list, tuple))

        self.split = split
        self.is_training = is_training
        self.epoch = 0

        self.anno_file = anno_file
        self.vid_feat_dir = vid_feat_dir
        self.text_feat_dir = text_feat_dir
        self.ext_score_dir = ext_score_dir
        self.ext_image_dir = ext_image_dir
        self.image_feat_dir = image_feat_dir

        self._vid_h5 = None
        self.text_feat_h5 = None

        self.tokenizer = tokenizer

        self.max_vid_len = max_vid_len
        self.max_text_len = max_text_len
        self.clip_size = clip_size
        self.clip_stride = clip_stride * downsample_rate
        self.downsample_rate = downsample_rate
        self.to_fixed_len = to_fixed_len

        self.normalize_vid = normalize_vid
        self.normalize_text = normalize_text
        self.normalize_scores = normalize_scores
        self.temperature = temperature

        self.crop_ratio = crop_ratio
        self.trunc_thresh = trunc_thresh
        self.max_num_text = max_num_text
        self.cur_level = cur_level
        if self.cur_level:
            print("self.cur_level is :", self.cur_level)

        self.is_text_h5 = (
            self.text_feat_dir is not None and
            isinstance(self.text_feat_dir, str) and
            self.text_feat_dir.lower().endswith('.h5')
        )
        self.is_vid_h5 = (
            self.vid_feat_dir is not None and
            isinstance(self.vid_feat_dir, str) and
            self.vid_feat_dir.lower().endswith('.h5')
        )

        self.vid_dict, self.text_dict, self.frame_dict = self._parse_annotations()

        self.group_method = group_method
        self.num_epochs = num_epochs


    def _normalize_gt_segments(self, raw_segment, duration):
        """
        Normalize GT segment format to shape [num_gt, 2].

        Compatible formats:
        [s, t]
        [[s1, t1], [s2, t2], ...]
        """
        gt_segments = np.asarray(raw_segment, dtype=np.float32)

        if gt_segments.ndim == 1:
            if gt_segments.size != 2:
                raise ValueError(f"Invalid segment shape: {gt_segments.shape}")
            gt_segments = gt_segments.reshape(1, 2)

        elif gt_segments.ndim == 2:
            if gt_segments.shape[1] != 2:
                raise ValueError(f"Invalid segment shape: {gt_segments.shape}")

        else:
            raise ValueError(f"Invalid segment shape: {gt_segments.shape}")

        gt_segments[:, 0] = np.clip(gt_segments[:, 0], 0.0, duration)
        gt_segments[:, 1] = np.clip(gt_segments[:, 1], 0.0, duration)

        valid = gt_segments[:, 1] > gt_segments[:, 0]
        gt_segments = gt_segments[valid]

        if gt_segments.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        return gt_segments.astype(np.float32)


    def _parse_annotations(self):
        with open(self.anno_file, 'r') as f:
            anno = json.load(f)

        anno_db = dict()
        for s in self.split:
            assert s in anno, 'split [{:s}] does not exist'.format(s)
            if self.cur_level:
                anno_db.update(anno[s][self.cur_level])
            else:
                anno_db.update(anno[s])

        vid_dict, text_dict, frame_dict = OrderedDict(), OrderedDict(), OrderedDict()

        for key, value in anno_db.items():
            if not self.is_vid_h5:
                file_path = os.path.join(self.vid_feat_dir, f"{key}.npy")
                if not os.path.exists(file_path):
                    continue

            if 'annotations' not in value:
                continue

            fps = float(value['fps'])
            num_frames = int(value['num_frames'])

            if 'duration' in value:
                duration = float(value['duration'])
            else:
                duration = num_frames / fps

            if 'num_clips' in value:
                num_clips = (
                    value['num_clips'] + self.downsample_rate - 1
                ) // self.downsample_rate
            else:
                num_clips = None

            text_ids, segments, frame_ids = tuple(), tuple(), tuple()
            texts = tuple()


            for s, pair in enumerate(value['annotations']):
                start = max(float(pair['segment'][0]), 0)
                end = min(float(pair['segment'][1]), duration)
                seg_len = end - start
                if seg_len <= 0:
                    continue
                segment = (start, end)

                text = pair['sentence'].strip()
                
                # text_id必须全局唯一
                if self.is_training:
                    text_id = pair.get('query_id', key + '_{:04d}'.format(s))
                else:
                    # text_id = pair.get('sentence_id', key + '_{:04d}'.format(s))
                    if 'text_id' in pair:
                        text_id = pair ['text_id']+'_{:04d}'.format(s)
                    else:
                        text_id= pair['query_id']
                    
                frame_id = pair ['query_id']
                    
                # print(pair["text_id"])
                # print(text_id)
                # print(frame_id)
                # print(text)
                # print(pair["image"])

                text_ids += (text_id,)
                texts += (text,)
                segments += (segment,)
                frame_ids += (frame_id,)

                text_dict[text_id] = {
                    'text': text,
                    'segment': np.array(segment)[None],
                    'text_idx': s,
                    'vid_id': key,
                }


                frame_dict[frame_id] = {
                    'frame_path': "",
                }

            if len(text_ids) == 0:
                continue
            if len(frame_ids) == 0:
                continue

            cur_vid_dict = {
                'fps': fps,
                'num_frames': num_frames,
                'num_clips': num_clips,
                'duration': duration,
                'text_ids': text_ids,
                'frame_ids': frame_ids,
                'segments': np.asarray(segments, dtype=np.float32),

                'texts': texts,
            }

            vid_dict[key] = cur_vid_dict

        return vid_dict, text_dict, frame_dict

    def __getstate__(self):
        """Ensure h5py file handles are not pickled when spawning workers."""
        state = self.__dict__.copy()
        state['_vid_h5'] = None
        state['text_feat_h5'] = None
        return state

    def _ensure_vid_h5_open(self):
        """Open the vid h5 file in the current process/worker if not opened yet."""
        if getattr(self, '_vid_h5', None) is None:
            if self.vid_feat_dir is None:
                raise RuntimeError("vid_feat_dir is None, cannot open h5 file")
            self._vid_h5 = h5py.File(self.vid_feat_dir, 'r')

    def _ensure_text_h5_open(self):
        """Open the text h5 file in the current process/worker if not opened yet."""
        if getattr(self, 'text_feat_h5', None) is None:
            if self.text_feat_dir is None:
                raise RuntimeError("text_feat_dir is None, cannot open h5 file")
            self.text_feat_h5 = h5py.File(self.text_feat_dir, 'r')

    def __del__(self):
        try:
            if getattr(self, '_vid_h5', None) is not None:
                self._vid_h5.close()
        except Exception:
            pass

        try:
            if getattr(self, 'text_feat_h5', None) is not None:
                self.text_feat_h5.close()
        except Exception:
            pass

    def _load_vid_feats(self, vid_id):
        if self.is_vid_h5:
            self._ensure_vid_h5_open()
            vid_feats = self._vid_h5[vid_id][:]
        else:
            file_path = os.path.join(self.vid_feat_dir, f"{vid_id}.npy")
            if os.path.exists(file_path):
                vid_feats = np.load(file_path)
            else:
                return None

        if self.downsample_rate > 1:
            vid_feats = vid_feats[::self.downsample_rate]

        vid_feats = vid_feats.transpose()  # (c, t)
        vid_feats = torch.from_numpy(np.ascontiguousarray(vid_feats))

        if self.normalize_vid:
            vid_feats = F.normalize(vid_feats, dim=0)

        return vid_feats

    def _truncate_vid_feats(
        self,
        feats,              # float tensor (c, t), full video features
        segments,           # float tensor (n, 2), event segments
        img_segments=None,  # optional: list/tuple of tensor (n_i, 2)
        offset=0.0,         # unused here, kept for compatibility
        num_trials=5000
    ):
        vid_len = feats.size(1)
        max_vid_len = self.max_vid_len

        if vid_len <= max_vid_len:
            if self.crop_ratio is None:
                if img_segments is None:
                    return feats, segments
                return feats, segments, img_segments

            max_vid_len = random.randint(
                max(int(np.ceil(self.crop_ratio[0] * vid_len)), 1),
                min(int(np.ceil(self.crop_ratio[1] * vid_len)), vid_len)
            )
            if max_vid_len == vid_len:
                if img_segments is None:
                    return feats, segments
                return feats, segments, img_segments

        s0 = max(0, int(np.floor(segments[:, 0].max().item() - max_vid_len)))
        s1 = min(vid_len - max_vid_len, int(np.ceil(segments[:, 1].min().item())))

        seg_lens = torch.clamp(segments[:, 1] - segments[:, 0], min=1e-5)

        for _ in range(num_trials):
            ws = random.randint(s0, s1)
            we = ws + max_vid_len

            start = torch.clamp(segments[:, 0], min=ws)
            end = torch.clamp(segments[:, 1], max=we)
            overlap = torch.clamp(end - start, min=0)

            if torch.all(overlap / seg_lens > self.trunc_thresh):
                feats = feats[:, ws:we]
                segments = torch.clamp(
                    segments - ws, min=0, max=we - ws
                )

                if img_segments is None:
                    return feats, segments

                new_img_segments = []
                for cur in img_segments:
                    cur = torch.as_tensor(cur, dtype=torch.float32)

                    if cur.numel() == 0:
                        new_img_segments.append(cur.reshape(0, 2))
                        continue

                    cur_start = torch.clamp(cur[:, 0], min=ws)
                    cur_end = torch.clamp(cur[:, 1], max=we)
                    cur_overlap = torch.clamp(cur_end - cur_start, min=0)

                    valid = cur_overlap > 0
                    cur = cur[valid]

                    if cur.numel() == 0:
                        new_img_segments.append(torch.zeros((0, 2), dtype=torch.float32))
                        continue

                    cur = torch.clamp(cur - ws, min=0, max=we - ws)
                    new_img_segments.append(cur)

                return feats, segments, new_img_segments

        raise ValueError('no valid truncation found')

    def _load_text_feats(self, text_id):
        if self.tokenizer is not None:
            text_feats = self.tokenizer(self.text_dict[text_id]['text'])
        else:
            if self.is_training:
                text_id=text_id
            else:
                text_id = text_id.split("_")[0]
            if self.is_text_h5:
                self._ensure_text_h5_open()
                text_feats = self.text_feat_h5[text_id][:].astype(np.float32)
            else:
                try:
                    text_feat_file = os.path.join(self.text_feat_dir, text_id + '.npy')
                    text_feats = np.load(text_feat_file).astype(np.float32)
                except Exception:
                    raise ValueError(
                        'failed to load features for sentence {:s}'.format(text_id)
                    )

            text_feats = text_feats.transpose()  # (c, t)
            text_feats = torch.from_numpy(np.ascontiguousarray(text_feats))

        if self.is_training and isinstance(text_feats, torch.Tensor):
            text_feats = text_feats[:, :self.max_text_len]

        if self.normalize_text and isinstance(text_feats, torch.Tensor):
            text_feats = F.normalize(text_feats, dim=0)

        return text_feats

    def _load_image_feats(self, frame_id):
        frame_feat_file = os.path.join(self.image_feat_dir, frame_id + '.npy')
        image_feats = np.load(frame_feat_file).astype(np.float32)
        image_feats = image_feats.transpose()
        image_feats = torch.from_numpy(np.ascontiguousarray(image_feats))
        return image_feats

    def _load_ext_scores(self, text_id):
        try:
            score_file = os.path.join(self.ext_score_dir, text_id + '.npy')
            scores = np.load(score_file).astype(np.float32)
        except Exception:
            raise ValueError(
                'failed to load external scores for sentence {:s}'.format(text_id)
            )

        if self.downsample_rate > 1:
            scores = scores[::self.downsample_rate]

        scores = torch.from_numpy(np.ascontiguousarray(scores))[None]  # (1, t)

        if self.normalize_scores:
            scores = torch.sigmoid(scores / self.temperature)

        return scores

    def _load_ext_image(self, text_id, target_size=224):
        image_path = self.frame_dict[text_id]['frame_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}. Returning a black image.")
            return torch.zeros((3, target_size, target_size))

        resized_img = image.resize((target_size, target_size), Image.BICUBIC)
        return T.ToTensor()(resized_img)

    def _avgpool_to_fixed_len(self, feats, size):
        vid_len = feats.size(1)
        sampling_ratio = math.ceil(vid_len / size)
        feats = F.interpolate(
            feats[None],
            size=size * sampling_ratio,
            mode='linear',
            align_corners=False
        )
        if sampling_ratio > 1:
            feats = F.avg_pool1d(feats, kernel_size=sampling_ratio)
        feats = feats[0]
        return feats

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


@register_dataset('video_centric')
class VideoCentricDataset(BaseDataset):
    """
    Dataset for video grounding where a training sample is defined by a
    video and a subset of its associated text queries.
    """

    def __init__(self, **kwargs):
        super(VideoCentricDataset, self).__init__(**kwargs)

        if self.is_training:
            self.data_list = self._build_train_samples()
        else:
            assert self.num_epochs == 1
            self.data_list = self._build_eval_samples()

    def _build_train_samples(self):
        samples = []
        for _ in range(self.num_epochs):
            for vid_id in self.vid_dict.keys():
                samples += self._group(vid_id)
        samples = samples[:len(samples) // self.num_epochs * self.num_epochs]
        return tuple(samples)

    def _build_eval_samples(self):
        samples = []
        for vid_id, vid_dict in self.vid_dict.items():
            samples += [(vid_id, tuple(range(len(vid_dict['segments']))))]
        return tuple(samples)

    def _group(self, vid_id):
        if self.to_fixed_len:
            return self._group_with_fixed_len(vid_id)
        return self._group_with_max_len(vid_id)

    def _group_with_fixed_len(self, vid_id):
        vid_dict = self.vid_dict[vid_id]
        idx = list(range(len(vid_dict['segments'])))

        if self.group_method in ("random", "all"):
            return [(vid_id, tuple(idx))]

        random.shuffle(idx)
        samples = []
        for i in range(0, len(idx), self.max_num_text):
            sample = (vid_id, tuple(idx[i:i + self.max_num_text]))
            samples += [sample]
        return samples

    def _group_with_max_len(self, vid_id):
        vid_dict = self.vid_dict[vid_id]

        if vid_dict['num_frames'] <= self.max_vid_len:
            win_len_frames = vid_dict['num_frames']
            if self.crop_ratio is not None:
                win_len_frames = max(int(np.ceil(self.crop_ratio[0] * win_len_frames)), 1)
        else:
            win_len_frames = self.max_vid_len

        win_len = win_len_frames / vid_dict['fps']

        sort_idx = np.argsort(vid_dict['segments'][:, 0])
        segments = vid_dict['segments'][sort_idx]
        mask = np.ones(len(segments), dtype=bool)

        samples = []
        while mask.sum() > 0:
            ptr = np.nonzero(mask)[0].min()

            ws, we = segments[ptr, 0], segments[ptr, 0] + win_len
            if segments[ptr, 1] - segments[ptr, 0] > win_len:
                idx = np.array([ptr])
            else:
                is_inside = (
                    (segments[:, 0] >= ws) & (segments[:, 1] <= we) & mask
                )
                idx = np.nonzero(is_inside)[0]
                if len(idx) > self.max_num_text:
                    idx = np.random.choice(idx, self.max_num_text, replace=False)

            sample = (vid_id, tuple(sort_idx[idx]))
            samples += [sample]
            mask[idx] = 0

        return samples

    def __len__(self):
        return len(self.data_list) // self.num_epochs

    def __getitem__(self, idx):
        vid_id, seg_idx = self.data_list[self.epoch * len(self) + idx]
        vid_dict = self.vid_dict[vid_id]

        # load video features (c, t)
        vid_feats = self._load_vid_feats(vid_id)
        vid_len = vid_feats.size(1)

        # resize video features and update clip stride / size
        clip_size, clip_stride = self.clip_size, self.clip_stride
        if self.to_fixed_len:
            vid_feats = self._avgpool_to_fixed_len(vid_feats, self.max_vid_len)
            clip_size = clip_stride = float(vid_len / self.max_vid_len)
        clip_offset = 0.0

        # locate timestamps in temporal feature grid
        ## NOTE: center feature around the middle frame of the clip
        segments = np.clip(
            vid_dict['segments'][np.array(seg_idx)] * vid_dict['fps'], 
            a_min=0, 
            a_max=(vid_dict['num_frames'])
        )
        if self.to_fixed_len:
            segments = segments * (self.max_vid_len / vid_len)
        segments = torch.from_numpy(
            np.ascontiguousarray(segments.astype(np.float32))
        )

        # truncate video features and update target segments
        if self.is_training:
            if not self.to_fixed_len:
                vid_feats, segments = self._truncate_vid_feats(
                    vid_feats, segments, offset=clip_offset
                )
        if self.group_method == "random" and len(seg_idx) > self.max_num_text:
                seg_idx = random.sample(seg_idx, k=self.max_num_text)
                segments = segments[seg_idx]
                

        # load text features / IDs
        text_feats_list = tuple()
        raw_texts_list = []
        image_feats_list = tuple()
        image_id_list=[]

        for local_i, orig_idx in enumerate(seg_idx):
            tid = vid_dict['text_ids'][orig_idx]
            text_feats = self._load_text_feats(tid)
            raw_texts = self.text_dict[tid]['text']
            text_feats_list += (text_feats,)
            raw_texts_list.append(raw_texts)
            image_feats = self._load_image_feats(vid_dict['frame_ids'][orig_idx])
            image_feats_list += (image_feats,)
            image_id_list.append(vid_dict['frame_ids'][orig_idx])

        # load external scores (only for inference)
        if not self.is_training and self.ext_score_dir is not None:
            ext_scores_list = tuple()
            for orig_idx in seg_idx:
                scores = self._load_ext_scores(vid_dict['text_ids'][orig_idx])
                if self.to_fixed_len:
                    scores = self._avgpool_to_fixed_len(scores, self.max_vid_len)
                ext_scores_list += (scores,)
            ext_scores = torch.cat(ext_scores_list)
        else:
            ext_scores = None



        if self.is_training:
            return {
                'num_frames': vid_dict['num_frames'],
                'duration': vid_dict['duration'],
                'segment'    : vid_dict['segments'],      # seconds
                'clip_size': clip_size,
                'clip_stride': clip_stride,
                'target': segments,                  # (m, 2), frame/grid unit
                'fps': vid_dict['fps'],
                'vid': vid_feats,
                'text': text_feats_list,
                'raw_text': raw_texts_list,
                'frame': image_feats_list,
                'ext_scores': ext_scores,
            }
        else:
            return {
                'duration': vid_dict['duration'],
                'segment': vid_dict['segments'],
                'clip_size': clip_size,
                'clip_stride': clip_stride,
                'target': segments,
                'fps': vid_dict['fps'],
                'vid': vid_feats,
                'text': text_feats_list,
                'raw_text': raw_texts_list,
                'frame': image_feats_list,
                'frame_ids': image_id_list,
                'ext_scores': ext_scores,
                'video_name': vid_id,
                'text_query': vid_dict['texts'],
                }

        


@register_dataset('text_centric')
class TextCentricDataset(BaseDataset):
    """
    Dataset for video grounding where a training sample is defined by a
    video-text pair (where the text serves as the probe) and optionally 
    includes addition text queries from the same video. The dataset size 
    is equal to the total number of text queries from all videos.

    Expected behavior:
    - train: a video + a single text query
    - eval: a video + a single text query
    """
    def __init__(self, **kwargs):
        super(TextCentricDataset, self).__init__(**kwargs)

        self.data_list = tuple(self.text_dict.keys())

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        text_id = self.data_list[idx]
        text_dict = self.text_dict[text_id]
        vid_id = text_dict['vid_id']
        vid_dict = self.vid_dict[vid_id]

        # load video features (c, t)
        vid_feats = self._load_vid_feats(vid_id)
        vid_len = vid_feats.size(1)

        # resize video features and update clip stride / size
        clip_size, clip_stride = self.clip_size, self.clip_stride
        if self.to_fixed_len:
            vid_feats = self._avgpool_to_fixed_len(vid_feats, self.max_vid_len)
            clip_size = clip_stride = float(
                ((vid_len - 1) * clip_stride + clip_size) / self.max_vid_len
            )
        clip_offset = 0.5 * clip_size / clip_stride

        # locate timestamps in temporal feature grid
        ## NOTE: center feature around the middle frame of the clip
        segments = np.clip(
            text_dict['segment'] * vid_dict['fps'], 
            a_min=0, a_max=vid_dict['num_frames']
        ) / clip_stride - clip_offset
        segments = torch.from_numpy(
            np.ascontiguousarray(segments.astype(np.float32))
        )

        # truncate video features and update target segments
        ## NOTE: use current text as probe
        if self.is_training and not self.to_fixed_len:
            vid_feats, segments = self._truncate_vid_feats(
                vid_feats, segments, clip_offset
            )

        # load text features
        text_feats = self._load_text_feats(text_id)

        # load external scores (only for inference)
        if not self.is_training and self.ext_score_dir is not None:
            ext_scores = self._load_ext_scores(text_id)
            if self.to_fixed_len:
                ext_scores = self._avgpool_to_fixed_len(
                    ext_scores, self.max_vid_len
                )
            ext_scores = ext_scores[0]
        else:
            ext_scores = None
        
        return {
                 'fps'        : vid_dict['fps'],        # frames per second
                 'num_frames' : vid_dict['num_frames'], # total number of frames
                 'duration'   : vid_dict['duration'],   # video duration in seconds
                 'segment'    : text_dict['segment'],   # ground-truth segments in seconds
                 'clip_size'  : clip_size,              # number of frames per clip
                 'clip_stride': clip_stride,            # effective clip stride
                 'target'     : segments,               # event segment in grid unit

                 'vid'        : vid_feats,              # video features (c2, t2)
                 'text'       : text_feats,             # text features (c1, t1)
                 'ext_scores' : ext_scores,             # external scores (t2, )
                }


def make_dataset(opt, num_epochs=1, is_training=True, cur_level=None):
    opt = deepcopy(opt)
    if 'tokenizer' in opt:
        tokenizer = make_tokenizer(opt.pop('tokenizer'))
    else:
        tokenizer = None
    if cur_level:
        opt['cur_level']=cur_level
        print("cur_level: ",cur_level)

    return datasets[opt.pop('name')](
        tokenizer=tokenizer, is_training=is_training, num_epochs=num_epochs,  **opt
    )


def make_dataloader(
    dataset,            # dataset
    generator,          # random number generator that controls worker seed
    batch_size,         # local batch size
    num_workers,        # local number of workers
    is_training,        # whether is in training
    world_size=1,       # number of processes (GPUs)
    rank=0,             # current process
):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=is_training)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=partial(worker_init_reset_seed, num_workers, rank),
        sampler=sampler,
        shuffle=(sampler is None and is_training),
        drop_last=is_training,
        generator=generator,
        persistent_workers=True if num_workers > 0 else False,
    )
    return loader, sampler