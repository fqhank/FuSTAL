import numpy as np
import torch
import core.utils as utils
import torch.utils.data as data
import os
import json

class NpyFeature(data.Dataset):
    def __init__(self, data_path, mode, modal, feature_fps, n_segments, sampling, class_dict, seed=-1, supervision='weak'):
        if seed >= 0:
            utils.set_seed(seed)

        self.mode = mode
        self.modal = modal
        self.feature_fps = feature_fps
        self.n_segments = n_segments

        if self.modal == 'all':
            self.feature_path = []
            for _modal in ['rgb', 'flow']:
                self.feature_path.append(os.path.join(data_path, 'features', self.mode, _modal))
        else:
            self.feature_path = os.path.join(data_path, 'features', self.mode, self.modal)

        split_path = os.path.join(data_path, 'split_{}.txt'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.strip())
        split_file.close()
        print('=> {} set has {} videos'.format(mode, len(self.vid_list)))

        anno_path = os.path.join(data_path, 'gt.json')
        anno_file = open(anno_path, 'r')
        self.anno = json.load(anno_file)
        anno_file.close()
        
        bkg_path = os.path.join(data_path, 'backgrounds.json')
        bkg_file = open(bkg_path, 'r')
        self.bkg = json.load(bkg_file)
        bkg_file.close()

        self.class_name_to_idx = class_dict
        self.n_classes = len(self.class_name_to_idx.keys())

        self.supervision = supervision
        self.sampling = sampling

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, vid_n_seg, sample_idx = self.get_data(index)
        label, temp_anno = self.get_label(index, vid_n_seg, sample_idx)
        if self.mode=='train':
            bkg_anno = self.get_bkg(index, vid_n_seg, sample_idx)
        else:
            bkg_anno = torch.Tensor(0)

        return data, label, temp_anno, self.vid_list[index], vid_n_seg

    def get_data(self, index):
        vid_name = self.vid_list[index]

        vid_n_seg = 0

        if self.modal == 'all':
            rgb_feature = np.load(os.path.join(self.feature_path[0],
                                    vid_name + '.npy')).astype(np.float32)
            flow_feature = np.load(os.path.join(self.feature_path[1],
                                    vid_name + '.npy')).astype(np.float32)

            vid_n_seg = rgb_feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(rgb_feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(rgb_feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            rgb_feature = rgb_feature[sample_idx]
            flow_feature = flow_feature[sample_idx]

            feature = np.concatenate((rgb_feature, flow_feature), axis=1)
        else:
            feature = np.load(os.path.join(self.feature_path,
                                    vid_name + '.npy')).astype(np.float32)

            vid_n_seg = feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            feature = feature[sample_idx]

        return torch.from_numpy(feature), vid_n_seg, sample_idx

    def get_label(self, index, vid_n_seg, sample_idx):
        vid_name = self.vid_list[index]
        anno_list = self.anno['database'][vid_name]['annotations']
        label = np.zeros([self.n_classes], dtype=np.float32)

        classwise_anno = [[]] * self.n_classes

        for _anno in anno_list:
            label[self.class_name_to_idx[_anno['label']]] = 1
            classwise_anno[self.class_name_to_idx[_anno['label']]].append(_anno)

        if self.supervision == 'weak':
            return label, torch.Tensor(0)
        else:
            temp_anno = np.zeros([vid_n_seg, self.n_classes])
            t_factor = self.feature_fps / 16

            for class_idx in range(self.n_classes):
                if label[class_idx] != 1:
                    continue

                for _anno in classwise_anno[class_idx]:
                    tmp_start_sec = float(_anno['segment'][0])
                    tmp_end_sec = float(_anno['segment'][1])

                    tmp_start = round(tmp_start_sec * t_factor)
                    tmp_end = round(tmp_end_sec * t_factor)

                    temp_anno[tmp_start:tmp_end+1, class_idx] = 1

            temp_anno = temp_anno[sample_idx, :]

            return label, torch.from_numpy(temp_anno)
        
    def get_bkg(self, index, vid_n_seg, sample_idx):
        vid_name = self.vid_list[index]
        bkg_list = self.bkg['results'][vid_name]

        if True:
            temp_bkg = np.zeros([vid_n_seg])
            t_factor = self.feature_fps / 16

            for _bkg in bkg_list:
                tmp_start_sec = float(_bkg['segment'][0])
                tmp_end_sec = float(_bkg['segment'][1])

                tmp_start = round(tmp_start_sec * t_factor)
                tmp_end = round(tmp_end_sec * t_factor)

                temp_bkg[tmp_start:tmp_end+1] = 1

            temp_bkg = temp_bkg[sample_idx]

            return torch.from_numpy(temp_bkg)

    def uniform_sampling(self, length):
        if length <= self.n_segments:
            return np.arange(length).astype(int)
        samples = np.arange(self.n_segments) * length / self.n_segments
        samples = np.floor(samples)
        return samples.astype(int)

    def random_perturb(self, length):
        if self.n_segments == length:
            return np.arange(self.n_segments).astype(int)
        samples = np.arange(self.n_segments) * length / self.n_segments
        for i in range(self.n_segments):
            if i < self.n_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)
