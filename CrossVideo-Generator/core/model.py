import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage

class Actionness_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(Actionness_Module, self).__init__()
        self.len_feature = len_feature
        self.f_embed = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
        )

        self.f_cls = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.f_embed(out)
        embeddings = out.permute(0, 2, 1)
        out = self.dropout(out)
        out = self.f_cls(out)
        cas = out.permute(0, 2, 1)
        aness = cas.sum(dim=2)
        return embeddings, cas, aness

# CoLA Pipeline
class CoLA(nn.Module):
    def __init__(self, cfg):
        super(CoLA, self).__init__()
        self.len_feature = cfg.FEATS_DIM
        self.num_classes = cfg.NUM_CLASSES

        self.aness_module = Actionness_Module(cfg.FEATS_DIM, cfg.NUM_CLASSES)

        self.softmax = nn.Softmax(dim=1)

        self.r_easy = cfg.R_EASY
        self.r_hard = cfg.R_HARD
        self.m = cfg.m
        self.M = cfg.M

        self.dropout = nn.Dropout(p=0.6)
        
    def select_topk_embeddings(self, scores, embeddings, k):
        _, index_DESC = scores.sort(descending=True, dim=1)
        index_topk = index_DESC[:, :k]
        index_topk = index_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, index_topk)
        return selected_embeddings
    
    def get_video_cls_scores(self, cas, k_easy):
        sorted_scores, _= cas.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :k_easy, :]
        video_scores = self.softmax(topk_scores.mean(1))
        return video_scores

    def easy_snippets_mining(self, aness, embeddings, k_easy):
        select_index = torch.ones_like(aness).cuda()
        select_index = self.dropout(select_index)

        aness_drop = aness * select_index

        aness_rev = torch.max(aness, dim=1, keepdim=True)[0] - aness
        aness_rev_drop = aness_rev * select_index

        easy_act = self.select_topk_embeddings(aness_drop, embeddings, k_easy)
        easy_bkg = self.select_topk_embeddings(aness_rev_drop, embeddings, k_easy)

        return easy_act, easy_bkg

    def hard_snippets_mining(self, aness, embeddings, k_hard):
        aness_np = aness.cpu().detach().numpy()
        aness_median = np.median(aness_np, 1, keepdims=True)
        aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)

        erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        index_region_inner = aness.new_tensor(erosion_m - erosion_M)
        aness_region_inner = aness * index_region_inner
        hard_act = self.select_topk_embeddings(aness_region_inner, embeddings, k_hard)

        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        index_region_outer = aness.new_tensor(dilation_M - dilation_m)
        aness_region_outer = aness * index_region_outer
        hard_bkg = self.select_topk_embeddings(aness_region_outer, embeddings, k_hard)

        return hard_act, hard_bkg

    def forward(self, x):
        num_segments = x.shape[1]
        k_easy = num_segments // self.r_easy
        k_hard = num_segments // self.r_hard

        embeddings, cas, aness = self.aness_module(x)

        easy_act, easy_bkg = self.easy_snippets_mining(aness, embeddings, k_easy)
        hard_act, hard_bkg = self.hard_snippets_mining(aness, embeddings, k_hard)
        
        video_scores = self.get_video_cls_scores(cas, k_easy)

        contrast_pairs = {
            'EA': easy_act,
            'HA': hard_act,
            'EB': easy_bkg,
            'HB': hard_bkg
        }
        
        return video_scores, contrast_pairs, aness, cas
