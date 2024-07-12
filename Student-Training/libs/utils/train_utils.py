import os
import pickle
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .postprocessing import postprocess_results
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm
from .cola_config import cfg
from scipy.interpolate import interp1d
from . import ANETdetection
from .eval_detection import ANETdetection_cola
import json
import torch.nn.functional as F
import copy

################################################################################
def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = False # True in origin
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(False) # True in origin
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return

def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer

def make_scheduler_actionformer(
    optimizer,
    optimizer_config,
    num_iters_per_epoch,
    last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=schedule_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler

def make_scheduler(
        optimizer,
        optimizer_config,
        num_iters_per_epoch,
        last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # get eta min
        eta_min = optimizer_config["eta_min"]

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                eta_min=eta_min,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get eta min
        eta_min = optimizer_config["eta_min"]

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                eta_min=eta_min,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=schedule_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEMA(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

################################################################################
def train_one_epoch(
        cola_train_loader,
        train_dataset,
        model,
        optimizer,
        scheduler,
        curr_epoch,
        model_ema=None,
        clip_grad_l2norm=-1,
        print_freq=20
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(cola_train_loader)
    # switch to train mode
    model.train()
    model_ema.module.eval()

    # main training loop
    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    
    if curr_epoch<=60:
        flag = 1
    elif curr_epoch<=100:
        flag = 0
    else:
        exit()
        
    iter_idx = -1
            
    for _, label, _, vid, _ in cola_train_loader:
        iter_idx += 1
        ################## by fqh ##################
        label = label.cuda()
        video_list = train_dataset.get_datadict(vid)
                
        # final finetune with EMA Teacher
        if not flag:
            print('on the processing of EMA final tunning')
            flag=0
            for j in range(len(video_list)):
                video = video_list[j].copy()
                pseudo_predicts = model_ema.module([video])
                if pseudo_predicts[0]['scores'].shape[0]==0:
                    flag=1
                    break
                
                mask = (pseudo_predicts[0]['scores']>0.4) # generate good pseudos
                if mask.sum() == 0:
                    mask[torch.argmax(pseudo_predicts[0]['scores'])] = True
                
                video_list[j]['labels'] = pseudo_predicts[0]['labels'][mask]
                pseudo_segs = (pseudo_predicts[0]['segments'][mask]*video_list[j]['fps'] - 0.5*16)/4
                pseudo_segs[pseudo_segs<0] = 0
                video_list[j]['segments'] = pseudo_segs
                video_list[j]['scores'] = pseudo_predicts[0]['scores'][mask]
            if flag==1:
                continue
        ################################################
        
        # zero out optim
        optimizer.zero_grad(set_to_none=True)
        # forward / backward the model
        losses = model(video_list, label)
        losses['final_loss'].backward()
        
        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensor board
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx

            # print to terminal
            block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            block3 = 'Loss {:.2f} ({:.2f})\n'.format(
                losses_tracker['final_loss'].val,
                losses_tracker['final_loss'].avg
            )
            block4 = ''
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block4 += '\t{:s} {:.2f} ({:.2f})'.format(
                        key, value.val, value.avg
                    )

            print('\t'.join([block1, block2, block3, block4]))
    
    # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return

@torch.no_grad()
def test_all(net, cfg, test_loader):
    net.eval()
    final_res = {'method': '[Test]', 'results': {}}
    
    n = 0
    N = 0
    for data, label, _, vid, vid_num_seg in test_loader:
        data, label = data.cuda(), label.cuda()
        vid_num_seg = vid_num_seg[0].cpu().item()

        video_scores, _, actionness, cas = net(data)

        label_np = label.cpu().data.numpy()
        score_np = video_scores[0].cpu().data.numpy()
        
        pred = np.where(score_np >= cfg.CLASS_THRESH)[0]
        if len(pred) == 0:
            pred = np.array([np.argmax(score_np)])

        cas_pred = get_pred_activations(cas, pred, cfg)
        aness_pred = get_pred_activations(actionness, pred, cfg)
        proposal_dict = get_proposal_dict(cas_pred, aness_pred, pred, score_np, vid_num_seg, cfg)
        
        final_proposals = [nms(v, cfg.NMS_THRESH) for _,v in proposal_dict.items()]
        selected_proposals = [[]]
        segs = []
        for j in range(len(final_proposals)):
            for k in final_proposals[j]:
                t_factor = k.pop(-1)
                selected_proposals[0].append(k) # add (s,e) to the list, prepare for localization 
        final_proposals = selected_proposals[0]
        final_res['results'][vid[0]] = result2json(final_proposals, cfg.CLASS_DICT)

    json_path = os.path.join('.', 'backgrounds.json')
    json.dump(final_res, open(json_path, 'w'))
    
    anet_detection = ANETdetection_cola(cfg.GT_PATH, json_path,
                                subset='test', tiou_thresholds=cfg.TIOU_THRESH,
                                verbose=False, check_status=False)
    mAP, average_mAP = anet_detection.evaluate()
    return mAP, average_mAP

def valid_one_epoch_ori(
        val_loader,
        model,
        curr_epoch,
        ext_score_file=None,
        evaluator=None,
        output_file=None,
        tb_writer=None,
        print_freq=20
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            output = model(video_list)

            # upack the results into ANet format
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    if evaluator is not None:
        if (ext_score_file is not None) and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
        # call the evaluator
        _, mAP = evaluator.evaluate(results, verbose=True)
    else:
        # dump to a pickle file that can be directly used for evaluation
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        mAP = 0.0

    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)

    return mAP

def valid_one_epoch(
        cola_test_loader,
        val_dataset,
        val_loader,
        model,
        curr_epoch,
        ext_score_file=None,
        evaluator=None,
        output_file=None,
        tb_writer=None,
        print_freq=20
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)
    
    final_res = {'method': '[Test]', 'results': {}}

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)

    # loop over validation set
    start = time.time()
    for data, label, _, vid, vid_num_seg in cola_test_loader:
        video_list = val_dataset.get_datadict(vid)
        # forward the model (wo. grad)
        with torch.no_grad():
            output = model(video_list)

            # upack the results into ANet format
            num_vids = len(output)
            final_proposals = [[]]
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    for j in range(len(output[vid_idx]['labels'])):
                        final_proposals[0].append([float(output[vid_idx]['labels'][j].item()),output[vid_idx]['scores'][j].item(),\
                                                output[vid_idx]['segments'][j, 0].item(),output[vid_idx]['segments'][j, 1].item()])
                    final_res['results'][vid[0]] = result2json(final_proposals, cfg.CLASS_DICT)

    json_path = os.path.join('.', 'result_val.json')
    json.dump(final_res, open(json_path, 'w'))
    
    # anet_detection = ANETdetection_cola(cfg.GT_PATH, json_path,
    #                             subset='test', tiou_thresholds=cfg.TIOU_THRESH,
    #                             verbose=False, check_status=False)
    anet_detection = ANETdetection_cola('/home/ma-user/work/CoLA-main/data/THUMOS14/gt.json', json_path,
                                subset='test', tiou_thresholds=cfg.TIOU_THRESH,
                                verbose=False, check_status=False)
    mAP, average_mAP = anet_detection.evaluate()
    print(mAP, average_mAP*100)
    return average_mAP*100

##################### module from CoLA ######################
@torch.no_grad()
def train_get_pseudo(video_scores, actionness, cas, vid_num_seg):
    # generate pseudo-segments for training, similar process as in original CoLA test.
    score_np = video_scores[0].clone().detach().cpu().data.numpy()
    pred = np.where(score_np >= cfg.CLASS_THRESH)[0]
    if len(pred) == 0:
        pred = np.array([np.argmax(score_np)])
    cas_pred = get_pred_activations(cas, pred, cfg)
    aness_pred = get_pred_activations(actionness, pred, cfg)
    proposal_dict = get_proposal_dict(cas_pred, aness_pred, pred, score_np, vid_num_seg, cfg)
    cas_norm = cas

    final_proposals = [nms(v, cfg.NMS_THRESH) for _,v in proposal_dict.items()]
    
    return final_proposals, cas_norm

def get_pseudo_label(video_scores, actionness, cas, vid_num_seg):
    t_factor = 16 / (24 * 25)
    gt_segments, gt_labels, gt_scores = [], [], [] # container
    gt_score_dstr = []
    for i in range(actionness.shape[0]): # get pseudo-label for each video in batch
        seg_lst_temp, lb_lst_temp, score_lst_temp = [], [], [] # temporary list to store segments and corresponding label, derived from original CoLA
        score_dstr_temp = []
        # generate pseudo proposals
        pseudo_proposals, cas_norm = train_get_pseudo(video_scores, actionness[i].unsqueeze(0).clone().detach().cpu(), cas[i].unsqueeze(0).clone().detach().cpu(), vid_num_seg) 
        for j in range(len(pseudo_proposals)):
            for k in pseudo_proposals[j]:
                c = cas_norm[0,int(k[2]):int(k[3])+1].mean(0).softmax(-1).numpy() # averaged soft prediction result for one action
                seg_lst_temp.append([k[2],k[3]]) # add (s,e) to the list, prepare for localization 
                lb_lst_temp.append(int(k[0])) 
                score_lst_temp.append(k[1])
                score_dstr_temp.append(c)
        gt_segments.append(torch.tensor(seg_lst_temp,dtype=torch.float).cuda()) # record segments for current video
        gt_labels.append(torch.tensor(lb_lst_temp).cuda()) # record labels for each segment
        gt_scores.append(torch.tensor(score_lst_temp))
        gt_score_dstr.append(torch.tensor(score_dstr_temp))
    return gt_segments, gt_labels, gt_scores, gt_score_dstr

def get_pred_activations(src, pred, config):
    src = minmax_norm(src)
    if len(src.size()) == 2:
        src = src.repeat((config.NUM_CLASSES, 1, 1)).permute(1, 2, 0)
    src_pred = src[0].cpu().numpy()[:, pred]
    src_pred = np.reshape(src_pred, (src.size(1), -1, 1))
    src_pred = upgrade_resolution(src_pred, config.UP_SCALE)
    return src_pred

def get_proposal_dict(cas_pred, aness_pred, pred, score_np, vid_num_seg, config):
    prop_dict = {}
    for th in config.CAS_THRESH:
        cas_tmp = cas_pred.copy()
        num_segments = cas_pred.shape[0]//config.UP_SCALE
        cas_tmp[cas_tmp[:, :, 0] < th] = 0
        seg_list = [np.where(cas_tmp[:, c, 0] > 0) for c in range(len(pred))]
        proposals = get_proposal_oic(seg_list, cas_tmp, score_np, pred, config.UP_SCALE, \
                        vid_num_seg, config.FEATS_FPS, num_segments)
        for i in range(len(proposals)):
            if len(proposals[i])==0:
                continue
            class_id = proposals[i][0][0]
            prop_dict[class_id] = prop_dict.get(class_id, []) + proposals[i]

    for th in config.ANESS_THRESH:
        aness_tmp = aness_pred.copy()
        num_segments = aness_pred.shape[0]//config.UP_SCALE
        aness_tmp[aness_tmp[:, :, 0] < th] = 0
        seg_list = [np.where(aness_tmp[:, c, 0] > 0) for c in range(len(pred))]
        proposals = get_proposal_oic(seg_list, cas_pred, score_np, pred, config.UP_SCALE, \
                        vid_num_seg, config.FEATS_FPS, num_segments)
        for i in range(len(proposals)):
            if len(proposals[i])==0:
                continue
            class_id = proposals[i][0][0]
            prop_dict[class_id] = prop_dict.get(class_id, []) + proposals[i]
    return prop_dict

def get_proposal_oic(tList, wtcam, final_score, c_pred, scale, v_len, sampling_frames, num_segments, _lambda=0.25, gamma=0.2):
    t_factor = (16 * v_len) / (scale * num_segments * sampling_frames)
    temp = []
    for i in range(len(tList)):
        c_temp = []
        temp_list = np.array(tList[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)
            for j in range(len(grouped_temp_list)):
                if len(grouped_temp_list[j]) < 2:
                    continue
                inner_score = np.mean(wtcam[grouped_temp_list[j], i, 0])
                len_proposal = len(grouped_temp_list[j])
                outer_s = max(0, int(grouped_temp_list[j][0] - _lambda * len_proposal))
                outer_e = min(int(wtcam.shape[0] - 1), int(grouped_temp_list[j][-1] + _lambda * len_proposal))
                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))               
                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(wtcam[outer_temp_list, i, 0])
                c_score = inner_score - outer_score + gamma * final_score[c_pred[i]]
                t_start = grouped_temp_list[j][0] * t_factor
                t_end = (grouped_temp_list[j][-1] + 1) * t_factor
                c_temp.append([c_pred[i], c_score, t_start, t_end, t_factor])
            temp.append(c_temp)
    return temp

def nms(proposals, thresh):
    proposals = np.array(proposals)
    x1 = proposals[:, 2]
    x2 = proposals[:, 3]
    scores = proposals[:, 1]

    areas = x2 - x1 + 1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(proposals[i].tolist())
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < thresh)[0]
        order = order[inds + 1]

    return keep

def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = torch.nn.ReLU()
        max_val = relu(torch.max(act_map, dim=1)[0])
        min_val = relu(torch.min(act_map, dim=1)[0])
    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta
    ret[ret > 1] = 1
    ret[ret < 0] = 0
    return ret

def upgrade_resolution(arr, scale):
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale

def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)

def preprocessing(video_list, padding_val=0.0):
    """
        Generate batched features and masks from a list of dict items
    """
    feats = [x['feats'] for x in video_list]
    feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
    max_len = feats_lens.max(0).values.item()

    if True:
        assert max_len <= 2304, "Input length must be smaller than max_seq_len during training"
        # set max_len to self.max_seq_len
        max_len = 2304
        # batch input shape B, C, T
        batch_shape = [len(feats), feats[0].shape[0], max_len]
        batched_inputs = feats[0].new_full(batch_shape, padding_val)
        for feat, pad_feat in zip(feats, batched_inputs):
            pad_feat[..., :feat.shape[-1]].copy_(feat)
        if False:
            # trick, adding noise slightly increases the variability between input features.
            noise = torch.randn_like(batched_inputs) * 0
            batched_inputs += noise
    return batched_inputs

def result2json(result, class_dict):
    result_file = []
    class_idx2name = dict((v, k) for k, v in class_dict.items())
    for i in range(len(result)):
        for j in range(len(result[i])):
            line = {'label': class_idx2name[result[i][j][0]], 'score': result[i][j][1],
                    'segment': [result[i][j][2], result[i][j][3]]}
            result_file.append(line)
    return result_file

def calculate_iou_matrix(segs1,segs2):
    iou_mtrx = segs1.unsqueeze(1).repeat(1,segs2.shape[0],1).cuda()
    iou_mtrx_T = segs2.unsqueeze(1).repeat(1,segs1.shape[0],1).permute(1,0,2).cuda()
    iou_mtrx_inter = torch.where(iou_mtrx[:,:,1]>iou_mtrx_T[:,:,1],iou_mtrx_T[:,:,1],iou_mtrx[:,:,1]) - torch.where(iou_mtrx[:,:,0]>iou_mtrx_T[:,:,0],iou_mtrx[:,:,0],iou_mtrx_T[:,:,0])
    iou_mtrx_right = torch.where(iou_mtrx[:,:,1]>iou_mtrx_T[:,:,1],iou_mtrx[:,:,1],iou_mtrx_T[:,:,1])
    iou_mtrx_left = torch.where(iou_mtrx[:,:,0]>iou_mtrx_T[:,:,0],iou_mtrx_T[:,:,0],iou_mtrx[:,:,0])
    
    iou_mtrx_inter[iou_mtrx_inter<0] = 0
    iou = iou_mtrx_inter / (iou_mtrx_right - iou_mtrx_left + 1e-9)
    return iou

def merge_box(segs,scores,labels):
    mask = (segs[:,1]!=segs[:,0])
    segs = segs[mask]
    scores = scores[mask]
    segs_list = []
    scores_list = []
    labels_list = []
    while segs.shape[0]!=0:
        iou = calculate_iou_matrix(segs,segs)
        max_idx = torch.argmax(scores)
        to_merge = (iou[max_idx]>0.1)
        merged = torch.tensor([torch.min(segs[to_merge,0],dim=0)[0],torch.max(segs[to_merge,1],dim=0)[0]])
        segs_list.append(merged)
        scores_list.append(scores[max_idx])
        labels_list.append(labels[max_idx])
        keep = torch.logical_not(to_merge)
        segs = segs[keep]
        scores = scores[keep]
    segs = torch.stack(segs_list)
    scores = torch.stack(scores_list)
    labels = torch.stack(labels_list)
    return segs,scores,labels
        
        
def valid_one_epoch_origin(
        val_loader,
        model,
        curr_epoch,
        ext_score_file=None,
        evaluator=None,
        output_file=None,
        tb_writer=None,
        print_freq=20
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start': [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            output = model(video_list)

            # upack the results into ANet format
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()

    if evaluator is not None:
        if (ext_score_file is not None) and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
        # call the evaluator
        _, mAP = evaluator.evaluate(results, verbose=True)
    else:
        # dump to a pickle file that can be directly used for evaluation
        with open(output_file, "wb") as f:
            pickle.dump(results, f)
        mAP = 0.0

    # log mAP to tb_writer
    if tb_writer is not None:
        tb_writer.add_scalar('validation/mAP', mAP, curr_epoch)

    return mAP

    
    