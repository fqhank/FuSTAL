import os
import time
import copy
import json
import torch
import numpy as np
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore")
from core.utils import select_pairs
from core.model import CoLA
from core.loss import TotalLoss
from core.config import cfg
import core.utils as utils
from torch.utils.tensorboard import SummaryWriter
from core.utils import AverageMeter
from core.dataset import NpyFeature
from eval.eval_detection import ANETdetection

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID
    worker_init_fn = None
    if cfg.SEED >= 0:
        utils.set_seed(cfg.SEED)
        worker_init_fn = np.random.seed(cfg.SEED)

    utils.set_path(cfg)
    utils.save_config(cfg)

    model = CoLA(cfg)
    model = model.cuda()

    train_loader = torch.utils.data.DataLoader(
        NpyFeature(data_path=cfg.DATA_PATH, mode='train',
                        modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                        num_segments=cfg.NUM_SEGMENTS, supervision='weak',
                        class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='random'),
            batch_size=cfg.BATCH_SIZE,
            shuffle=True, num_workers=cfg.NUM_WORKERS,
            worker_init_fn=worker_init_fn)

    test_loader = torch.utils.data.DataLoader(
        NpyFeature(data_path=cfg.DATA_PATH, mode='test',
                        modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                        num_segments=cfg.NUM_SEGMENTS, supervision='weak',
                        class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='uniform'),
            batch_size=1,
            shuffle=False, num_workers=cfg.NUM_WORKERS,
            worker_init_fn=worker_init_fn)

    test_info = {"step": [], "test_acc": [], "average_mAP": [],
                "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [], 
                "mAP@0.4": [], "mAP@0.5": [], "mAP@0.6": [],
                "mAP@0.7": []}
    best_mAP = -1
    criterion = TotalLoss()

    cfg.LR = eval(cfg.LR)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR[0],
        betas=(0.9, 0.999), weight_decay=0.0005)

    writter = SummaryWriter(cfg.LOG_PATH)
        
    print('=> start training pseudo label generator...')
    for step in range(1, cfg.NUM_ITERS + 1):
        if step > 1 and cfg.LR[step - 1] != cfg.LR[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg.LR[step - 1]

        if (step - 1) % len(train_loader) == 0:
            loader_iter = iter(train_loader)

        batch_time = AverageMeter()
        losses = AverageMeter()
        
        end = time.time()
        cost = train_one_step(model, loader_iter, optimizer, criterion, writter, step)
        losses.update(cost.item(), cfg.BATCH_SIZE)
        batch_time.update(time.time() - end)
        end = time.time()
        if step == 1 or step % cfg.PRINT_FREQ == 0:
            print(('Step: [{0:04d}/{1}]\t' \
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    step, cfg.NUM_ITERS, batch_time=batch_time, loss=losses)))
            
        if step > -1 and step % cfg.TEST_FREQ == 0:

            mAP_50, mAP_AVG = test_all(model, cfg, test_loader, test_info, step, writter)

            if test_info["average_mAP"][-1] > best_mAP:
                best_mAP = test_info["average_mAP"][-1]
                best_test_info = copy.deepcopy(test_info)

                utils.save_best_record_thumos(test_info, 
                    os.path.join(cfg.OUTPUT_PATH, "best_results.txt"))

                torch.save(model.state_dict(), os.path.join(cfg.MODEL_PATH, \
                    "model_best.pth.tar"))

            print(('- Test result: \t' \
                   'mAP@0.5 {mAP_50:.2%}\t' \
                   'mAP@AVG {mAP_AVG:.2%} (best: {best_mAP:.2%})'.format(
                   mAP_50=mAP_50, mAP_AVG=mAP_AVG, best_mAP=best_mAP)))

    print(utils.table_format(best_test_info, cfg.TIOU_THRESH, '[Generator] THUMOS\'14 Performance'))

def train_one_step(model, loader_iter, optimizer, criterion, writter, step):
    model.train()
    
    data, label, _, _, _ = next(loader_iter)
    data = data.cuda()
    label = label.cuda()
    
    selected_pairs = select_pairs(label)
    optimizer.zero_grad()
    video_scores, contrast_pairs, _, _ = model(data)
    cost, loss = criterion(video_scores, label, contrast_pairs, selected_pairs)

    cost.backward()
    optimizer.step()

    for key in loss.keys():
        writter.add_scalar(key, loss[key].cpu().item(), step)
    
    return cost

@torch.no_grad()
def test_all(model, cfg, test_loader, test_info, step, writter=None, model_file=None):
    model.eval()

    if model_file:
        print('=> loading model: {}'.format(model_file))
        model.load_state_dict(torch.load(model_file))
        print('=> tesing model...')

    final_res = {'method': '[Generator]', 'results': {}}
    
    acc = AverageMeter()

    for data, label, _, vid, vid_num_seg in test_loader:
        data, label = data.cuda(), label.cuda()
        vid_num_seg = vid_num_seg[0].cpu().item()

        video_scores, _, actionness, cas = model(data)

        label_np = label.cpu().data.numpy()
        score_np = video_scores[0].cpu().data.numpy()
        
        pred_np = np.where(score_np < cfg.CLASS_THRESH, 0, 1)
        correct_pred = np.sum(label_np == pred_np, axis=1)
        acc.update(float(np.sum((correct_pred == cfg.NUM_CLASSES))), correct_pred.shape[0])

        pred = np.where(score_np >= cfg.CLASS_THRESH)[0]
        if len(pred) == 0:
            pred = np.array([np.argmax(score_np)])
        
        cas_pred = utils.get_pred_activations(cas, pred, cfg)
        aness_pred = utils.get_pred_activations(actionness, pred, cfg)
        proposal_dict = utils.get_proposal_dict(cas_pred, aness_pred, pred, score_np, vid_num_seg, cfg)

        final_proposals = [utils.nms(v, cfg.NMS_THRESH) for _,v in proposal_dict.items()]
        final_res['results'][vid[0]] = utils.result2json(final_proposals, cfg.CLASS_DICT)

    json_path = os.path.join(cfg.OUTPUT_PATH, 'result.json')
    json.dump(final_res, open(json_path, 'w'))
    
    anet_detection = ANETdetection(cfg.GT_PATH, json_path,
                                subset='val', tiou_thresholds=cfg.TIOU_THRESH,
                                verbose=False, check_status=False)
    mAP, average_mAP = anet_detection.evaluate()

    if writter:
        writter.add_scalar('Test Performance/Accuracy', acc.avg, step)
        writter.add_scalar('Test Performance/mAP@AVG', average_mAP, step)
        for i in range(cfg.TIOU_THRESH.shape[0]):
            writter.add_scalar('mAP@tIOU/mAP@{:.1f}'.format(cfg.TIOU_THRESH[i]), mAP[i], step)

    test_info["step"].append(step)
    test_info["test_acc"].append(acc.avg)
    test_info["average_mAP"].append(average_mAP)

    for i in range(cfg.TIOU_THRESH.shape[0]):
        test_info["mAP@{:.1f}".format(cfg.TIOU_THRESH[i])].append(mAP[i])
    return test_info['mAP@0.5'][-1], average_mAP
    
if __name__ == "__main__":
    main()