import torch
import torch.nn as nn

class CrossVideoLoss(nn.Module):
    def __init__(self):
        super(CrossVideoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.07):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):
        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['HA'], 1), 
            torch.mean(contrast_pairs['EA'], 1), 
            contrast_pairs['EB']
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['HB'], 1), 
            torch.mean(contrast_pairs['EB'], 1), 
            contrast_pairs['EA']
        )
        
        loss = HA_refinement + HB_refinement
        return loss   

class ActionLoss(nn.Module):
    def __init__(self):
        super(ActionLoss, self).__init__()
        self.bce_criterion = nn.BCELoss()

    def forward(self, video_scores, label):
        label = label / torch.sum(label, dim=1, keepdim=True)
        loss = self.bce_criterion(video_scores, label)
        return loss

class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.action_criterion = ActionLoss()
        self.cv_criterion = CrossVideoLoss()

    def forward(self, video_scores, label, contrast_pairs, pairs):
        loss_cls = self.action_criterion(video_scores, label)
        loss_cv = self.cv_criterion(contrast_pairs)
        
        # cross video loss
        for i,vs in enumerate(pairs):
            temp_pairs = {
            'EA': torch.stack([contrast_pairs['EA'][list(pairs[i])[0]],contrast_pairs['EA'][list(pairs[i])[1]]]),
            'HA': torch.stack([contrast_pairs['HA'][list(pairs[i])[1]],contrast_pairs['HA'][list(pairs[i])[0]]]),
            'EB': torch.stack([contrast_pairs['EB'][list(pairs[i])[0]],contrast_pairs['EB'][list(pairs[i])[1]]]),
            'HB': torch.stack([contrast_pairs['HB'][list(pairs[i])[1]],contrast_pairs['HB'][list(pairs[i])[0]]]),
            }
            loss_cv += 0.2/len(pairs) * self.cv_criterion(temp_pairs)
                
        loss_total = loss_cls + 0.01 * loss_cv
        
        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/cv': loss_cv,
            'Loss/Total': loss_total
        }
        
        return loss_total, loss_dict
