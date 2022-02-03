# import phyre
import torch
import numpy as np
from glob import glob
import torch.nn.functional as F
from config import _C as C
from utils import tprint, pprint
import os
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import utils

class PredEvaluator(object):
    def __init__(self, device, data_loader, model, output_dir, checkpoint=None):
        self.device = device
        self.output_dir = output_dir
        self.data_loader = data_loader
        self.model = model
            
        self.metadata = utils.update_metadata(self.data_loader.dataset.metadata, self.device)


    def test(self):
        self.model.eval()
                
        for batch_idx, data in enumerate(self.data_loader):
            # only support B=1 for now
            assert data[0].shape[0] == 1
            
            for i in range(len(data)):
                data[i] = data[i][0].to(self.device)
            pos_seq, _, _, particle_type, nonk_mask, tgt_pos_seq = data

            tprint(f'eval: {batch_idx}/{len(self.data_loader)} ')

            with torch.no_grad():
                num_rollouts = tgt_pos_seq.shape[1]

                outputs = self.model(pos_seq, particle_type, self.metadata, nonk_mask, tgt_pos_seq, num_rollouts=num_rollouts, phase='test')

                if outputs['pred_collaposed']:
                    print('Rollout collaposed!!!!!!!!!!!!!')
                    continue
                    
                labels = {
                    'poss': tgt_pos_seq
                }
                print('loss: ', self.loss(outputs, labels, nonk_mask))
                bounds = self.metadata['bounds'].cpu().numpy()

                tgt_pos_seq = tgt_pos_seq.cpu().numpy()[:, ::2]
                pred_pos_seq = outputs['pred_poss'].cpu().numpy()[:, ::2]
                num_rollouts = tgt_pos_seq.shape[1]

                # plot gifs and save
                color = np.zeros([particle_type.shape[0], 3])
                color[nonk_mask.cpu().numpy().astype(np.bool)] = [0.122, 0.467, 0.706]
                fig = plt.figure(figsize=(12, 6))

                points1 = tgt_pos_seq[:, 0]
                ax1 = fig.add_subplot(121)

                ax1.set_xlim(bounds[0][0], bounds[0][1]); ax1.set_ylim(bounds[1][0], bounds[1][1]);
                ax1.set_title('Ground truth')
                ax1.get_xaxis().set_visible(False)
                ax1.get_yaxis().set_visible(False)
                pts1 = ax1.scatter(points1[:,0], points1[:,1], c=color, s=6.5)

                points2 = pred_pos_seq[:, 0]
                ax2 = fig.add_subplot(122)
                ax2.set_xlim(bounds[0][0], bounds[0][1]); ax2.set_ylim(bounds[1][0], bounds[1][1]);
                ax2.set_title('Prediction')
                ax2.get_xaxis().set_visible(False)
                ax2.get_yaxis().set_visible(False)
                pts2 = ax2.scatter(points2[:,0], points2[:,1], c=color, s=6.5)

                ani=animation.FuncAnimation(fig, update_points, frames=num_rollouts, fargs=(pts1, pts2, tgt_pos_seq, pred_pos_seq))
                os.makedirs(self.output_dir, exist_ok=True)
                os.path.join(self.output_dir, str(batch_idx)+'.gif')
                ani.save(os.path.join(self.output_dir, str(batch_idx)+'.gif'), fps=50)

    def loss(self, outputs, labels, weighting):
        loss = ((outputs['pred_poss'] - labels['poss']) * torch.unsqueeze(torch.unsqueeze(weighting, -1), 1)) ** 2
        loss = loss.mean(1).sum() / torch.sum(weighting)
        return loss.item()

def update_points(t, pts1, pts2, gt_pos, pred_pos):
    points1 = gt_pos[:, t]
    pts1.set_offsets(points1)

    points2 = pred_pos[:, t]
    pts2.set_offsets(points2)
