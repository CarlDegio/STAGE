import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
import numpy as np
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.style_weight = 10
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, style_control=None, prefer_dict=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions['traj_action'] = actions['traj_action'][:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries+1]

            a_hat, is_pad_hat, (mu, logvar), style_value = self.model(qpos, image, env_state, actions, is_pad)
            if np.random.rand() < 0.01:
                print(style_value[:30])
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            steer_throttle_l1 = F.l1_loss(actions['steer_throttle'], a_hat['steer_throttle'], reduction='none')*torch.tensor([1.0,1.0],device=actions['steer_throttle'].device)
            traj_l1 = F.l1_loss(actions['traj_action'], a_hat['traj_action'], reduction='none')*0.5
            all_l1 = torch.cat([steer_throttle_l1, traj_l1], dim=1)
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['style'] = style_preference(style_value, prefer_dict, actions)
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight + loss_dict['style'] * self.style_weight
            return loss_dict, style_value
        else: # inference time
            style_control = torch.tensor(style_control, device=image.device, dtype=torch.float32)
            style_control = style_control.unsqueeze(0).unsqueeze(0)
            a_hat, _, (_, _), style_value = self.model(qpos, image, env_state, style_control=style_control) # no action, sample from prior
            return a_hat, style_value

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

def style_preference(style_value, prefer_dict, actions):
    prefer_score = []
    for i in range(style_value.size(0)):
        speed_score = prefer_dict['ego_vel_kmh'][i].mean() / 50
        lane_diff_score = torch.abs(prefer_dict['ego_lane_diff_angle'][i]).mean() / (90 / 180 * np.pi)
        throttle_score = prefer_dict['ego_control'][i].mean(axis=0)[1] / 1.0
        if prefer_dict['has_nearest'][i] == True:
            distance_score = 10 / prefer_dict['nearest_distance'][i].mean()
        else:
            distance_score = 0
        score = speed_score + throttle_score + distance_score + lane_diff_score
        prefer_score.append(score)

    loss = []
    for i in range(0, style_value.size(0) - 1, 2):
        if prefer_score[i] > prefer_score[i + 1]:
            loss.append(-F.logsigmoid(style_value[i] - style_value[i + 1]))
        elif prefer_score[i + 1] > prefer_score[i]:
            loss.append(-F.logsigmoid(style_value[i + 1] - style_value[i]))
    loss = torch.stack(loss).mean()
    return loss