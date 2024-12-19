import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
import numpy as np
from gpt_process import multithread_api_call, generate_english_prompt
e = IPython.embed


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.style_weight = 10.0
        try:
            self.persistent_dict = np.load(
                "result_dict.npy", allow_pickle=True).item()
        except:
            self.persistent_dict = {}
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, style_control=None, prefer_dict=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions['traj_action'] = actions['traj_action'][:,
                                                            :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries+1]

            a_hat, is_pad_hat, (mu, logvar), style_value = self.model(
                qpos, image, env_state, actions, is_pad)
            if np.random.rand() < 0.01:
                print(style_value[:30])
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            steer_throttle_l1 = F.l1_loss(actions['steer_throttle'], a_hat['steer_throttle'],
                                          reduction='none')*torch.tensor([1.0, 1.0], device=actions['steer_throttle'].device)
            traj_l1 = F.l1_loss(
                actions['traj_action'], a_hat['traj_action'], reduction='none')*0.5
            all_l1 = torch.cat([steer_throttle_l1, traj_l1], dim=1)
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            # loss_dict['style'] = prompt_style_preference(
            #     style_value, prefer_dict, self.persistent_dict)
            loss_dict['style'] = rule_style_preference(style_value=style_value, prefer_dict=prefer_dict)
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * \
                self.kl_weight + loss_dict['style'] * self.style_weight
            return loss_dict, style_value
        else:  # inference time
            style_control = torch.tensor(
                style_control, device=image.device, dtype=torch.float32)
            style_control = style_control.unsqueeze(0).unsqueeze(0)
            a_hat, _, (_, _), style_value = self.model(
                qpos, image, env_state, style_control=style_control)  # no action, sample from prior
            return a_hat, style_value

    def configure_optimizers(self):
        return self.optimizer

    def save_preference_dict(self):
        print(f'Saving persistent_dict')
        np.save("result_dict.npy", self.persistent_dict)


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else:  # inference time
            # no action, sample from prior
            a_hat = self.model(qpos, image, env_state)
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


def rule_style_preference(style_value, prefer_dict):
    prefer_score = []
    for i in range(style_value.size(0)):
        speed_score = prefer_dict['ego_vel_kmh'][i].mean() / 30
        throttle_score = prefer_dict['ego_control'][i].mean(axis=0)[1] / 0.5
        if prefer_dict['has_nearest'][i] == True:
            if prefer_dict['nearest_distance'][i][0] < 20:
                distance_score = 1 / prefer_dict['nearest_distance'][i].mean()
                lane_diff_score = torch.abs(prefer_dict['ego_lane_diff_angle'][i]).mean() / (360 / 180 * np.pi)
                # lane_diff_score = 0
            else:
                distance_score = 0
                lane_diff_score = 0.0
        else:
            distance_score = 0
            lane_diff_score = 0.0
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


# def prompt_style_preference(style_value, prefer_dict, persistent_dict):
#     messages_with_id = []
#     for i in range(0, style_value.size(0)//4, 2):
#         prefer_dict1 = {'episode_id': prefer_dict['episode_id'][i], 'start_ts': prefer_dict['start_ts'][i], 'ego_pos': prefer_dict['ego_pos'][i], 'has_nearest': prefer_dict['has_nearest'][i], 'nearest_other_pos': prefer_dict['nearest_other_pos']
#                         [i], 'nearest_distance': prefer_dict['nearest_distance'][i], 'ego_vel_kmh': prefer_dict['ego_vel_kmh'][i], 'ego_lane_diff_angle': prefer_dict['ego_lane_diff_angle'][i], 'ego_control': prefer_dict['ego_control'][i]}
#         prefer_dict2 = {'episode_id': prefer_dict['episode_id'][i+1], 'start_ts': prefer_dict['start_ts'][i+1], 'ego_pos': prefer_dict['ego_pos'][i+1], 'has_nearest': prefer_dict['has_nearest'][i+1], 'nearest_other_pos': prefer_dict['nearest_other_pos']
#                         [i+1], 'nearest_distance': prefer_dict['nearest_distance'][i+1], 'ego_vel_kmh': prefer_dict['ego_vel_kmh'][i+1], 'ego_lane_diff_angle': prefer_dict['ego_lane_diff_angle'][i+1], 'ego_control': prefer_dict['ego_control'][i+1]}
#         for key in prefer_dict1:
#             prefer_dict1[key] = prefer_dict1[key].cpu().numpy()
#         for key in prefer_dict2:
#             prefer_dict2[key] = prefer_dict2[key].cpu().numpy()
#         messages_with_id.append((generate_english_prompt(prefer_dict1, prefer_dict2), (prefer_dict1['episode_id'].item(
#         ), prefer_dict1['start_ts'].item(), prefer_dict2['episode_id'].item(), prefer_dict2['start_ts'].item())))

#     results, completion_tokens, prompt_tokens = multithread_api_call(
#         messages_with_id, workers=100, persistent_dict=persistent_dict)

#     loss = []
#     for i in range(0, style_value.size(0)//4, 2):
#         if results[i//2] == '1':
#             loss.append(-F.logsigmoid(style_value[i] - style_value[i + 1]))
#         elif results[i//2] == '2':
#             loss.append(-F.logsigmoid(style_value[i + 1] - style_value[i]))
#     loss = torch.stack(loss).mean()
#     return loss
