import argparse
import os
import time
from hashlib import sha1

import numpy as np
import torch as th
from runx.logx import logx
from torch.distributions import Categorical

from components import REGISTRY as mem_REGISTRY
from config import Config
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from schedgym.mySubproc_vec_env import SubprocVecEnv
from schedgym.sched_env import SchedEnv
from utils.rl_utils import linear_decay, time_format


def make_env(N, cpu, mem, allow_release, double_thr=1e10):
    def _init():
        env = SchedEnv(N, cpu, mem, DATA_PATH,
                       render_path=None,
                       rew_fn=REW_FN,
                       allow_release=allow_release,
                       ouble_thr=double_thr,
                       topk=args.topk)
        # env.seed(seed + rank)
        return env

    # set_global_seeds(seed)
    return _init


def run(envs, step_list, mac, mem, learner, eps, args, x, flag):
    episode_reward = {}
    episode_action = {}

    total_reward = np.array([0. for j in range(args.num_process)])
    total_length = np.array([0. for j in range(args.num_process)])
    step = 0
    stop_indices = np.array([0 for j in range(args.num_process)])

    # 初始化后面需要的lists
    avail = []
    feat = []
    obs = []
    state = []
    tmp_state = [[] for _ in range(args.num_process)]
    tmp_action = [[] for _ in range(args.num_process)]
    tmp_return = [[] for _ in range(args.num_process)]
    remains = [[] for _ in range(args.num_process)]

    prev_alives = envs.get_alives()
    while True:
        step += 1
        envs.update_alives()
        curr_alives = envs.get_alives().copy()
        if all(~curr_alives):
            if args.eps2 > 0:
                for i in range(args.num_process):
                    for j in range(len(tmp_state[i])):
                        key = tmp_state[i][j]
                        episode_reward[key] = tmp_return[i][-1] - tmp_return[i][j]
                        episode_action[key] = tmp_action[i][j]
            left = np.sum(remains, axis=2).sum(1).sum(0)
            print(total_length)
            return total_length.mean(), total_reward.mean(), \
                (2 * args.cpu * args.N * args.num_process - left[0]) / (2 * args.cpu * args.N * args.num_process), \
                (2 * args.mem * args.N * args.num_process - left[1]) / (2 * args.mem * args.N * args.num_process)

        if step == 1:
            avail = envs.get_attr('avail')
            feat = envs.get_attr('req')
            obs = envs.get_attr('obs')

        if curr_alives.tolist() != prev_alives.tolist():
            for i in range(len(curr_alives)):
                if prev_alives[i]:
                    indices.append(curr_alives[i])
            avail = avail[indices]
            feat = feat[indices]
            obs = obs[indices]

        state = {'obs': obs, 'feat': feat, 'avail': avail}
        action = mac.select_actions(state, flag=flag, eps=eps)

        if args.eps2 > 0:
            for j in range(action.shape[0]):
                if action[j] == -1:
                    key = sha1(obs[j]).hexdigest() + sha1(feat[j]).hexdigest()
                    if key in episode_action.keys() and avail[j][episode_action[key]] == 1:
                        action[j] = episode_action[key]
                    else:
                        action[j] = Categorical(th.from_numpy(np.float32(avail[j]))).sample()

        action_after, next_obs, reward, done, info = envs.step(action)

        # --------------------------------------------
        # indexes = []
        # for i in range(len(curr_alives)):
        #     if curr_alives[i]:
        #         indexes.append(i)
        # for i in range(len(indexes)):
        #     remains[indexes[i]] = next_obs[i]

        indices = np.where(curr_alives)[0]
        remains[indices] = next_obs[curr_alives]
        # ---------------------------------------------

        stop_indices[curr_alives] += 1

        if args.eps2 > 0:
            k = 0
            for j in range(args.num_process):
                if curr_alives[j]:
                    tmp_state[j].append(sha1(obs[k]).hexdigest() + sha1(feat[k]).hexdigest())
                    tmp_action[j].append(action[k])
                    if len(tmp_return[j]) > 0:
                        tmp_return[j].append(reward[k] + tmp_return[j][-1])
                    else:
                        tmp_return[j].append(reward[k])
                    k += 1

        next_avail = info['avail']
        next_feat = info['feat']

        total_reward[curr_alives] += reward
        total_length[curr_alives] += 1

        buf = {'obs': obs, 'feat': feat, 'avail': avail, 'action': action,
               'reward': reward, 'next_obs': next_obs, 'next_feat': next_feat,
               'next_avail': next_avail, 'done': done}
        mem.push(buf)

        avail = next_avail
        feat = next_feat
        obs = info['obs']

        prev_alives = curr_alives


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sched More Servers')
    parser.add_argument('--env', type=str, default='recovering')
    parser.add_argument('--alg', type=str, default='dqn_ep_modify')
    parser.add_argument('--gamma', type=float, default=0.85)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--entropy', type=float, default=0.001)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--rew-fn', type=str, default='iden')
    parser.add_argument('--memory', type=str, default='replay')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--num-process', type=int, default=1)
    parser.add_argument('--eps2', type=float, default=0.6)
    parser.add_argument('--capacity', type=int, default=1000000)
    parser.add_argument('--train-n', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--mac', type=str, default='vectormac')

    conf = parser.parse_args()
    args = Config(conf.env, conf.alg)

    DATA_PATH = 'vmagent/data/Huawei-East-1.csv'

    args.batch_size = conf.batch_size
    args.train_n = conf.train_n
    args.capacity = conf.capacity
    args.gamma = conf.gamma
    args.lr = conf.lr
    args.tau = conf.tau
    args.entropy_coef = conf.entropy
    args.memory = conf.memory
    args.topk = conf.topk
    args.rew_fn = conf.rew_fn
    args.num_process = conf.num_process
    args.eps2 = conf.eps2
    args.mac = conf.mac

    MAX_EPOCH = conf.epoch

    BATCH_SIZE = args.batch_size
    REW_FN = conf.rew_fn

    logpath = f'mylogs/server{args.N}/{conf.env}/{conf.alg}_{BATCH_SIZE}x{args.train_n}_{args.capacity}/{args.gamma}_{args.lr}/{args.entropy_coef}_{args.tau}/{args.eps2}'

    logx.initialize(logdir=logpath, coolname=True, tensorboard=True, hparams=vars(args))

    double_thr = args.double_thr or 1000

    envs = SubprocVecEnv([
        make_env(
            args.N, args.cpu, args.mem,
            allow_release=args.allow_release,
            double_thr=double_thr) for _ in range(args.num_process)])

    mac = mac_REGISTRY[args.mac](args)
    print(f'Sampling with {args.mac} for {MAX_EPOCH} epochs; Learn with {args.learner}')
    learner = le_REGISTRY[args.learner](mac, args)
    learner.cuda()
    mem = mem_REGISTRY[args.memory](args)
    args.capacity = 100
    mem_numa = mem_REGISTRY[args.memory](args)
    t_start = time.time()

    my_list = [23500]
    my_list = np.array(my_list)

    for x in range(MAX_EPOCH):
        eps = linear_decay(
            x,
            [0, int(MAX_EPOCH * 0.2), int(MAX_EPOCH * 0.75), MAX_EPOCH],
            [0.8, 0.3, 0.01, 0.001]
        )
        args.eps = eps
        args.entropy_coef = linear_decay(
            x,
            [0, int(MAX_EPOCH * 0.25), int(MAX_EPOCH * 0.75),
             MAX_EPOCH], [conf.entropy, conf.entropy / 5, conf.entropy / 40, 0.001]
        )

        envs.reset(my_list)

        train_len, train_rew, cpu_rate, mem_rate = run(
            envs, my_list, mac, mem, learner, eps, args, x, flag=False)

        actor_loss, critic_loss, critic1_loss, critic2_loss, alpha_loss = [0 for i in range(5)]

        for i in range(args.train_n):
            if args.memory == 'priority_replay':
                idx, batch, ISWeights = mem.sample(BATCH_SIZE)
                metrics, td_error = learner.train(batch, i, True, ISWeights)
                mem.batch_update(idx, np.abs(td_error.detach().cpu().numpy()))
            if args.memory == 'replay':
                batch = mem.sample(BATCH_SIZE)
                metrics = learner.train(batch, i)

        # log training curves
        metrics['eps'] = eps
        metrics['entropy'] = args.entropy_coef
        metrics['tot_reward'] = train_rew.mean()
        metrics['tot_length'] = train_len.mean()
        logx.metric('train', metrics, x)

        if x % args.test_interval == 0:
            train_rews = []
            train_lens = []
            cpu_rates = []
            mem_rates = []

            for i in range(1):
                envs.reset(my_list)
                train_len, train_rew, cpu_rate, mem_rate = run(
                    envs, my_list, mac, mem, learner, 0, args, x, flag=True)
                train_rews.append(train_rew)
                train_lens.append(train_len)
                cpu_rates.append(cpu_rate)
                mem_rates.append(mem_rate)

            train_rews = np.array(train_rews)
            train_lens = np.array(train_lens)
            cpu_rates = np.array(cpu_rates)
            mem_rates = np.array(mem_rates)

            val_metric = {
                'train_len': train_lens.mean(),
                'train_rew': train_rews.mean(),
                'cpu_rates': cpu_rates.mean(),
                'mem_rates': mem_rates.mean(),
            }
            logx.metric('val', val_metric, x)

            if x >= MAX_EPOCH - 250:
                path = f'models/'

                if not os.path.exists(path):
                    os.makedirs(path)

                learner.save_models(path, x)

            t_end = time.time()
            print(f'Epoch {x}/{MAX_EPOCH}; lasted %d hour, %d min, %d sec ' % time_format(t_end - t_start))
