import argparse
from calendar import c
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import random
from numpy.lib.function_base import append
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
import logging
from module.ac_meta import MetaAC
from net import Env, CHUNK_PERIOD
from qoe import QoEModel, cal_chunk_size, cal_max_in_bitrate, generate_bitrate
from replay_memory import ReplayMemory
from util import *
from copy import deepcopy
from time import strftime
from matplotlib import pyplot as plt
from tqdm import tqdm




RANDOM_SEED = 18
# S_INFO = 6
# S_LEN = 8
S_INFO = 4
S_LEN = 2

LEARNING_RATE_ACTOR = 0.001
LEARNING_RATE_CRITIC = 0.001
# TRAIN_SEQ_LEN = 100  # take as a train batch
VIDEO_BIT_RATE = [1, 5, 8, 16, 35]  # Mbps
# VIDEO_BIT_RATE = [1, 3, 5, 7, 9]  # Mbps
A_DIM = len(VIDEO_BIT_RATE) * 2# * WIDTH_TILE_NUMBER * HEIGHT_TILE_NUMBER
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 2.66  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
# RANDOM_SEED = 42
# GAMMA = 0.90
# ENTROPY_WEIGHT = 0.99
UPDATE_INTERVAL = 1000
RAND_RANGE = 1000
ENTROPY_EPS = 1e-6
SUMMARY_DIR = './Results/sim/a2c'
LOG_FILE = './Results/sim/a2c/log'
# TEST_PATH = './models/A3C/BC/360_a3c_240000.model'
BANDWIDTH_TRACE_PATH = "../belgium/report_tram_0008.log"
CONFIG = ""
parser = argparse.ArgumentParser(description='a2c_pytorch')
parser.add_argument('--test_basic',  type=bool, default=True, help='test basic')
parser.add_argument('--test_lr',  type=bool, default=True, help='test lr')
parser.add_argument('--test_ac',  type=bool, default=False, help='test ac')
parser.add_argument('--model', type=str, default="ac_basic", help='model_name')
parser.add_argument('--ftrain', type=bool, default=False, help='ftrain')
parser.add_argument('--load_name', type=str, default="ac_meta_without_1_1_1_1_1_4", help='load_name')
parser.add_argument('--ftrain_set', type=str, default="1_1_4", help='ftrain_set')
args = parser.parse_args()
EPOCH = 100
UPDATE_EPOCH = 1
MINI_BATCH_SIZE = 64

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
QOE_SET = [(1, 0.25, 0.25), (1, 1, 4), (1, 4, 1)]
# QOE_SET = [(1, 1, 1), (1, 4, 1), (1, 0.25, 0.25)]
# QOE_SET = [(1, 1, 1), (1, 1, 4), (1, 0.25, 0.25)]
# QOE_SET = [(1, 1, 1), (1, 1, 4), (1, 4, 1)]
# QOE_SET = [(1, 1, 1), (1, 1, 4), (1, 4, 1), (1, 0.25, 0.25)]
if args.ftrain or args.test_basic or args.test_ac or args.test_lr: 
    QOE_SET = [[float(item) for item in args.ftrain_set.split("_")]]

class Agent:
    def __init__(self, name) -> None:
        self.name = name
        self.states = []
        self.actions = []
        self.rewards_comparison = []
        self.rewards = []
        self.values = []
        self.returns = []
        self.advantages = []
        self.log_probs = []
        self.qoe_model = QoEModel(1, 1, 1)
        self.state = np.zeros((S_INFO, S_LEN))
        self.last_bitrate_level = DEFAULT_QUALITY
        self.curr_time = 0.0
        self.last_update_time = 0.0
        self.viewport_pred = [0, 0] 
        self.last_time = torch.tensor(0.0)
        self.last_reward = torch.tensor(0.0)
        self.last_action = torch.tensor([0, 0])
        self.last_times = []
        self.last_rewards = []
        self.last_actions = []
    
    def init_state(self):
        self.state = np.zeros((S_INFO, S_LEN))

    def next_steps(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.returns = []
        self.advantages = []
        self.log_probs = []
        self.last_times = []
        self.last_rewards = []
        self.last_actions = []

    def clip(self):
        self.states = self.states[1:]
        self.actions = self.actions[1:]
        self.rewards = self.rewards[1:]
        self.values = self.values[1:]
        self.returns = self.returns[1:]
        self.advantages = self.advantages[1:]
        self.log_probs = self.log_probs[1:]
        self.last_times = self.last_times[1:]
        self.last_rewards = self.last_rewards[1:]
        self.last_actions = self.last_actions[1:]

VIEWPORT_INI = [0, 0]
BUFFER_TARGET = 2.5

#logger
global print
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
try:
    os.makedirs("logger", exist_ok=True)
except TypeError:
    raise Exception("Direction not create!")
if args.ftrain == True:
    logger.addHandler(logging.FileHandler(strftime('logger/{}_%m-%d-%Y-%H:%M:%S.log'.format(args.model + "_ftrain_" + args.ftrain_set)), 'a'))
elif args.test_basic == True or args.test_ac == True or args.test_lr == True:
    logger.addHandler(logging.FileHandler(strftime('logger/{}_%m-%d-%Y-%H:%M:%S.log'.format(args.model + "_test_" + args.ftrain_set)), 'a'))
else:
    logger.addHandler(logging.FileHandler(strftime('logger/{}_%m-%d-%Y-%H:%M:%S.log'.format(args.model)), 'a'))
    
curr_strftime = strftime('{}_%m-%d-%Y-%H:%M:%S'.format("abr_meta"))
print = logger.info

def train_a2c():
    a_losses = []
    v_losses = []
    entropies = []
    rewards = []
    a_losses_valid = [[], [], [], []]
    v_losses_valid = [[], [], [], []]
    entropies_valid = [[], [], [], []]
    rewards_valid = [[], [], [], []]
    rewards_test = [[], [], [], []]
    test_topic_name = "topic7"
    v_lstms_train, v_hiddens_train, processed_v_data_train, user_cluster_train = create_processed_viewport_data_and_models(VIEWPORT_TRAIN_PATH)
    v_lstms_valid, v_hiddens_valid, processed_v_data_valid, user_cluster_valid = create_processed_viewport_data_and_models(VIEWPORT_VALID_PATH)
    v_lstms_test, v_hiddens_test, processed_v_data_test, user_cluster_test = create_processed_viewport_data_and_models(VIEWPORT_TEST_PATH)
    bw_lstm, bw_hidden, processed_bw_data = create_processed_bw_data_and_models()
    
    if args.test_basic:
        test_r, avg_r = test_basic((v_lstms_test, v_hiddens_test, processed_v_data_test, user_cluster_test), (bw_lstm, bw_hidden, processed_bw_data), QOE_SET[0])
        print("test_r:{}".format(test_r))
        print("avg_r:{}".format(avg_r))
        return

    if args.test_lr:
        test_r, avg_r = test_lr((v_lstms_test, v_hiddens_test, processed_v_data_test, user_cluster_test), (bw_lstm, bw_hidden, processed_bw_data), QOE_SET[0])
        print("test_r:{}".format(test_r))
        print("avg_r:{}".format(avg_r))
        return
    # entropy_weight = ENTROPY_WEIGHT
    # value_loss_coef = 0.5
    # torch.manual_seed(RANDOM_SEED)

    model = MetaAC(A_DIM).type(dtype)
    if args.ftrain or args.test_ac:
        load_path = "../models/" + args.load_name +".pt"
        model.load_state_dict(torch.load(load_path)["model"])
    model = model.to(DEVICE)

    if args.test_ac:
        test_r, avg_r = test_a2c(model, (v_lstms_test, v_hiddens_test, processed_v_data_test, user_cluster_test), (bw_lstm, bw_hidden, processed_bw_data), QOE_SET[0])
        print("test_r:{}".format(test_r))
        print("avg_r:{}".format(avg_r))
        return
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_ACTOR)

    # max_grad_norm = MAX_GRAD_NORM 

    state = np.zeros((S_INFO,S_LEN))
    state = torch.from_numpy(state)
    last_bit_rate = DEFAULT_QUALITY
    # bit_rate = DEFAULT_QUALITY
    # action_vec = np.zeros(A_DIM)
    # action_vec[bit_rate] = 1

    done = True
    epoch = 0
    time_stamp = 0

    agent_num = 37
    episode_steps = 10
    ROUND = 6
    # update_num = 1
    # batch_size = exploration_size * episode_steps #64
    gamma = 1
    gae_param = 0.95
    ent_coeff = 1
    # cl_coeff = 0.2
    memory = ReplayMemory(agent_num * episode_steps * ROUND)
    env = Env(BANDWIDTH_TRACE_PATH)
    last_group = -1
    hidden = model.init_hidden(1)

    for epoch in range(EPOCH):
        # train_topic_name = "topic" + str(random.randint(0, 6))
        train_topic_name = "topic" + "0"
        r = []
        tmp_a_losses = []
        tmp_v_losses = []
        tmp_entropies = []
        curr_group = int((epoch / EPOCH) * len(QOE_SET))
        if curr_group != last_group: hidden = model.init_hidden(1)
        last_group = curr_group
        for i in range(len(user_cluster_train) - 1):
            # if i != 0: continue
            # i = 0
            # round = 0
            ret_bw = [] 
            for agent_id in tqdm(user_cluster_train[i]):
                # hidden = model.init_hidden(1)
                hidden = (hidden[0].type(dtype).to(DEVICE), hidden[1].type(dtype).to(DEVICE))
                # if agent_id != user_cluster_train[i][0]: continue
                # agent_id = user_cluster[i][0]
                ## print("Group: {}, name: {}".format(i, agent_id))
                v_data = processed_v_data_train[i][train_topic_name][agent_id]
                agent = Agent("N")
                agent.name = agent_id
                # qoe_set = QOE_SET[random.randint(0, len(QOE_SET) - 1)]
                qoe_set = QOE_SET[curr_group]
                agent.qoe_model = QoEModel(qoe_set[0], qoe_set[1], qoe_set[2])
                # print("Group: {}, name: {}".format(0, user_cluster[i][0]))
                # v_data = processed_v_data[i][train_topic_name][user_cluster[i][0]]
                # agent = Agent(user_cluster[i][0])
                # for round in range(ROUND):
                for round in range(ROUND):
                    agent.next_steps()
                    for step in range(episode_steps):
                        curr_step = step + round * episode_steps
                        if type(agent.state) != torch.Tensor: agent.state = torch.from_numpy(agent.state).type(dtype).to(DEVICE)
                        if type(agent.last_reward) != torch.Tensor: agent.last_reward = torch.tensor(agent.last_reward).type(dtype).to(DEVICE)
                        if type(agent.last_action) != torch.Tensor: agent.last_action = torch.tensor(agent.last_action).type(dtype).to(DEVICE)
                        if type(agent.last_time) != torch.Tensor: agent.last_time = torch.tensor(agent.last_time).type(dtype).to(DEVICE)
                        agent.last_rewards.append(agent.last_reward.unsqueeze(0))
                        agent.last_actions.append(agent.last_action.unsqueeze(0))
                        agent.last_times.append(agent.last_time.unsqueeze(0))
                        ## print("last_reward: {}, last_action: {}, last_time:{}".format(agent.last_reward, agent.last_action, agent.last_time))

                        input_s = agent.state.unsqueeze(0).type(dtype).to(DEVICE)
                        input_r = agent.last_reward.view(1, 1, -1).type(dtype).to(DEVICE)
                        input_a = agent.last_action.view(1, 1, -1).type(dtype).to(DEVICE)
                        input_t = agent.last_time.view(1, 1, -1).type(dtype).to(DEVICE)
                        hidden = (hidden[0].detach(), hidden[1].detach())
                        input = (input_s, input_r, input_a, input_t, hidden)
                        prob, v, hidden = model(input)
                        prob = prob.view(-1, int(A_DIM / len(VIDEO_BIT_RATE)), len(VIDEO_BIT_RATE))
                        prob_p = prob[-1, :, :]
                        action = prob_p.multinomial(num_samples=1)
                        prob_value = torch.gather(prob, dim=2, index=action.unsqueeze(0).type(dlongtype))
                        log_prob_value = torch.log(prob_value.squeeze(2).squeeze(0))
                        agent.log_probs.append(log_prob_value.unsqueeze(0))
                        ## print("input: {}, prob: {}, action: {}, prob_value: {}, log_prob_value: {}".format(input, prob, action, prob_value, log_prob_value))
                        # seed_ = np.random.uniform(0,1)
                        # if np.random.uniform(0,1) <= exploration_threhold:
                        #     action = random.randint(0, 5)
                        #     action = torch.tensor([[action]]).type(dlongtype)
                        # else:
                        #     action = prob.multinomial(num_samples=1)
                        agent.values.append(v)

                        bitrate_levels = action.squeeze().cpu().numpy()

                        agent.actions.append(action)
                        
                        agent.last_action = torch.tensor(bitrate_levels)
                        
                        
                        agent.states.append(agent.state.unsqueeze(0))
                        if curr_step != 0:
                            bit_rates = np.array([VIDEO_BIT_RATE[bitrate_level] for bitrate_level in bitrate_levels])
                        else:
                            bit_rates = np.array([1, 1])
                        # bit_rate = np.mean(bit_rates)
                        # tile_bit_rates = bit_rates.reshape(WIDTH_TILE_NUMBER, HEIGHT_TILE_NUMBER)
                        chunk_size = cal_chunk_size(VIDEO_CONFIG, bit_rates, CHUNK_PERIOD)
                        bit_rate = chunk_size / CHUNK_PERIOD
                        tile_bit_rates = generate_bitrate(VIDEO_CONFIG, bit_rates, agent.state[2, :2].cpu().numpy())
                        agent.last_update_time = agent.curr_time
                        
                        agent.last_time = torch.tensor(agent.curr_time)                     
                        agent.curr_time = env.push_video_chunk(chunk_size)
                        env.update(agent.curr_time)
                        buffer_remaining = env.buffer_remaining
                        rebuffer_time = env.rebuffer_time
                        viewport_gt = [int(WIDTH_TILE_NUMBER * v_data[curr_step][1].numpy()[0][0]), int(HEIGHT_TILE_NUMBER * v_data[curr_step][1].numpy()[0][1])]
                        _, reward, wq = agent.qoe_model.cal_qoe(tile_bit_rates, VIDEO_CONFIG, viewport_gt, rebuffer_time)
                        agent.rewards.append(reward)
                        
                        agent.last_reward = torch.tensor(reward)
                        if curr_step!=0: r.append(reward)
                        # print("current time: {}".format(agent.curr_time))
                        # print("bitrate: {}, avg bitrate: {}, window quality: {}, true viewport: {}, buffer_remaining: {}, rebuffer_time: {}, reward: {}".format(tile_bit_rates, bit_rate, wq, viewport_gt, buffer_remaining, rebuffer_time, reward))

                        if int(agent.curr_time) != 0 and int(agent.curr_time) != int(agent.last_update_time):
                            ret_bw = []
                            t = int(agent.curr_time) % (len(env.bandwidth_trace) - 2)
                            bw_hidden = predict_next_few_seconds_bw(bw_lstm, processed_bw_data[t:t + 2], bw_hidden, ret_bw)  
                        elif int(agent.curr_time) == 0:
                            ret_bw = [[VIDEO_BIT_RATE[DEFAULT_QUALITY], "Unknown"], [VIDEO_BIT_RATE[DEFAULT_QUALITY], "Unknown"]]
                        # print("predict bandwidth for next 2s: {} (gt:{}), {} (gt:{})".format(ret_bw[0][0], ret_bw[0][1], ret_bw[1][0], ret_bw[1][1]))

                        ret_viewport = []
                        if curr_step != 0:
                            # print(v_data[curr_step:curr_step + 1])
                            v_hiddens_train[i] = predict_next_few_seconds_viewport(v_lstms_train[i], v_data[curr_step:curr_step + 1], i, v_hiddens_train[i], ret_viewport)
                            ret_viewport = [ret_viewport[0][0], ret_viewport[0][1]]
                        else:
                            ret_viewport = [(int(0.5 * WIDTH_TILE_NUMBER), int(0.5 * HEIGHT_TILE_NUMBER)), (None, None)]
                        # print("predict viewport for next 1s: {} (gt:{})".format(ret_viewport[0], ret_viewport[1]))
                        agent.init_state()
                        # agent.state[0, -1] = agent.curr_time # agent.curr_time - agent.last_update_time # int() / 100.0
                        agent.state[0, -1] = agent.curr_time - agent.last_update_time
                        agent.state[1, -1] = buffer_remaining
                        agent.state[2, :2] = np.array(ret_viewport[0])
                        agent.state[3, :2] = np.array([ret_bw[0][0], ret_bw[1][0]])
                        ## print("current state: {}".format(agent.state))
                        agent.state = torch.from_numpy(agent.state)
                        

                    if round == 0: agent.clip()
                    # one last step
                    R = torch.zeros(1, 1)
                    # if end_of_video == False:
                    input_s = agent.state.unsqueeze(0).type(dtype).to(DEVICE)
                    input_r = agent.last_reward.view(1, 1, -1).type(dtype).to(DEVICE)
                    input_a = agent.last_action.view(1, 1, -1).type(dtype).to(DEVICE)
                    input_t = agent.last_time.view(1, 1, -1).type(dtype).to(DEVICE)
                    hidden = (hidden[0].detach(), hidden[1].detach())
                    input = (input_s, input_r, input_a, input_t, hidden)
                    _, v, _ = model(input)
                    # v = v.detach().cpu()
                    # R = v.data
                    R = v.detach()
                    #================================结束一个ep========================================
                    # compute returns and GAE(lambda) advantages:
                    # values.append(Variable(R))
                    # R = Variable(R)
                    # A = Variable(torch.zeros(1, 1))
                    # for i in reversed(range(len(rewards))):
                    #     td = rewards[i] + gamma * values[i + 1].data[0, 0] - values[i].data[0, 0]
                    #     A = float(td) + gamma * gae_param * A
                    #     advantages.insert(0, A)
                    #     # R = A + values[i]
                    #     R = gamma * R + rewards[i]
                    #     # R = Variable(R)
                    #     returns.insert(0, R)

                    ## compute returns and advantages with Monte Carlo sampling
                    agent.values.append(R)
                    ## print("v: {}".format(agent.values))
                    ## print("r: {}".format(agent.rewards))
                    # td = Variable(torch.zeros(1, 1))
                    for j in reversed(range(len(agent.rewards))):
                        R = gamma * R + agent.rewards[j]
                        agent.returns.insert(0, R)
                        td = R - agent.values[j]
                        ## print("R: {}, v: {}, td: {}".format(R, agent.values[j], td))
                        agent.advantages.insert(0, td) 
                    # store usefull info:
                    # memory.push([states[1:], actions[1:], rewards_comparison[1:], returns[1:], advantages[1:]])
                    # memory.push([states[1:], actions[1:], returns[1:], advantages[1:]])
                    if torch.eq(agent.states[0][0], torch.from_numpy(np.zeros((S_INFO,S_LEN)))).sum() == S_INFO * S_LEN: ## judge if states[0] equals to torch.from_numpy(np.zeros((S_INFO,S_LEN)))
                        memory.push([agent.states[1:], agent.actions[1:], agent.returns[1:], agent.advantages[1:], agent.log_probs[1:], agent.last_rewards[1:], agent.last_actions[1:], agent.last_times[1:]])
                    else:  
                        memory.push([agent.states, agent.actions, agent.returns, agent.advantages, agent.log_probs, agent.last_rewards, agent.last_actions, agent.last_times])
                    # print(memory.memory)
                    # memory.push([agent.states[1:], agent.actions[1:], agent.returns[1:], agent.advantages[1:]])
                    # print(len(memory.memory))
                    # print("====================================================")
                env.reset()  
                    # round += 1
                    # restore the initial state
            bw_hidden = bw_lstm.init_hidden(BATCH_SIZE)
            v_hiddens_train[i] = v_lstms_train[i].init_hidden(BATCH_SIZE)

           
        # print("####################################################")
        # policy grad updates:
        model.zero_grad()

        # large_batch
        # batch_size = memory.return_size()
        # batch_states, batch_actions, batch_returns, batch_advantages = memory.pop(batch_size)
        # batch_actions = batch_actions.view(batch_size, -1)
        # probs_pre = model_actor(batch_states.type(dtype))
        # values_pre = model_critic(batch_states.type(dtype))
        # prob_value = torch.gather(probs_pre.view(batch_size, -1, len(VIDEO_BIT_RATE)), dim=2, index=batch_actions.unsqueeze(2).type(dlongtype))
        # log_prob_value = torch.log(prob_value.squeeze(2))
        # policy_loss = -torch.mean(log_prob_value * batch_advantages.type(dtype))
        # loss_ent = ent_coeff * torch.mean(probs_pre * torch.log(probs_pre + 1e-5))
        # vf_loss = (values_pre - batch_returns.type(dtype)) ** 2 # V_\theta - Q'
        # critic_loss = 0.5 * torch.mean(vf_loss)
        # # update
        # actor_total_loss = policy_loss + loss_ent
        # optimizer_actor.zero_grad()
        # optimizer_critic.zero_grad()
        # actor_total_loss.backward()
        # torch.nn.utils.clip_grad_norm(model_actor.parameters(), 0.5)
        # optimizer_actor.step()
        # critic_loss.backward()
        # torch.nn.utils.clip_grad_norm(model_critic.parameters(), 0.5)
        # optimizer_critic.step()
        # ## test and save the model
        # epoch += 1
        # memory.clear()
        # a_losses.append(float(policy_loss.detach().cpu().numpy()))
        # v_losses.append(float(critic_loss.detach().cpu().numpy()))
        # entropies.append(A_DIM * float(loss_ent.detach().cpu().numpy()))
        # rewards.append(np.array(r).mean())


        # mini_batch
        batch_size = memory.return_size()
        batch_states, batch_actions, batch_returns, batch_advantages, batch_log_probs, batch_last_rewards, batch_last_actions, batch_last_times = memory.pop(batch_size)
        # batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-5)
        mini_batch_num = int(batch_size / batch_size)
        b_hidden = model.init_hidden(batch_size)
        b_hidden = (b_hidden[0].type(dtype).to(DEVICE), b_hidden[1].type(dtype).to(DEVICE))



        for update_step in tqdm(range(mini_batch_num)):
            tmp_batch_states = batch_states[update_step * batch_size: (update_step + 1) * batch_size].type(dtype).to(DEVICE)
            tmp_batch_advantages = batch_advantages[update_step * batch_size: (update_step + 1) * batch_size].type(dtype).to(DEVICE)
            # tmp_batch_advantages = tmp_batch_advantages.type(dtype)
            tmp_batch_log_probs = batch_log_probs[update_step * batch_size: (update_step + 1) * batch_size].type(dtype).to(DEVICE)
            # tmp_batch_log_probs = tmp_batch_log_probs.type(dtype)
            tmp_batch_last_rewards = batch_last_rewards[update_step * batch_size: (update_step + 1) * batch_size].type(dtype).to(DEVICE).unsqueeze(1)
            tmp_batch_last_actions = batch_last_actions[update_step * batch_size: (update_step + 1) * batch_size].type(dtype).to(DEVICE).unsqueeze(1)
            tmp_batch_last_times = batch_last_times[update_step * batch_size: (update_step + 1) * batch_size].type(dtype).to(DEVICE).unsqueeze(1)
            b_hidden = (b_hidden[0].detach(), b_hidden[1].detach())
            input = (tmp_batch_states, tmp_batch_last_rewards, tmp_batch_last_actions, tmp_batch_last_times, b_hidden)
            probs_pre, _, b_hidden = model(input)
            # print("prob_value: {} {}".format(prob_value.shape, prob_value))
            # print("prob_value shape: {}".format(prob_value.squeeze(2).shape))
            ## print("tmp_batch_advantages: {}".format(tmp_batch_advantages))
            policy_loss = -torch.mean(tmp_batch_log_probs * tmp_batch_advantages.detach())
            # print("loss: {}".format((-log_prob_value * batch_advantages.type(dtype)).detach().cpu().numpy().tolist()))
            loss_ent = -ent_coeff * torch.mean(probs_pre * torch.log(probs_pre + 1e-5))

            # critic_loss
            # print("vf_loss: {}".format(vf_loss.detach().cpu().numpy().tolist()))
            # vf_loss = batch_advantages
            critic_loss = 0.5 * tmp_batch_advantages.pow(2).mean()

            # update
            actor_total_loss = policy_loss + 0.01 * loss_ent
            loss = actor_total_loss + critic_loss
            torch.autograd.set_detect_anomaly(True)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
            optimizer.step()
 
            tmp_a_losses.append(float(policy_loss.detach().cpu().numpy()))
            tmp_v_losses.append(float(critic_loss.detach().cpu().numpy()))
            tmp_entropies.append(A_DIM * float(loss_ent.detach().cpu().numpy()))
        memory.clear()
        


        ## test and save the model
        epoch += 1
        a_losses.append(np.mean(np.array(tmp_a_losses)))
        v_losses.append(np.mean(np.array(tmp_v_losses)))
        entropies.append(np.mean(np.array(tmp_entropies)))
        rewards.append(np.array(r).mean())
        print("Training...")
        logging.info('Epoch: ' + str(epoch) +
                        ' Avg_policy_loss: ' + str(a_losses[-1]) +
                        ' Avg_value_loss: ' + str(v_losses[-1]) +
                        ' Avg_entropy_loss: ' + str(entropies[-1]) +
                        ' Avg reward: ' + str(rewards[-1]))
        if epoch % UPDATE_EPOCH == 0:
            logging.info("Model saved in file")
            for tt in range(len(QOE_SET)):
                # ret = valid_a2c(model, (v_lstms_valid, v_hiddens_valid, processed_v_data_valid, user_cluster_valid), (bw_lstm, bw_hidden, processed_bw_data), (QOE_SET[tt][0], QOE_SET[tt][1], QOE_SET[tt][2]))
                # a_losses_valid[tt].append(ret[0])
                # v_losses_valid[tt].append(ret[1])
                # entropies_valid[tt].append(ret[2])
                # rewards_valid[tt].append(ret[3])
                _, test_r = test_a2c(model, (v_lstms_test, v_hiddens_test, processed_v_data_test, user_cluster_test), (bw_lstm, bw_hidden, processed_bw_data), (QOE_SET[tt][0], QOE_SET[tt][1], QOE_SET[tt][2]))
                rewards_test[tt].append(test_r)
                print("Avg Reward: {} for QoE set: {}, {}, {}".format(test_r, QOE_SET[tt][0], QOE_SET[tt][1], QOE_SET[tt][2]))
                if len(rewards_test[tt]) == 0 or test_r == max(rewards_test[tt]):
                    if args.ftrain==False: 
                        torch.save({'model': model.state_dict(), 'reward': test_r}, "../models/" + args.model + "_" + str(QOE_SET[tt][0]) + "_" + str(QOE_SET[tt][1]) + "_" + str(QOE_SET[tt][2]) + ".pt")
                    else: 
                        torch.save({'model': model.state_dict(), 'reward': test_r}, "../models/" + args.model + "_ftrain_" + str(QOE_SET[tt][0]) + "_" + str(QOE_SET[tt][1]) + "_" + str(QOE_SET[tt][2]) + ".pt")
            # entropy_weight = 0.95 * entropy_weight
            ent_coeff = 0.95 * ent_coeff
    return (a_losses, v_losses, entropies, rewards), (a_losses_valid, v_losses_valid, entropies_valid, rewards_valid), rewards_test

def valid_a2c(test_model, view_data, bw_data, qoe_set):
    print("Validing...")
    a_losses = []
    v_losses = []
    entropies = []
    rewards = []
    v_lstms, v_hiddens, processed_v_data, user_cluster = view_data
    bw_lstm, bw_hidden, processed_bw_data = bw_data
    model = MetaAC(A_DIM).type(dtype)
    model = model.to(DEVICE)
    model.load_state_dict(test_model.state_dict())
    model.eval()


    state = np.zeros((S_INFO,S_LEN))
    state = torch.from_numpy(state)
    last_bit_rate = DEFAULT_QUALITY
    done = True
    epoch = 0
    time_stamp = 0

    agent_num = 37
    episode_steps = 10
    ROUND = 6
    # update_num = 1
    # batch_size = exploration_size * episode_steps #64
    gamma = 1
    gae_param = 0.95
    ent_coeff = 1
    # cl_coeff = 0.2
    memory = ReplayMemory(agent_num * episode_steps * ROUND)
    env = Env(BANDWIDTH_TRACE_PATH)
    valid_topic_name = "topic7"
    r = []
    tmp_a_losses = []
    tmp_v_losses = []
    tmp_entropies = []
    hidden = model.init_hidden(1)
    hidden = (hidden[0].type(dtype).to(DEVICE), hidden[1].type(dtype).to(DEVICE))
    for i in range(len(user_cluster) - 1):
        # if i != 0: continue
        # i = 0
        # round = 0
        ret_bw = []
        # hidden = model.init_hidden(1)
        # hidden = (hidden[0].type(dtype).to(DEVICE), hidden[1].type(dtype).to(DEVICE))
        for agent_id in tqdm(user_cluster[i]):
            # if agent_id != user_cluster_train[i][0]: continue
            # agent_id = user_cluster[i][0]
            # print("Group: {}, name: {}".format(i, agent_id))
            v_data = processed_v_data[i][valid_topic_name][agent_id]
            agent = Agent("N")
            agent.name = agent_id
            agent.qoe_model = QoEModel(qoe_set[0], qoe_set[1], qoe_set[2])
            # print("Group: {}, name: {}".format(0, user_cluster[i][0]))
            # v_data = processed_v_data[i][train_topic_name][user_cluster[i][0]]
            # agent = Agent(user_cluster[i][0])
            # for round in range(ROUND):
            for round in range(ROUND):
                agent.next_steps()
                for step in range(episode_steps):
                    curr_step = step + round * episode_steps
                    if type(agent.state) != torch.Tensor: agent.state = torch.from_numpy(agent.state).type(dtype).to(DEVICE)
                    if type(agent.last_reward) != torch.Tensor: agent.last_reward = torch.tensor(agent.last_reward).type(dtype).to(DEVICE)
                    if type(agent.last_action) != torch.Tensor: agent.last_action = torch.tensor(agent.last_action).type(dtype).to(DEVICE)
                    if type(agent.last_time) != torch.Tensor: agent.last_time = torch.tensor(agent.last_time).type(dtype).to(DEVICE)
                    agent.last_rewards.append(agent.last_reward.unsqueeze(0))
                    agent.last_actions.append(agent.last_action.unsqueeze(0))
                    agent.last_times.append(agent.last_time.unsqueeze(0))
                    # print("last_reward: {}, last_action: {}, last_time:{}".format(agent.last_reward, agent.last_action, agent.last_time))

                    input_s = agent.state.unsqueeze(0).type(dtype).to(DEVICE)
                    input_r = agent.last_reward.view(1, 1, -1).type(dtype).to(DEVICE)
                    input_a = agent.last_action.view(1, 1, -1).type(dtype).to(DEVICE)
                    input_t = agent.last_time.view(1, 1, -1).type(dtype).to(DEVICE)
                    hidden = (hidden[0].detach(), hidden[1].detach())
                    input = (input_s, input_r, input_a, input_t, hidden)
                    prob, v, hidden = model(input)
                    prob = prob.view(-1, int(A_DIM / len(VIDEO_BIT_RATE)), len(VIDEO_BIT_RATE))
                    prob_p = prob[-1, :, :]
                    action = prob_p.multinomial(num_samples=1)
                    prob_value = torch.gather(prob, dim=2, index=action.unsqueeze(0).type(dlongtype))
                    log_prob_value = torch.log(prob_value.squeeze(2).squeeze(0))
                    agent.log_probs.append(log_prob_value.unsqueeze(0))
                    # print("input: {}, prob: {}, action: {}, prob_value: {}, log_prob_value: {}".format(input, prob, action, prob_value, log_prob_value))
                    # seed_ = np.random.uniform(0,1)
                    # if np.random.uniform(0,1) <= exploration_threhold:
                    #     action = random.randint(0, 5)
                    #     action = torch.tensor([[action]]).type(dlongtype)
                    # else:
                    #     action = prob.multinomial(num_samples=1)
                    agent.values.append(v)

                    bitrate_levels = action.squeeze().cpu().numpy()

                    agent.actions.append(action)
                    
                    agent.last_action = torch.tensor(bitrate_levels)
                    
                    
                    agent.states.append(agent.state.unsqueeze(0))
                    if curr_step != 0:
                        bit_rates = np.array([VIDEO_BIT_RATE[bitrate_level] for bitrate_level in bitrate_levels])
                    else:
                        bit_rates = np.array([1, 1])
                    # bit_rate = np.mean(bit_rates)
                    # tile_bit_rates = bit_rates.reshape(WIDTH_TILE_NUMBER, HEIGHT_TILE_NUMBER)
                    chunk_size = cal_chunk_size(VIDEO_CONFIG, bit_rates, CHUNK_PERIOD)
                    bit_rate = chunk_size / CHUNK_PERIOD
                    tile_bit_rates = generate_bitrate(VIDEO_CONFIG, bit_rates, agent.state[2, :2].cpu().numpy())
                    agent.last_update_time = agent.curr_time
                    
                    agent.last_time = torch.tensor(agent.curr_time)                     
                    agent.curr_time = env.push_video_chunk(chunk_size)
                    env.update(agent.curr_time)
                    buffer_remaining = env.buffer_remaining
                    rebuffer_time = env.rebuffer_time
                    viewport_gt = [int(WIDTH_TILE_NUMBER * v_data[curr_step][1].numpy()[0][0]), int(HEIGHT_TILE_NUMBER * v_data[curr_step][1].numpy()[0][1])]
                    _, reward, wq = agent.qoe_model.cal_qoe(tile_bit_rates, VIDEO_CONFIG, viewport_gt, rebuffer_time)
                    agent.rewards.append(reward)
                    
                    agent.last_reward = torch.tensor(reward)
                    if curr_step!=0: r.append(reward)
                    # print("current time: {}".format(agent.curr_time))
                    # print("bitrate: {}, avg bitrate: {}, window quality: {}, true viewport: {}, buffer_remaining: {}, rebuffer_time: {}, reward: {}".format(tile_bit_rates, bit_rate, wq, viewport_gt, buffer_remaining, rebuffer_time, reward))

                    if int(agent.curr_time) != 0 and int(agent.curr_time) != int(agent.last_update_time):
                        ret_bw = []
                        t = int(agent.curr_time) % (len(env.bandwidth_trace) - 2)
                        bw_hidden = predict_next_few_seconds_bw(bw_lstm, processed_bw_data[t:t + 2], bw_hidden, ret_bw)  
                    elif int(agent.curr_time) == 0:
                        ret_bw = [[VIDEO_BIT_RATE[DEFAULT_QUALITY], "Unknown"], [VIDEO_BIT_RATE[DEFAULT_QUALITY], "Unknown"]]
                    # print("predict bandwidth for next 2s: {} (gt:{}), {} (gt:{})".format(ret_bw[0][0], ret_bw[0][1], ret_bw[1][0], ret_bw[1][1]))

                    ret_viewport = []
                    if curr_step != 0:
                        # print(v_data[curr_step:curr_step + 1])
                        v_hiddens[i] = predict_next_few_seconds_viewport(v_lstms[i], v_data[curr_step:curr_step + 1], i, v_hiddens[i], ret_viewport)
                        ret_viewport = [ret_viewport[0][0], ret_viewport[0][1]]
                    else:
                        ret_viewport = [(int(0.5 * WIDTH_TILE_NUMBER), int(0.5 * HEIGHT_TILE_NUMBER)), (None, None)]
                    # print("predict viewport for next 1s: {} (gt:{})".format(ret_viewport[0], ret_viewport[1]))
                    agent.init_state()
                    # agent.state[0, -1] = agent.curr_time # agent.curr_time - agent.last_update_time # int() / 100.0
                    agent.state[0, -1] = agent.curr_time - agent.last_update_time
                    agent.state[1, -1] = buffer_remaining
                    agent.state[2, :2] = np.array(ret_viewport[0])
                    agent.state[3, :2] = np.array([ret_bw[0][0], ret_bw[1][0]])
                    # print("current state: {}".format(agent.state))
                    agent.state = torch.from_numpy(agent.state)
                    

                if round == 0: agent.clip()
                # one last step
                R = torch.zeros(1, 1)
                # if end_of_video == False:
                input_s = agent.state.unsqueeze(0).type(dtype).to(DEVICE)
                input_r = agent.last_reward.view(1, 1, -1).type(dtype).to(DEVICE)
                input_a = agent.last_action.view(1, 1, -1).type(dtype).to(DEVICE)
                input_t = agent.last_time.view(1, 1, -1).type(dtype).to(DEVICE)
                hidden = (hidden[0].detach(), hidden[1].detach())
                input = (input_s, input_r, input_a, input_t, hidden)
                _, v, _ = model(input)
                # v = v.detach().cpu()
                # R = v.data
                R = v.detach()
                #================================结束一个ep========================================
                # compute returns and GAE(lambda) advantages:
                # values.append(Variable(R))
                # R = Variable(R)
                # A = Variable(torch.zeros(1, 1))
                # for i in reversed(range(len(rewards))):
                #     td = rewards[i] + gamma * values[i + 1].data[0, 0] - values[i].data[0, 0]
                #     A = float(td) + gamma * gae_param * A
                #     advantages.insert(0, A)
                #     # R = A + values[i]
                #     R = gamma * R + rewards[i]
                #     # R = Variable(R)
                #     returns.insert(0, R)

                ## compute returns and advantages with Monte Carlo sampling
                agent.values.append(R)
                # print("v: {}".format(agent.values))
                # print("r: {}".format(agent.rewards))
                # td = Variable(torch.zeros(1, 1))
                for j in reversed(range(len(agent.rewards))):
                    R = gamma * R + agent.rewards[j]
                    agent.returns.insert(0, R)
                    td = R - agent.values[j]
                    # print("R: {}, v: {}, td: {}".format(R, agent.values[j], td))
                    agent.advantages.insert(0, td) 
                # store usefull info:
                # memory.push([states[1:], actions[1:], rewards_comparison[1:], returns[1:], advantages[1:]])
                # memory.push([states[1:], actions[1:], returns[1:], advantages[1:]])
                if torch.eq(agent.states[0][0], torch.from_numpy(np.zeros((S_INFO,S_LEN)))).sum() == S_INFO * S_LEN: ## judge if states[0] equals to torch.from_numpy(np.zeros((S_INFO,S_LEN)))
                    memory.push([agent.states[1:], agent.actions[1:], agent.returns[1:], agent.advantages[1:], agent.log_probs[1:], agent.last_rewards[1:], agent.last_actions[1:], agent.last_times[1:]])
                else:  
                    memory.push([agent.states, agent.actions, agent.returns, agent.advantages, agent.log_probs, agent.last_rewards, agent.last_actions, agent.last_times])
                # print(memory.memory)
                # memory.push([agent.states[1:], agent.actions[1:], agent.returns[1:], agent.advantages[1:]])
                # print(len(memory.memory))
                # print("====================================================")
            env.reset()  
                # round += 1
                # restore the initial state
        bw_hidden = bw_lstm.init_hidden(BATCH_SIZE)
        v_hiddens[i] = v_lstms[i].init_hidden(BATCH_SIZE)

        
    # print("####################################################")
    # policy grad updates:
    model.zero_grad()

    # large_batch
    # batch_size = memory.return_size()
    # batch_states, batch_actions, batch_returns, batch_advantages = memory.pop(batch_size)
    # batch_actions = batch_actions.view(batch_size, -1)
    # probs_pre = model_actor(batch_states.type(dtype))
    # values_pre = model_critic(batch_states.type(dtype))
    # prob_value = torch.gather(probs_pre.view(batch_size, -1, len(VIDEO_BIT_RATE)), dim=2, index=batch_actions.unsqueeze(2).type(dlongtype))
    # log_prob_value = torch.log(prob_value.squeeze(2))
    # policy_loss = -torch.mean(log_prob_value * batch_advantages.type(dtype))
    # loss_ent = ent_coeff * torch.mean(probs_pre * torch.log(probs_pre + 1e-5))
    # vf_loss = (values_pre - batch_returns.type(dtype)) ** 2 # V_\theta - Q'
    # critic_loss = 0.5 * torch.mean(vf_loss)
    # # update
    # actor_total_loss = policy_loss + loss_ent
    # optimizer_actor.zero_grad()
    # optimizer_critic.zero_grad()
    # actor_total_loss.backward()
    # torch.nn.utils.clip_grad_norm(model_actor.parameters(), 0.5)
    # optimizer_actor.step()
    # critic_loss.backward()
    # torch.nn.utils.clip_grad_norm(model_critic.parameters(), 0.5)
    # optimizer_critic.step()
    # ## test and save the model
    # epoch += 1
    # memory.clear()
    # a_losses.append(float(policy_loss.detach().cpu().numpy()))
    # v_losses.append(float(critic_loss.detach().cpu().numpy()))
    # entropies.append(A_DIM * float(loss_ent.detach().cpu().numpy()))
    # rewards.append(np.array(r).mean())


    # mini_batch
    batch_size = memory.return_size()
    batch_states, batch_actions, batch_returns, batch_advantages, batch_log_probs, batch_last_rewards, batch_last_actions, batch_last_times = memory.pop(batch_size)
    # batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-5)
    mini_batch_num = int(batch_size / batch_size)
    hidden = model.init_hidden(batch_size)
    hidden = (hidden[0].type(dtype).to(DEVICE), hidden[1].type(dtype).to(DEVICE))



    for update_step in tqdm(range(mini_batch_num)):
        tmp_batch_states = batch_states[update_step * batch_size: (update_step + 1) * batch_size].type(dtype).to(DEVICE)
        tmp_batch_advantages = batch_advantages[update_step * batch_size: (update_step + 1) * batch_size].type(dtype).to(DEVICE)
        # tmp_batch_advantages = tmp_batch_advantages.type(dtype)
        tmp_batch_log_probs = batch_log_probs[update_step * batch_size: (update_step + 1) * batch_size].type(dtype).to(DEVICE)
        # tmp_batch_log_probs = tmp_batch_log_probs.type(dtype)
        tmp_batch_last_rewards = batch_last_rewards[update_step * batch_size: (update_step + 1) * batch_size].type(dtype).to(DEVICE).unsqueeze(1)
        tmp_batch_last_actions = batch_last_actions[update_step * batch_size: (update_step + 1) * batch_size].type(dtype).to(DEVICE).unsqueeze(1)
        tmp_batch_last_times = batch_last_times[update_step * batch_size: (update_step + 1) * batch_size].type(dtype).to(DEVICE).unsqueeze(1)
        hidden = (hidden[0].detach(), hidden[1].detach())
        input = (tmp_batch_states, tmp_batch_last_rewards, tmp_batch_last_actions, tmp_batch_last_times, hidden)
        probs_pre, _, hidden = model(input)
        # print("prob_value: {} {}".format(prob_value.shape, prob_value))
        # print("prob_value shape: {}".format(prob_value.squeeze(2).shape))
        # print("tmp_batch_advantages: {}".format(tmp_batch_advantages))
        policy_loss = -torch.mean(tmp_batch_log_probs * tmp_batch_advantages.detach())
        # print("loss: {}".format((-log_prob_value * batch_advantages.type(dtype)).detach().cpu().numpy().tolist()))
        loss_ent = -ent_coeff * torch.mean(probs_pre * torch.log(probs_pre + 1e-5))

        # critic_loss
        # print("vf_loss: {}".format(vf_loss.detach().cpu().numpy().tolist()))
        # vf_loss = batch_advantages
        critic_loss = 0.5 * tmp_batch_advantages.pow(2).mean()

        # update
        actor_total_loss = policy_loss + 0.01 * loss_ent
        loss = actor_total_loss + critic_loss

        tmp_a_losses.append(float(policy_loss.detach().cpu().numpy()))
        tmp_v_losses.append(float(critic_loss.detach().cpu().numpy()))
        tmp_entropies.append(A_DIM * float(loss_ent.detach().cpu().numpy()))
    memory.clear()
    


    ## test and save the model
    epoch += 1
    a_losses.append(np.mean(np.array(tmp_a_losses)))
    v_losses.append(np.mean(np.array(tmp_v_losses)))
    entropies.append(np.mean(np.array(tmp_entropies)))
    rewards.append(np.array(r).mean())
    logging.info('Avg_policy_loss: ' + str(a_losses[-1]) +
                    ' Avg_value_loss: ' + str(v_losses[-1]) +
                    ' Avg_entropy_loss: ' + str(entropies[-1]) +
                    ' Avg reward: ' + str(rewards[-1])) 
    return a_losses, v_losses, entropies, rewards

def test_a2c(test_model, view_data, bw_data, qoe_set):
    print("Testing...")
    test_topic_name = "topic8"
    v_lstms, v_hiddens, processed_v_data, user_cluster = view_data
    bw_lstm, bw_hidden, processed_bw_data = bw_data
    model = MetaAC(A_DIM).type(dtype)
    model = model.to(DEVICE)
    model.load_state_dict(test_model.state_dict())
    model.eval()
    state = np.zeros((S_INFO,S_LEN))
    state = torch.from_numpy(state)
    last_bit_rate = DEFAULT_QUALITY
    # bit_rate = DEFAULT_QUALITY
    # action_vec = np.zeros(A_DIM)
    # action_vec[bit_rate] = 1

    done = True
    epoch = 0
    time_stamp = 0

    agent_num = 37
    episode_steps = 10
    ROUND = 6
    # update_num = 1
    # batch_size = exploration_size * episode_steps #64
    gamma = 1
    gae_param = 0.95
    ent_coeff = 1
    # cl_coeff = 0.2
    memory = ReplayMemory(agent_num * episode_steps * ROUND)
    env = Env(BANDWIDTH_TRACE_PATH)
    r = []
    tmp_a_losses = []
    tmp_v_losses = []
    tmp_entropies = []
    hidden = model.init_hidden(1)
    hidden = (hidden[0].type(dtype).to(DEVICE), hidden[1].type(dtype).to(DEVICE))
    for i in range(len(user_cluster) - 1):
        # if i != 0: continue
        # i = 0
        # round = 0
        ret_bw = []
        # hidden = model.init_hidden(1)
        # hidden = (hidden[0].type(dtype).to(DEVICE), hidden[1].type(dtype).to(DEVICE))
        for agent_id in tqdm(user_cluster[i]):
            # if agent_id != user_cluster_train[i][0]: continue
            # agent_id = user_cluster[i][0]
            # print("Group: {}, name: {}".format(i, agent_id))
            v_data = processed_v_data[i][test_topic_name][agent_id]
            agent = Agent("N")
            agent.name = agent_id
            agent.qoe_model = QoEModel(qoe_set[0], qoe_set[1], qoe_set[2])
            # print("Group: {}, name: {}".format(0, user_cluster[i][0]))
            # v_data = processed_v_data[i][train_topic_name][user_cluster[i][0]]
            # agent = Agent(user_cluster[i][0])
            # for round in range(ROUND):
            for round in range(ROUND):
                agent.next_steps()
                for step in range(episode_steps):
                    curr_step = step + round * episode_steps
                    if type(agent.state) != torch.Tensor: agent.state = torch.from_numpy(agent.state).type(dtype).to(DEVICE)
                    if type(agent.last_reward) != torch.Tensor: agent.last_reward = torch.tensor(agent.last_reward).type(dtype).to(DEVICE)
                    if type(agent.last_action) != torch.Tensor: agent.last_action = torch.tensor(agent.last_action).type(dtype).to(DEVICE)
                    if type(agent.last_time) != torch.Tensor: agent.last_time = torch.tensor(agent.last_time).type(dtype).to(DEVICE)
                    agent.last_rewards.append(agent.last_reward.unsqueeze(0))
                    agent.last_actions.append(agent.last_action.unsqueeze(0))
                    agent.last_times.append(agent.last_time.unsqueeze(0))
                    # print("last_reward: {}, last_action: {}, last_time:{}".format(agent.last_reward, agent.last_action, agent.last_time))

                    input_s = agent.state.unsqueeze(0).type(dtype).to(DEVICE)
                    input_r = agent.last_reward.view(1, 1, -1).type(dtype).to(DEVICE)
                    input_a = agent.last_action.view(1, 1, -1).type(dtype).to(DEVICE)
                    input_t = agent.last_time.view(1, 1, -1).type(dtype).to(DEVICE)
                    hidden = (hidden[0].detach(), hidden[1].detach())
                    input = (input_s, input_r, input_a, input_t, hidden)
                    prob, v, hidden = model(input)
                    prob = prob.view(-1, int(A_DIM / len(VIDEO_BIT_RATE)), len(VIDEO_BIT_RATE))
                    prob_p = prob[-1, :, :]
                    action = prob_p.multinomial(num_samples=1)
                    prob_value = torch.gather(prob, dim=2, index=action.unsqueeze(0).type(dlongtype))
                    log_prob_value = torch.log(prob_value.squeeze(2).squeeze(0))
                    agent.log_probs.append(log_prob_value.unsqueeze(0))
                    # print("input: {}, prob: {}, action: {}, prob_value: {}, log_prob_value: {}".format(input, prob, action, prob_value, log_prob_value))
                    # seed_ = np.random.uniform(0,1)
                    # if np.random.uniform(0,1) <= exploration_threhold:
                    #     action = random.randint(0, 5)
                    #     action = torch.tensor([[action]]).type(dlongtype)
                    # else:
                    #     action = prob.multinomial(num_samples=1)
                    agent.values.append(v)

                    bitrate_levels = action.squeeze().cpu().numpy()

                    agent.actions.append(action)
                    
                    agent.last_action = torch.tensor(bitrate_levels)
                    
                    
                    agent.states.append(agent.state.unsqueeze(0))
                    if curr_step != 0:
                        bit_rates = np.array([VIDEO_BIT_RATE[bitrate_level] for bitrate_level in bitrate_levels])
                    else:
                        bit_rates = np.array([1, 1])
                    # bit_rate = np.mean(bit_rates)
                    # tile_bit_rates = bit_rates.reshape(WIDTH_TILE_NUMBER, HEIGHT_TILE_NUMBER)
                    chunk_size = cal_chunk_size(VIDEO_CONFIG, bit_rates, CHUNK_PERIOD)
                    bit_rate = chunk_size / CHUNK_PERIOD
                    tile_bit_rates = generate_bitrate(VIDEO_CONFIG, bit_rates, agent.state[2, :2].cpu().numpy())
                    agent.last_update_time = agent.curr_time
                    
                    agent.last_time = torch.tensor(agent.curr_time)                     
                    agent.curr_time = env.push_video_chunk(chunk_size)
                    env.update(agent.curr_time)
                    buffer_remaining = env.buffer_remaining
                    rebuffer_time = env.rebuffer_time
                    viewport_gt = [int(WIDTH_TILE_NUMBER * v_data[curr_step][1].numpy()[0][0]), int(HEIGHT_TILE_NUMBER * v_data[curr_step][1].numpy()[0][1])]
                    _, reward, wq = agent.qoe_model.cal_qoe(tile_bit_rates, VIDEO_CONFIG, viewport_gt, rebuffer_time)
                    agent.rewards.append(reward)
                    
                    agent.last_reward = torch.tensor(reward)
                    if curr_step!=0: r.append(reward)
                    # print("current time: {}".format(agent.curr_time))
                    # print("bitrate: {}, avg bitrate: {}, window quality: {}, true viewport: {}, buffer_remaining: {}, rebuffer_time: {}, reward: {}".format(tile_bit_rates, bit_rate, wq, viewport_gt, buffer_remaining, rebuffer_time, reward))

                    if int(agent.curr_time) != 0 and int(agent.curr_time) != int(agent.last_update_time):
                        ret_bw = []
                        t = int(agent.curr_time) % (len(env.bandwidth_trace) - 2)
                        bw_hidden = predict_next_few_seconds_bw(bw_lstm, processed_bw_data[t:t + 2], bw_hidden, ret_bw)  
                    elif int(agent.curr_time) == 0:
                        ret_bw = [[VIDEO_BIT_RATE[DEFAULT_QUALITY], "Unknown"], [VIDEO_BIT_RATE[DEFAULT_QUALITY], "Unknown"]]
                    # print("predict bandwidth for next 2s: {} (gt:{}), {} (gt:{})".format(ret_bw[0][0], ret_bw[0][1], ret_bw[1][0], ret_bw[1][1]))

                    ret_viewport = []
                    if curr_step != 0:
                        # print(v_data[curr_step:curr_step + 1])
                        v_hiddens[i] = predict_next_few_seconds_viewport(v_lstms[i], v_data[curr_step:curr_step + 1], i, v_hiddens[i], ret_viewport)
                        ret_viewport = [ret_viewport[0][0], ret_viewport[0][1]]
                    else:
                        ret_viewport = [(int(0.5 * WIDTH_TILE_NUMBER), int(0.5 * HEIGHT_TILE_NUMBER)), (None, None)]
                    # print("predict viewport for next 1s: {} (gt:{})".format(ret_viewport[0], ret_viewport[1]))
                    agent.init_state()
                    # agent.state[0, -1] = agent.curr_time # agent.curr_time - agent.last_update_time # int() / 100.0
                    agent.state[0, -1] = agent.curr_time - agent.last_update_time
                    agent.state[1, -1] = buffer_remaining
                    agent.state[2, :2] = np.array(ret_viewport[0])
                    agent.state[3, :2] = np.array([ret_bw[0][0], ret_bw[1][0]])
                    # print("current state: {}".format(agent.state))
                    agent.state = torch.from_numpy(agent.state)
                    

                if round == 0: agent.clip()
                # one last step
                R = torch.zeros(1, 1)
                # if end_of_video == False:
                input_s = agent.state.unsqueeze(0).type(dtype).to(DEVICE)
                input_r = agent.last_reward.view(1, 1, -1).type(dtype).to(DEVICE)
                input_a = agent.last_action.view(1, 1, -1).type(dtype).to(DEVICE)
                input_t = agent.last_time.view(1, 1, -1).type(dtype).to(DEVICE)
                hidden = (hidden[0].detach(), hidden[1].detach())
                input = (input_s, input_r, input_a, input_t, hidden)
                _, v, _ = model(input)
                # v = v.detach().cpu()
                # R = v.data
                R = v.detach()
                #================================结束一个ep========================================
                # compute returns and GAE(lambda) advantages:
                # values.append(Variable(R))
                # R = Variable(R)
                # A = Variable(torch.zeros(1, 1))
                # for i in reversed(range(len(rewards))):
                #     td = rewards[i] + gamma * values[i + 1].data[0, 0] - values[i].data[0, 0]
                #     A = float(td) + gamma * gae_param * A
                #     advantages.insert(0, A)
                #     # R = A + values[i]
                #     R = gamma * R + rewards[i]
                #     # R = Variable(R)
                #     returns.insert(0, R)

                ## compute returns and advantages with Monte Carlo sampling
                agent.values.append(R)
                # print("v: {}".format(agent.values))
                # print("r: {}".format(agent.rewards))
                # td = Variable(torch.zeros(1, 1))
                for j in reversed(range(len(agent.rewards))):
                    R = gamma * R + agent.rewards[j]
                    agent.returns.insert(0, R)
                    td = R - agent.values[j]
                    # print("R: {}, v: {}, td: {}".format(R, agent.values[j], td))
                    agent.advantages.insert(0, td) 
                # store usefull info:
                # memory.push([states[1:], actions[1:], rewards_comparison[1:], returns[1:], advantages[1:]])
                # memory.push([states[1:], actions[1:], returns[1:], advantages[1:]])
                if torch.eq(agent.states[0][0], torch.from_numpy(np.zeros((S_INFO,S_LEN)))).sum() == S_INFO * S_LEN: ## judge if states[0] equals to torch.from_numpy(np.zeros((S_INFO,S_LEN)))
                    memory.push([agent.states[1:], agent.actions[1:], agent.returns[1:], agent.advantages[1:], agent.log_probs[1:], agent.last_rewards[1:], agent.last_actions[1:], agent.last_times[1:]])
                else:  
                    memory.push([agent.states, agent.actions, agent.returns, agent.advantages, agent.log_probs, agent.last_rewards, agent.last_actions, agent.last_times])
                # print(memory.memory)
                # memory.push([agent.states[1:], agent.actions[1:], agent.returns[1:], agent.advantages[1:]])
                # print(len(memory.memory))
                # print("====================================================")
            env.reset()  
                # round += 1
                # restore the initial state
        bw_hidden = bw_lstm.init_hidden(BATCH_SIZE)
        v_hiddens[i] = v_lstms[i].init_hidden(BATCH_SIZE)
    return r, np.array(r).mean()


def test_basic(view_data, bw_data, qoe_set):
    print("Testing...")
    test_topic_name = "topic8"
    v_lstms, v_hiddens, processed_v_data, user_cluster = view_data
    bw_lstm, bw_hidden, processed_bw_data = bw_data

    agent_num = 37
    episode_steps = 10
    ROUND = 6
    env = Env(BANDWIDTH_TRACE_PATH)
    r = []
    for i in range(len(user_cluster) - 1):
        ret_bw = []
        for agent_id in tqdm(user_cluster[i]):
            v_data = processed_v_data[i][test_topic_name][agent_id]
            agent = Agent("N")
            agent.name = agent_id
            agent.qoe_model = QoEModel(qoe_set[0], qoe_set[1], qoe_set[2])
            for round in range(ROUND):
                agent.next_steps()
                for step in range(episode_steps):
                    curr_step = step + round * episode_steps
                    if curr_step != 0:
                        for rate in reversed(VIDEO_BIT_RATE):
                            if rate <= ret_bw[1][0]:
                                bit_rates = np.array([rate, rate])
                                break
                    else:
                        bit_rates = np.array([1, 1])
                    # bit_rate = np.mean(bit_rates)
                    # tile_bit_rates = bit_rates.reshape(WIDTH_TILE_NUMBER, HEIGHT_TILE_NUMBER)
                    chunk_size = cal_chunk_size(VIDEO_CONFIG, bit_rates, CHUNK_PERIOD)
                    tile_bit_rates = generate_bitrate(VIDEO_CONFIG, bit_rates, (0, 0))
                    agent.last_update_time = agent.curr_time
                    
                    agent.last_time = torch.tensor(agent.curr_time)                     
                    agent.curr_time = env.push_video_chunk(chunk_size)
                    env.update(agent.curr_time)
                    buffer_remaining = env.buffer_remaining
                    rebuffer_time = env.rebuffer_time
                    viewport_gt = [int(WIDTH_TILE_NUMBER * v_data[curr_step][1].numpy()[0][0]), int(HEIGHT_TILE_NUMBER * v_data[curr_step][1].numpy()[0][1])]
                    _, reward, wq = agent.qoe_model.cal_qoe(tile_bit_rates, VIDEO_CONFIG, viewport_gt, rebuffer_time)
                    
                    agent.last_reward = torch.tensor(reward)
                    if curr_step!=0: r.append(reward)
                    # print("current time: {}".format(agent.curr_time))
                    # print("bitrate: {}, avg bitrate: {}, window quality: {}, true viewport: {}, buffer_remaining: {}, rebuffer_time: {}, reward: {}".format(tile_bit_rates, bit_rate, wq, viewport_gt, buffer_remaining, rebuffer_time, reward))

                    if int(agent.curr_time) != 0 and int(agent.curr_time) != int(agent.last_update_time):
                        ret_bw = []
                        t = int(agent.curr_time) % (len(env.bandwidth_trace) - 2)
                        bw_hidden = predict_next_few_seconds_bw(bw_lstm, processed_bw_data[t:t + 2], bw_hidden, ret_bw)  
                    elif int(agent.curr_time) == 0:
                        ret_bw = [[VIDEO_BIT_RATE[DEFAULT_QUALITY], "Unknown"], [VIDEO_BIT_RATE[DEFAULT_QUALITY], "Unknown"]]

                if round == 0: agent.clip()
            env.reset() 
    return r, np.array(r).mean()

def test_lr(view_data, bw_data, qoe_set):
    print("Testing...")
    test_topic_name = "topic8"
    v_lstms, v_hiddens, processed_v_data, user_cluster = view_data
    bw_lstm, bw_hidden, processed_bw_data = bw_data

    agent_num = 37
    episode_steps = 10
    ROUND = 6
    env = Env(BANDWIDTH_TRACE_PATH)
    r = []
    for i in range(len(user_cluster) - 1):
        ret_bw = []
        for agent_id in tqdm(user_cluster[i]):
            v_data = processed_v_data[i][test_topic_name][agent_id]
            agent = Agent("N")
            agent.name = agent_id
            agent.qoe_model = QoEModel(qoe_set[0], qoe_set[1], qoe_set[2])
            buffer_remaining = 0
            for round in range(ROUND):
                agent.next_steps()
                for step in range(episode_steps):
                    curr_step = step + round * episode_steps
                    if curr_step != 0:
                        bitrate = max(1, (buffer_remaining + 1 - BUFFER_TARGET) * ret_bw[1][0])
                        in_bitrate = cal_max_in_bitrate(VIDEO_CONFIG, 5, bitrate)
                        for rate in reversed(VIDEO_BIT_RATE):
                            if rate <= in_bitrate:
                                bit_rates = np.array([5, rate])
                                break
                    else:
                        bit_rates = np.array([5, 5])
                    # print(bit_rates)
                    # bit_rate = np.mean(bit_rates)
                    # tile_bit_rates = bit_rates.reshape(WIDTH_TILE_NUMBER, HEIGHT_TILE_NUMBER)
                    chunk_size = cal_chunk_size(VIDEO_CONFIG, bit_rates, CHUNK_PERIOD)
                    viewport_pred = predict_viewport_lr(v_data[curr_step][0])
                    viewport_pred = [int(WIDTH_TILE_NUMBER * viewport_pred[0]), int(HEIGHT_TILE_NUMBER * viewport_pred[1])]
                    tile_bit_rates = generate_bitrate(VIDEO_CONFIG, bit_rates, viewport_pred)
                    # print(tile_bit_rates)
                    agent.last_update_time = agent.curr_time
                    
                    
                    agent.last_time = torch.tensor(agent.curr_time)                     
                    agent.curr_time = env.push_video_chunk(chunk_size)
                    env.update(agent.curr_time)
                    buffer_remaining = env.buffer_remaining
                    rebuffer_time = env.rebuffer_time
                    viewport_gt = [int(WIDTH_TILE_NUMBER * v_data[curr_step][1].numpy()[0][0]), int(HEIGHT_TILE_NUMBER * v_data[curr_step][1].numpy()[0][1])]
                    # print("pred: {}, gt: {}".format(viewport_pred, viewport_gt))
                    _, reward, wq = agent.qoe_model.cal_qoe(tile_bit_rates, VIDEO_CONFIG, viewport_gt, rebuffer_time)
                    
                    agent.last_reward = torch.tensor(reward)
                    if curr_step!=0: r.append(reward)
                    # print("current time: {}".format(agent.curr_time))
                    # print("bitrate: {}, avg bitrate: {}, window quality: {}, true viewport: {}, buffer_remaining: {}, rebuffer_time: {}, reward: {}".format(tile_bit_rates, bit_rates, wq, viewport_gt, buffer_remaining, rebuffer_time, reward))

                    if int(agent.curr_time) != 0 and int(agent.curr_time) != int(agent.last_update_time):
                        ret_bw = []
                        t = int(agent.curr_time) % (len(env.bandwidth_trace) - 2)
                        bw_hidden = predict_next_few_seconds_bw(bw_lstm, processed_bw_data[t:t + 2], bw_hidden, ret_bw)  
                    elif int(agent.curr_time) == 0:
                        ret_bw = [[VIDEO_BIT_RATE[DEFAULT_QUALITY], "Unknown"], [VIDEO_BIT_RATE[DEFAULT_QUALITY], "Unknown"]]

                if round == 0: agent.clip()
            env.reset() 
    return r, np.array(r).mean()

def plot(data, legend):
    plt.figure(dpi=600)
    plt.plot(data, 'r', label=legend, lw=1, alpha=0.8)
    plt.legend(loc="best")
    os.makedirs("./Results/" + curr_strftime + "_" + args.model + "_ftrain_" + args.ftrain_set, exist_ok=True)
    plt.savefig("./Results/" + curr_strftime + "_" + args.model + "_ftrain_" + args.ftrain_set + "/" + legend + '.jpg')
 
def plot2(data1, data2, legend1, legend2):
    data2 = [10000 * item for item in data2]
    plt.figure(dpi=600)
    plt.plot(data1, 'r', label=legend1, lw=1, alpha=0.8)
    plt.plot(data2, 'b', label=legend2, lw=1, alpha=0.8)
    plt.legend(loc="best")
    os.makedirs("./Results/" + curr_strftime + "_" + args.model, exist_ok=True)
    plt.savefig("./Results/" + curr_strftime + "_" + args.model + "/" + legend1 +  "_and_" + legend2 + '.jpg')

def main():
    (a_losses_train, v_losses_train, entropies_train, rewards_train), (a_losses_valid, v_losses_valid, entropies_valid, rewards_valid), rewards_test = train_a2c()
    print("a_losses_train: {}".format(a_losses_train))
    print("v_losses_train: {}".format(v_losses_train))
    print("entropies_train: {}".format(entropies_train))
    print("rewards_train: {}".format(rewards_train))

    print("")
    print("a_losses_valid: {}".format(a_losses_valid))
    print("v_losses_valid: {}".format(v_losses_valid))
    print("entropies_valid: {}".format(entropies_valid))
    print("rewards_valid: {}".format(rewards_valid))

    print("")
    print("rewards_test: {}".format(rewards_test))

    plot(a_losses_train, "actor_losses_train")
    plot(v_losses_train, "critic_losses_train")
    plot(entropies_train, "entropies_train")
    plot(rewards_train, "rewards_train")
    for i in range(len(QOE_SET)):
        plot(a_losses_valid[i], "actor_losses_valid" + "_" + str(QOE_SET[i][0]) + "_" + str(QOE_SET[i][1]) + "_" + str(QOE_SET[i][2]))
        plot(v_losses_valid[i], "critic_losses_valid" + "_" + str(QOE_SET[i][0]) + "_" + str(QOE_SET[i][1]) + "_" + str(QOE_SET[i][2]))
        plot(entropies_valid[i], "entropies_valid" + "_" + str(QOE_SET[i][0]) + "_" + str(QOE_SET[i][1]) + "_" + str(QOE_SET[i][2]))
        plot(rewards_valid[i], "rewards_valid" + "_" + str(QOE_SET[i][0]) + "_" + str(QOE_SET[i][1]) + "_" + str(QOE_SET[i][2]))
        plot(rewards_test[i], "rewards_test" + "_" + str(QOE_SET[i][0]) + "_" + str(QOE_SET[i][1]) + "_" + str(QOE_SET[i][2]))

    # plot2(v_losses_train, rewards_train, "critic_losses_train", "rewards_train")

if __name__ == '__main__':
    main()
