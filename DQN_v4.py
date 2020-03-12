"""
Double DQN (Nature 2015)
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
Notes:
    The difference is that now there are two DQNs (DQN & Target DQN)
    y_i = r_i + 𝛾 * max(Q(next_state, action; 𝜃_target))
    Loss: (y_i - Q(state, action; 𝜃))^2
    Every C step, 𝜃_target <- 𝜃
"""
import numpy as np
import tensorflow as tf
import random
from collections import deque
# import dqn4 as dqn
import dqn4_with_RNN as dqn
import matplotlib.pyplot as plt
import copy
# import gym
from typing import List
from Simulation_v4 import Simulation



INPUT_SIZE = 26
OUTPUT_SIZE = 3

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
BATCH_SIZE = 80
TARGET_UPDATE_FREQUENCY = 1
MAX_EPISODES = 20000



def replay_train(mainDQN: dqn.DQN, targetDQN: dqn.DQN, train_batch: list) -> float:

    # print(train_batch[0])
    x_stack = np.empty(0).reshape(0, INPUT_SIZE)
    y_stack = np.empty(0).reshape(0, OUTPUT_SIZE)

    for state, action, reward, next_state in train_batch:
        Q = mainDQN.predict(state)
        # print(Q)

        Q[0, action] = reward + DISCOUNT_RATE * np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])


    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack)

def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:
    """Creates TF operations that copy weights from `src_scope` to `dest_scope`
    Args:
        dest_scope_name (str): Destination weights (copy to)
        src_scope_name (str): Source weight (copy from)
    Returns:
        List[tf.Operation]: Update operations are created and returned
    """
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

def main():
    # store the previous observations in replay memory



    replay_buffer = deque(maxlen=REPLAY_MEMORY)
    tot_reward_history = []
    tot_cost_history = []
    tot_soc_history = []
    tot_diff_soc_history = []

    pdata = np.loadtxt("data/smp_data.csv", delimiter=',', dtype=str)
    pdata = pdata[:, 1:]
    pdata = pdata[::-1]
    pdata = pdata.astype(np.float32)

    price_data = pdata[:, :-3]


    data_set = price_data.reshape(-1, 1)


    train_day = int(len(data_set)/24 * 0.8)
    train_size = train_day*24

    test_day = int(len(data_set)/24 - train_day)
    test_size = test_day*24

    train_set = data_set[0:train_size]
    test_set = data_set[train_size:]

    train_set = MinMaxScaler(train_set)
    test_set = MinMaxScaler(test_set)

    # print(price_data[train_day])
    # print(test_set[0:24])

    with tf.Session() as sess:

        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        sess.run(tf.global_variables_initializer())

        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)
        step = 0
        for epi in range(1, MAX_EPISODES):
        # for episode in range(1, 2):
            day = np.random.randint(1, train_day)

            e = 1. / ((epi / 200) + 1)
            print("\n################### {0}-th....   e:{1:03.2f} ########################".format(epi, e))

            sim = Simulation(1)  # slot/hour == 1
            ev = sim.sim_init(price_data[day])

            tot_reward = 0
            tot_cost = 0
            done = 0

            ts = ev.TS_Arrive
            past_24_price_data = train_set[(day - 1) * 24 + ts:day * 24 + ts]
            past_24_price_data = np.reshape(past_24_price_data, [1, -1])

            soc = ev.SoC
            remainTS = ev.TS_Depart - ts
            state = np.array([soc, remainTS])
            state = np.reshape(state, [1, -1])
            state = np.concatenate((state, past_24_price_data), axis=1)

            while done == 0:
                # print("State: ", state)
                Qpre = mainDQN.predict(state)

                if np.random.rand() < e:
                    action = np.random.randint(0, 3)
                else:
                    action = np.argmax(Qpre)

                # print('Origin Action: ', action)
                nextstate, next_ts, reward, done, cost, amount, action = sim.sim_step_LSTM(action, ev, ts)
                nextstate = np.reshape(nextstate, [1, -1])
                tot_cost += cost
                past_24_price_data = train_set[(day - 1) * 24 + next_ts:day * 24 + next_ts]
                past_24_price_data = np.reshape(past_24_price_data, [1, -1])
                nextstate = np.concatenate((nextstate, past_24_price_data), axis=1)
                # print('Adjust Action: ', action)
                print(action, end=', ')

                if done == 1:
                    reward = cost - 150*(ev.battery_capa - ev.cur_bat_power)
                else:
                    reward = cost
                #todo make


                tot_reward += reward
                replay_buffer.append((state, action, reward, nextstate))
                state = nextstate
                ts = next_ts

                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)

                if step % 100 == 0:
                    sess.run(copy_ops)
                step += 1
            tot_reward_history.append(tot_reward)
            tot_cost_history.append(tot_cost)
            tot_soc_history.append(ev.SoC)
            tot_diff_soc_history.append(ev.SoC - ev.init_SoC)

            print('\nTot Reward: {0:9.2f}'.format(tot_reward))
            print('initial SoC: {0:05.4f}, final SoC: {1:05.4f}'.format(ev.init_SoC, ev.SoC))
            print('BT: {0:05.2f}, current BT: {1:05.2f}'.format(ev.battery_capa,ev.cur_bat_power))
            print('Cost: {0:06.2f}'.format(tot_cost))
            print('Charing: {0:06.2f}, Discharing: {1:06.2f}'.format(np.sum(sim.charging_load_list_grid), np.sum(sim.discharging_load_list_ev)))

        plt.plot(tot_reward_history)
        plt.show(block=False)
        fig = plt.gcf()
        fig.savefig('result/tot_reward_history.png', dpi=fig.dpi)
        plt.clf()

        plt.plot(tot_cost_history)
        plt.show(block=False)
        fig = plt.gcf()
        fig.savefig('result/tot_cost_history.png', dpi=fig.dpi)
        plt.clf()

        plt.plot(tot_soc_history)
        plt.show(block=False)
        fig = plt.gcf()
        fig.savefig('result/tot_soc_history.png', dpi=fig.dpi)
        plt.clf()

        plt.plot(tot_diff_soc_history)
        plt.show(block=False)
        fig = plt.gcf()
        fig.savefig('result/tot_diff_soc_history.png', dpi=fig.dpi)
        plt.clf()


##########################################################################################################

        for test in range(1, test_day):
        # for test in range(1, 3):
            print("\n################### RL TEST {}-th....    ########################".format(test))
            sim = Simulation(1)  # slot/hour == 1
            ev = sim.sim_init(price_data[train_day+test])

            tot_reward = 0
            tot_cost = 0
            done = 0

            ts = ev.TS_Arrive
            past_24_price_data = test_set[(test-1) * 24 + ts:test * 24 + ts]
            past_24_price_data = np.reshape(past_24_price_data, [1, -1])

            soc = ev.SoC
            remainTS = ev.TS_Depart - ts
            state = np.array([soc, remainTS])
            state = np.reshape(state, [1, -1])
            state = np.concatenate((state, past_24_price_data), axis=1)

            while done == 0:
                print("State: ", state)
                Qpre = mainDQN.predict(state)


                action = np.argmax(Qpre)

                print('Origin Action: ', action)
                nextstate, next_ts, reward, done, cost, amount, action = sim.sim_step_LSTM(action, ev, ts)
                nextstate = np.reshape(nextstate, [1, -1])
                tot_cost += cost
                past_24_price_data = test_set[(test-1) * 24 + next_ts:test * 24 + next_ts]
                past_24_price_data = np.reshape(past_24_price_data, [1, -1])
                nextstate = np.concatenate((nextstate, past_24_price_data), axis=1)
                print('Adjust Action: ', action)

                if done == 1:
                    reward = cost - 150*(ev.battery_capa - ev.cur_bat_power)
                else:
                    reward = cost


                tot_reward += reward
                state = nextstate
                ts = next_ts

            print('Tot Reward: {0:9.2f}'.format(tot_reward))
            print('initial SoC: {0:05.4f}, final SoC: {1:05.4f}'.format(ev.init_SoC, ev.SoC))
            print('Cost: {0:06.2f}'.format(tot_cost))
            print('Charing: {0:06.2f}, Discharing: {1:06.2f}'.format(np.sum(sim.charging_load_list_grid),
                                                                     np.sum(sim.discharging_load_list_ev)))
            if test % 10 == 0:
                B_line, = plt.plot(sim.today_basecost)
                C_line, = plt.plot(sim.charging_load_list_grid)
                D_line, = plt.plot(sim.discharging_load_list_ev)
                plt.legend(handles=(B_line, C_line, D_line), labels=('Price', 'Charging', 'Discharging'))
                plt.show(block=False)
                fig = plt.gcf()
                fig.savefig('result/Random_test_day_{}.png'.format(test), dpi=fig.dpi)
                plt.clf()

        for test in range(1, test_day):
        # for test in range(1, 3):

            print("\n################### Random TEST {}-th....    ########################".format(test))
            sim = Simulation(1)  # slot/hour == 1
            ev = sim.sim_init(price_data[train_day+test])

            tot_reward = 0
            tot_cost = 0
            done = 0

            ts = ev.TS_Arrive
            past_24_price_data = test_set[(test-1) * 24 + ts:test * 24 + ts]
            past_24_price_data = np.reshape(past_24_price_data, [1, -1])

            soc = ev.SoC
            remainTS = ev.TS_Depart - ts
            state = np.array([soc, remainTS])
            state = np.reshape(state, [1, -1])
            state = np.concatenate((state, past_24_price_data), axis=1)

            while done == 0:
                print("State: ", state)
                Qpre = mainDQN.predict(state)


                action = np.random.randint(0, 3)

                print('action: ', action)
                nextstate, next_ts, reward, done, cost, amount, action = sim.sim_step_LSTM(action, ev, ts)
                nextstate = np.reshape(nextstate, [1, -1])
                tot_cost += cost
                past_24_price_data = train_set[(test - 1) * 24 + next_ts:test * 24 + next_ts]
                past_24_price_data = np.reshape(past_24_price_data, [1, -1])
                nextstate = np.concatenate((nextstate, past_24_price_data), axis=1)

                if done == 1:
                    reward = cost - 150*(ev.battery_capa - ev.cur_bat_power)
                else:
                    reward = cost

                tot_reward += reward
                state = nextstate
                ts = next_ts

            print('Tot Reward: {0:9.2f}'.format(tot_reward))
            print('initial SoC: {0:05.4f}, final SoC: {1:05.4f}'.format(ev.init_SoC, ev.SoC))
            print('Cost: {0:06.2f}'.format(tot_cost))
            print('Charing: {0:06.2f}, Discharing: {1:06.2f}'.format(np.sum(sim.charging_load_list_grid),
                                                                     np.sum(sim.discharging_load_list_ev)))
            if test % 10 == 0:
                B_line, = plt.plot(sim.today_basecost)
                C_line, = plt.plot(sim.charging_load_list_grid)
                D_line, = plt.plot(sim.discharging_load_list_ev)
                plt.legend(handles=(B_line, C_line, D_line), labels=('Price','Charging', 'Discharging'))
                plt.show(block=False)
                fig = plt.gcf()
                fig.savefig('result/Random_test_day_{}.png'.format(test), dpi=fig.dpi)
                plt.clf()

if __name__ == "__main__":
    main()