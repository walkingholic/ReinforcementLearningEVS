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
import dqn4 as dqn
import matplotlib.pyplot as plt
import copy
# import gym
from typing import List
from Simulation_v4 import Simulation


sim = Simulation(100, 0)
INPUT_SIZE = 6
OUTPUT_SIZE = 3

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
BATCH_SIZE = 80
TARGET_UPDATE_FREQUENCY = 1
MAX_EPISODES = 3000

filename = 'data/load.txt'
fo = open(filename, 'w')


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

# def bot_play(mainDQN: dqn.DQN, env: gym.Env) -> None:
#     """Test runs with rendering and prints the total score
#     Args:
#         mainDQN (dqn.DQN): DQN agent to run a test
#         env (gym.Env): Gym Environment
#     """
#     state = env.reset()
#     reward_sum = 0
#
#     while True:
#
#         env.render()
#         action = np.argmax(mainDQN.predict(state))
#         state, reward, done, _ = env.step(action)
#         reward_sum += reward
#
#         if done:
#             print("Total score: {}".format(reward_sum))
#             break

def main():
    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=REPLAY_MEMORY)
    tot_reward_history = []
    tot_cost_history = []
    tot_soc_history = []
    tot_diff_soc_history = []
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        sess.run(tf.global_variables_initializer())

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(MAX_EPISODES):
            e = 1. / ((episode / 150) + 1)
            tot_reward = 0
            tot_cost = 0
            done = 0
            step_count = 0
            ev = sim.sim_init( np.random.randint(0, 7) )
            # ev = sim.sim_init(0)
            # slot number, soc, at, dt, curr load, load state
            ts = ev.TS_Arrive
            soc = ev.SoC*100
            curload = sim.baseload[ts]
            # loadstate = sim.sim_get_peak_price_at_ts(ts)
            loadstate = sim.sim_get_load_state(ts)
            state = np.array([ts, soc, ev.TS_Arrive, ev.TS_Depart, curload, loadstate])

            print("\nEpisode: {0} e:{1:06.4f} ".format(episode, e))
            # ev.get_info_EV();

            while done == 0 :
                Qpre = mainDQN.predict(state)
                if np.random.rand() < e:
                    action = np.random.randint(0, 3)
                else:
                    action = np.argmax(Qpre)
                next_state, reward, done, cost, amount= sim.sim_step(action, ev, ts)
                # print('TS: {0}, Action: {1}, Reward: {2:9.2f}, SoC: {3:05.4f}, loadstate: {4} cur bat: {5:05.4f}'.format(ts, action, reward, ev.SoC, loadstate, ev.cur_bat_power))
                tot_cost += cost
                ts += 1
                if done == 1:
                    reward = 100
                elif done == -1:
                    reward = -50
                elif done == -2:
                    reward = -100
                tot_reward += reward
                replay_buffer.append((state, action, reward, next_state))
                state = next_state

                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)

            if episode % TARGET_UPDATE_FREQUENCY == 0:
                sess.run(copy_ops)
            tot_reward_history.append(tot_reward)
            tot_cost_history.append(tot_cost)
            tot_soc_history.append(ev.SoC)
            tot_diff_soc_history.append(ev.SoC - ev.init_SoC)
            print('Tot Reward: {0:9.2f}'.format(tot_reward))
            print('initial SoC: {0:05.4f}, final SoC: {1:05.4f}'.format(ev.init_SoC, ev.SoC))
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

        tot_reward_history.clear()
        tot_cost_history.clear()
        tot_soc_history.clear()
        tot_diff_soc_history.clear()


        for day in range(1):
            sim.sim_init_test(day)
            print('day', day)
            for ts in range(96):
                print('##################################################  ts : ', ts)
                sim.sim_check_EVs(ts)

                e = 0
                print('Entry: ', len(sim.entry_EV))
                print('Stay: ', len(sim.entry_EV_Stay))
                print('Depart: ', len(sim.entry_EV_Depart))
                while e < len(sim.entry_EV_Stay):
                    ev = sim.entry_EV_Stay[e]
                    soc = ev.SoC
                    curload = sim.baseload[ts]+sim.charging_load_list_grid[ts]+sim.discharging_load_list_grid[ts]
                    loadstate = sim.sim_get_load_state(ts)
                    state = np.array([ts, soc, ev.TS_Arrive, ev.TS_Depart, curload, loadstate])

                    # Qpre = mainDQN.predict(state)
                    # action = np.argmax(Qpre)
                    action = np.random.randint(0, 3)
                    state, reward, done, cost, amount = sim.sim_step(action, ev, ts)

                    sim.sim_depart_check_EVs(ev, ts, done)
                    if done == 0 :
                        e += 1

            plt.title('Random')
            plt.plot(sim.baseload)
            plt.plot(sim.charging_load_list_grid)
            plt.plot(sim.discharging_load_list_grid)
            plt.plot(sim.baseload + sim.charging_load_list_grid + sim.discharging_load_list_grid)
            plt.show()


        for day in range(1):
            sim.sim_init_test(day)
            print('day', day)
            for ts in range(96):
                print('##################################################  ts : ', ts)
                sim.sim_check_EVs(ts)

                e = 0
                print('Entry: ', len(sim.entry_EV))
                print('Stay: ', len(sim.entry_EV_Stay))
                print('Depart: ', len(sim.entry_EV_Depart))
                while e < len(sim.entry_EV_Stay):
                    ev = sim.entry_EV_Stay[e]
                    soc = ev.SoC
                    curload = sim.baseload[ts]+sim.charging_load_list_grid[ts]+sim.discharging_load_list_grid[ts]
                    loadstate = sim.sim_get_load_state(ts)
                    state = np.array([ts, soc, ev.TS_Arrive, ev.TS_Depart, curload, loadstate])

                    Qpre = mainDQN.predict(state)
                    action = np.argmax(Qpre)
                    # action = np.random.randint(0, 3)
                    state, reward, done, cost, amount = sim.sim_step(action, ev, ts)

                    sim.sim_depart_check_EVs(ev, ts, done)
                    if done == 0 :
                        e += 1

            plt.title('Reinforcement')
            plt.plot(sim.baseload)
            plt.plot(sim.charging_load_list_grid)
            plt.plot(sim.discharging_load_list_grid)
            plt.plot(sim.baseload + sim.charging_load_list_grid + sim.discharging_load_list_grid)
            plt.show()


        #
        #
        #
        #
        #
        #
        #
        #
        #
        # for n in range(1000):
        #     tot_reward = 0
        #     tot_cost = 0
        #     done = 0
        #     ev = sim.sim_init(0)
        #     # slot number, soc, at, dt, curr load, load state
        #     ts = ev.TS_Arrive
        #     soc = ev.SoC*100
        #     curload = sim.baseload[ts]
        #     # loadstate = sim.sim_get_peak_price_at_ts(ts)
        #     loadstate = sim.sim_get_load_state(ts)
        #
        #     state = np.array([ts, soc, ev.TS_Arrive, ev.TS_Depart, curload, loadstate])
        #
        #     while done == 0:
        #         Qpre = mainDQN.predict(state)
        #         action = np.argmax(Qpre)
        #
        #         state, reward, done, cost, amount = sim.sim_step(action, ev, ts)
        #         # print('TS: {0}, Action: {1}, Reward: {2:9.2f}, SoC: {3:05.4f}, loadstate: {4} cur bat: {5:05.4f} cost: {6:05.2f}'.format(ts, action, reward, ev.SoC, loadstate, ev.cur_bat_power, cost))
        #         tot_cost += cost
        #         ts += 1
        #         if done == 1:
        #             reward = 100
        #         elif done == -1:
        #             reward = -50
        #         elif done == -2:
        #             reward = -100
        #         tot_reward += reward
        #
        #     tot_reward_history.append(tot_reward)
        #     tot_cost_history.append(tot_cost)
        #     tot_soc_history.append(ev.SoC)
        #     tot_diff_soc_history.append(ev.SoC - ev.init_SoC)
        #
        #     print('Tot Reward: {0:9.2f}'.format(tot_reward))
        #     print('initial SoC: {0:05.4f}, final SoC: {1:05.4f}'.format(ev.init_SoC, ev.SoC))
        #     print('Cost: {0:06.2f}'.format(tot_cost))
        #     print('Charing: {0:06.2f}, Discharing: {1:06.2f}'.format(np.sum(sim.charging_load_list),
        #                                                          np.sum(sim.discharging_load_list_ev)))
        #
        #
        #
        # plt.bar(np.arange(len(tot_reward_history)), tot_reward_history)
        # plt.show(block=False)
        # fig = plt.gcf()
        # fig.savefig('result/test_tot_reward_history.png', dpi=fig.dpi)
        # plt.clf()
        #
        # plt.bar(np.arange(len(tot_reward_history)), tot_cost_history)
        # plt.show(block=False)
        # fig = plt.gcf()
        # fig.savefig('result/test_tot_cost_history.png', dpi=fig.dpi)
        # plt.clf()
        #
        # plt.bar(np.arange(len(tot_reward_history)), tot_soc_history)
        # plt.show(block=False)
        # fig = plt.gcf()
        # fig.savefig('result/test_tot_soc_history.png', dpi=fig.dpi)
        # plt.clf()
        #
        # plt.bar(np.arange(len(tot_reward_history)), tot_diff_soc_history)
        # plt.show(block=False)
        # fig = plt.gcf()
        # fig.savefig('result/test_tot_diff_soc_history.png', dpi=fig.dpi)
        # plt.clf()




if __name__ == "__main__":
    main()