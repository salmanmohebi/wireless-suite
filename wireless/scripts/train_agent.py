from os.path import join
import json
import datetime

import gym
import numpy as np
import matplotlib.pyplot as plt
from sacred import Experiment

from wireless.agents.random_agent import RandomAgent
from wireless.agents.round_robin_agent import *
from wireless.agents.proportional_fair import *
from wireless.agents.basic_dqn import *


MODEL_PATH = '../../models_tuesday/'

# Load agent parameters
with open('../../config/config_agent.json') as f:
    ac = json.load(f)

# Configure experiment
with open('../../config/config_sacred.json') as f:
    sc = json.load(f)   # Sacred Configuration
    ns = sc["sacred"]["n_metrics_points"]  # Number of points per episode to log in Sacred
    ex = Experiment(ac["agent"]["agent_type"])
    ex.add_config(sc)
    ex.add_config(ac)
mongo_db_url = f'mongodb://{sc["sacred"]["sacred_user"]}:{sc["sacred"]["sacred_pwd"]}@' +\
               f'{sc["sacred"]["sacred_host"]}:{sc["sacred"]["sacred_port"]}/{sc["sacred"]["sacred_db"]}'
# ex.observers.append(MongoObserver(url=mongo_db_url, db_name=sc["sacred"]["sacred_db"]))  # Uncomment to save to DB

# Load environment parameters
with open('../../config/config_environment.json') as f:
    ec = json.load(f)
    ex.add_config(ec)


@ex.automain
def main(_run):
    print(f'The program started at: {datetime.datetime.now()}')

    n_eps = _run.config["agent"]["n_episodes"]
    t_max = _run.config['agent']['t_max']
    rwd = np.zeros((n_eps, t_max))  # Memory allocation

    env = gym.make('TimeFreqResourceAllocation-v1', n_ues=_run.config['env']['n_ues'],
                   n_prbs=_run.config['env']['n_prbs'], buffer_max_size=_run.config['env']['buffer_max_size'],
                   eirp_dbm=_run.config['env']['eirp_dbm'], f_carrier_mhz=_run.config['env']['f_carrier_mhz'],
                   max_pkt_size_bits=_run.config['env']['max_pkt_size_bits'],
                   it=_run.config['env']['non_gbr_traffic_mean_interarrival_time_ttis'])  # Init environment


    agent = BasicDQNAgent(state_size=len(env.reset()), action_space=env.action_space)
    # Simulate
    total_steps = 0
    for ep in range(1, n_eps):  # Run episodes
        state = env.reset()
        for t in range(t_max):  # Run one episode
            total_steps += 1
            action = agent.epsilon_greedy(state)
            new_state, reward, done, _ = env.step(action)
            agent.experience_replay.store([state, action, reward, new_state, done])
            state = new_state
            if agent.experience_replay.size > 2*agent.batch_size:
                agent.train_network()

            if total_steps % agent.target_update_interval == 0:
                agent.update_epsilon()
                # agent.update_lr()
                agent.update_target_network()
            rwd[ep, t] = reward
            if done:
                break
        if ep % 10 == 0:
            print(f'Average reward in ep: {ep} : {np.mean(rwd[ep, :]):2f}, '
                  f'Average reward by now : {np.mean(rwd[:ep, :]):2f}, '
                  f'Learning rate: {agent.lr}, '
                  f'Epsilon: {agent.epsilon}'
                  )
            agent.target_network.save_weights(join(MODEL_PATH, f'model_{ep}.h5'))
            np.save(join(MODEL_PATH, f'reward.npy'), rwd)

            plt.plot(range(ep), np.mean(rwd, axis=1)[:ep])
            # plt.show()
            plt.savefig(join(MODEL_PATH, f'fig.png'))
            plt.clf()

    result = np.mean(rwd)  # Save experiment result

    print(f"Result: {result}")

    return result
