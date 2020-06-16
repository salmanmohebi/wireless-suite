"""
© 2020 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause
"""
import gym
import numpy as np
import json

from sacred import Experiment
from sacred.observers import MongoObserver

from wireless.agents.random_agent import RandomAgent
from wireless.agents.round_robin_agent import *
from wireless.agents.proportional_fair import *
from wireless.agents.basic_dqn import *

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
    n_eps = _run.config["agent"]["n_episodes"]
    t_max = _run.config['agent']['t_max']
    n_sf = t_max//_run.config['env']['n_prbs']  # Number of complete subframes to run per episode
    log_period_t = max(1, (n_sf//ns)*_run.config['env']['n_prbs'])  # Only log rwd on last step of each subframe

    rwd = np.zeros((n_eps, t_max))  # Memory allocation

    env = gym.make('TimeFreqResourceAllocation-v0', n_ues=_run.config['env']['n_ues'],
                   n_prbs=_run.config['env']['n_prbs'], buffer_max_size=_run.config['env']['buffer_max_size'],
                   eirp_dbm=_run.config['env']['eirp_dbm'], f_carrier_mhz=_run.config['env']['f_carrier_mhz'],
                   max_pkt_size_bits=_run.config['env']['max_pkt_size_bits'],
                   it=_run.config['env']['non_gbr_traffic_mean_interarrival_time_ttis'])  # Init environment
    # env.seed(seed=_run.config['seed'] + ep)

    q_network = MyModel(env.action_space.n)
    target_network = MyModel(env.action_space.n)
    agent = BasicDQNAgent(q_network, target_network, env.action_space)
    # Simulate
    for ep in range(n_eps):  # Run episodes
        total_steps = 0
        state = env.reset()
        for t in range(t_max):  # Run one episode
            total_steps += 1
            # Collect progress
            if t_max < ns or (t > 0 and (t+1) % log_period_t == 0):  # If it's time to log
                s = np.reshape(state[env.K:env.K * (1 + env.L)], (env.K, env.L))
                qi_ohe = np.reshape(state[env.K+2*env.K*env.L:5*env.K + 2*env.K*env.L], (env.K, 4))
                qi = [np.where(r == 1)[0][0] for r in qi_ohe]  # Decode One-Hot-Encoded QIs
                for u in range(0, env.K, env.K//2):  # Log KPIs for some UEs
                    _run.log_scalar(f"Episode {ep}. UE {u}. CQI vs time step", state[u], t)
                    _run.log_scalar(f"Episode {ep}. UE {u}. Buffer occupancy [bits] vs time step", np.sum(s[u, :]), t)
                    _run.log_scalar(f"Episode {ep}. UE {u}. QoS Identifier vs time step", qi[u], t)

            action = agent.epsilon_greedy(state)
            new_state, reward, done, _ = env.step(action)
            agent.experience_replay.store([state, action, reward, new_state, done])
            state = new_state
            if agent.experience_replay.size > agent.batch_size:
                agent.train_network()
            if total_steps % agent.target_update_interval == 0:
                agent.update_target_network()
            # Collect progress
            if t_max < ns or (t > 0 and (t+1) % log_period_t == 0):
                _run.log_scalar(f"Episode {ep}. Rwd vs time step", reward, t)

            rwd[ep, t] = reward
            if done:
                break
        print(f' average reward in ep: {ep} : {np.mean(rwd[ep, :]):2f}')


    if n_eps > 1:
        rwd_avg = np.mean(rwd, axis=0)
        for t in range(t_max):
            if t_max < ns or (t > 0 and (t+1) % log_period_t == 0):  # If it's time to log
                _run.log_scalar(f"Mean rwd vs time step", rwd_avg[t], t)

    result = np.mean(rwd)  # Save experiment result
    print(f"Result: {result}")
    return result
