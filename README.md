# Deep Reinforcement Learning Algorithms Implementation

This repository contains implementations of several popular deep reinforcement learning algorithms and their evaluations on continuous control tasks from the OpenAI Gym/Gymnasium environments.

## Project Overview

This project implements and compares four state-of-the-art deep reinforcement learning algorithms:

1. **PPO (Proximal Policy Optimization)** - Implementation for both:
   - Implementation tested on Pendulum-v1 and MountainCarContinuous-v0 environments

2. **TD3 (Twin Delayed Deep Deterministic Policy Gradient)**
   - Implementation tested on Pendulum-v1

3. **SAC (Soft Actor-Critic)**
   - Implementation tested on Pendulum-v1 and MountainCarContinuous-v0
   
4. **DQN (Deep Q-Network)**
   - Implementation tested on MountainCar-v0, Acrobot-v0, Cartpole-v0

Each algorithm is implemented from scratch and compared with the reference implementations from Stable-Baselines3 to validate correctness and performance.

## Environments

The algorithms are tested on the following OpenAI Gym/Gymnasium environments:

- **Pendulum-v1**: A classic control problem where the goal is to balance an inverted pendulum.
- **MountainCarContinuous-v0**: A continuous control problem where an underpowered car must drive up a steep mountain.
- **CartPole-v1**: A classic control task where the agent must balance a pole on a moving cart by applying left or right forces.
- **Acrobot-v1**: A two-link pendulum environment where the goal is to swing the lower link high enough to reach a target height.
- **MountainCar-v0**: A discrete control environment where a car must build momentum by moving back and forth to reach the top of a hill.

## Implementation Details

### PPO (Proximal Policy Optimization)

The PPO implementation uses:
- Actor-Critic architecture with shared network features
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective function
- Entropy bonus for exploration
- Mini-batch updates for better stability

Notable features:
- Rendering and video recording capabilities
- Training progress tracking and visualization
- Model saving and loading functionality

### TD3 (Twin Delayed Deep Deterministic Policy Gradient)

The TD3 implementation includes:
- Clipped double Q-learning to reduce Q-value overestimation
- Delayed policy updates
- Target policy smoothing with noise
- Experience replay buffer for off-policy learning

### SAC (Soft Actor-Critic)

The SAC implementation features:
- Entropy maximization for improved exploration
- Automatic temperature adjustment
- Dual Q-networks to mitigate overestimation bias
- Tanh-Gaussian policy with Jacobian correction for bounded actions

### DQN (Deep Q-Network)

The DQN implementation includes:
- Value-based learning with a neural network approximating the Q-function
- Epsilon-greedy exploration strategy for balancing exploration and exploitation
- Experience replay buffer to break correlation between samples
- Target network to stabilize training and reduce oscillations

## Project Structure

- `PPO_Pendulum-v1.ipynb`: PPO implementation for the Pendulum environment
- `PPO_MountainCarContinuous-v0.ipynb`: PPO implementation for the MountainCar environment
- `TD3.ipynb`: TD3 implementation with comparison to SB3
- `compare_SAC.ipynb`: SAC implementation with comparison to SB3
- `ppo_pendulum.pth`: Saved model weights for the trained PPO agent on Pendulum
- `ppo_training_rewards_pendulum.csv`: Training rewards data for PPO on Pendulum

## Key Features

- **Reproducibility**: Seed fixing for all sources of randomness
- **Visualization**: Tools for plotting training progress and rendering agent behavior
- **Comparative Analysis**: Direct comparison with Stable-Baselines3 implementations
- **Video Recording**: Functionality to record videos of trained policies
- **Hyperparameter Tuning**: Different configurations for each environment

## Usage

Each notebook is self-contained and can be run independently. The notebooks include:
1. Environment setup
2. Model definition
3. Training loop
4. Evaluation and visualization
5. Optional model saving and video recording


## References

- PPO: Schulman et al. "[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)"
- SAC: Haarnoja et al. "[Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)"
- TD3: Fujimoto et al. "[Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)"
- DQN: Mnih et al. "[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)"

## Acknowledgements

The PPO implementation for continuous environments was adapted from [RL-Adventure-2](https://github.com/henanmemeda/RL-Adventure-2/blob/master/3.ppo.ipynb) and modified to work with newer versions of the libraries and the MountainCarContinuous environment. Additional functionality was added to save training data and record demonstration videos.
