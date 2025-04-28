import torch
from envs.osmEnv import OSMGraphEnv
from envs.until_osm import load_graph, sample_start_goal
from agents.DQN import DQNAgent, DQNNetwork
import numpy as np
import os

# Tải các mô hình DQN từ các file .pth của các client
def load_models_from_files(file_paths):
    models = []
    G = load_graph()
    start, goal = sample_start_goal(G)
    env = OSMGraphEnv(G, start, goal)
    for file_path in file_paths:
        model = DQNNetwork(state_size=8, action_size=env.action_space.n)
        model.load_state_dict(torch.load(file_path))
        model.eval()
        models.append(model)
    return models

# Hàm thực hiện Federated Averaging
def federated_averaging(global_model, local_models):
    global_dict = global_model.state_dict()
    num_clients = len(local_models)
    for key in global_dict:
        global_dict[key] = torch.stack([local_model.state_dict()[key] for local_model in local_models], dim=0).mean(dim=0)
    global_model.load_state_dict(global_dict)
    return global_model

# Huấn luyện bổ sung mô hình toàn cục
def train_global_model(global_model, env, episodes=50, max_steps=300):
    agent = DQNAgent(state_size=8, action_size=env.action_space.n)
    agent.model = global_model
    # Khởi tạo target model để huấn luyện DQN
    agent.target_model = DQNNetwork(state_size=8, action_size=env.action_space.n)
    agent.update_target_model()

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.act(state, valid_actions=list(range(env.action_space.n)), eval_mode=False)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

            # Huấn luyện nếu bộ nhớ đủ lớn
            if len(agent.memory) >= agent.batch_size:
                agent.train()

        print(f"Training Episode {episode+1}: Total reward: {total_reward:.2f}")

    return global_model

# Đánh giá mô hình toàn cục trên nhiều môi trường
def evaluate_global_model(global_model, env, num_environments=5, episodes_per_env=10, max_steps=300):
    global_model.eval()
    total_rewards = []
    agent = DQNAgent(state_size=8, action_size=env.action_space.n)
    agent.model = global_model
    G = load_graph()

    # Lặp qua nhiều môi trường
    for env_idx in range(num_environments):
        start, goal = sample_start_goal(G)
        env = OSMGraphEnv(G, start, goal)
        env_rewards = []

        for episode in range(episodes_per_env):
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done and steps < max_steps:
                action = agent.act(state, valid_actions=list(range(env.action_space.n)), eval_mode=True)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward
                steps += 1

            env_rewards.append(total_reward)
            print(f"Env {env_idx+1}, Episode {episode+1}: Total reward: {total_reward:.2f}")

        total_rewards.extend(env_rewards)

    mean_reward = np.mean(total_rewards)
    print(f"Evaluation completed. Average reward across {num_environments} environments: {mean_reward:.2f}")
    return mean_reward

def main():
    # Đường dẫn tới các file .pth của các client
    file_paths = [f"D:/Framework FLX-RL/training_progress/cycle_{i}_dqn_agent.pth" for i in range(1, 21)]
    
    # Tải mô hình từ các file
    local_models = load_models_from_files(file_paths)
    
    # Khởi tạo mô hình toàn cục
    G = load_graph()
    start, goal = sample_start_goal(G)
    env = OSMGraphEnv(G, start, goal)
    global_model = DQNNetwork(state_size=8, action_size=env.action_space.n)
    
    # Áp dụng Federated Averaging để hợp nhất mô hình
    global_model = federated_averaging(global_model, local_models)
    
    # Huấn luyện bổ sung mô hình toàn cục
    print("Training global model...")
    global_model = train_global_model(global_model, env, episodes=50, max_steps=300)
    
    # Đánh giá mô hình toàn cục trên nhiều môi trường
    print("Evaluating global model...")
    evaluate_global_model(global_model, env, num_environments=5)

if __name__ == "__main__":
    main()