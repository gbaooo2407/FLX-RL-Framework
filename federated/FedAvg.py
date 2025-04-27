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
        model = DQNNetwork(state_size=8, action_size=env.action_space.n)  # action_size theo môi trường
        model.load_state_dict(torch.load(file_path))
        model.eval()  # Chế độ đánh giá
        models.append(model)
    return models

def federated_averaging(global_model, local_models):
    # Lấy trọng số của mô hình toàn cục
    global_dict = global_model.state_dict()

    # Lấy số lượng mô hình (clients)
    num_clients = len(local_models)

    # Tính trung bình trọng số của các mô hình client
    for key in global_dict:
        global_dict[key] = torch.stack([local_model.state_dict()[key] for local_model in local_models], dim=0).mean(dim=0)

    # Cập nhật trọng số của mô hình toàn cục
    global_model.load_state_dict(global_dict)
    return global_model

def evaluate_global_model(global_model, env, episodes=10, max_steps=300):
    global_model.eval()  # Đảm bảo mô hình ở chế độ đánh giá
    total_rewards = []

    # Khởi tạo DQNAgent với mô hình toàn cục
    agent = DQNAgent(state_size=8, action_size=env.action_space.n)  # action_size theo môi trường
    agent.model = global_model  # Gán mô hình toàn cục cho agent

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.act(state, valid_actions=list(range(env.action_space.n)), eval_mode=True)  # Chọn hành động từ agent
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1

        total_rewards.append(total_reward)
        print(f"Episode {episode+1}: Total reward: {total_reward:.2f}")

    mean_reward = np.mean(total_rewards)
    print(f"Evaluation completed. Average reward: {mean_reward:.2f}")
    return mean_reward

def main():
    # Đường dẫn tới các file .pth của các client
    file_paths = [f"D:/Framework FLX-RL/training_progress/cycle_{i}_dqn_agent.pth" for i in range(1,21)]
    
    # Tải mô hình từ các file
    local_models = load_models_from_files(file_paths)
    
    # Khởi tạo mô hình toàn cục
    G = load_graph()
    start, goal = sample_start_goal(G)
    env = OSMGraphEnv(G, start, goal)
    
    global_model = DQNNetwork(state_size=8, action_size=env.action_space.n)  # action_size theo môi trường
    
    # Áp dụng Federated Averaging để hợp nhất mô hình
    global_model = federated_averaging(global_model, local_models)
    
    # Đánh giá mô hình toàn cục
    evaluate_global_model(global_model, env)  # env là môi trường của bạn

if __name__ == "__main__":
    main()
