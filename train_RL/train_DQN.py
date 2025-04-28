from envs.osmEnv import OSMGraphEnv
from envs.until_osm import load_graph, sample_start_goal
from agents.DQN import DQNAgent,DQNNetwork
import networkx as nx
import matplotlib.pyplot as plt
import os
import time
import torch 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def evaluate_agent(agent, env, episodes=10, max_steps=1000): 
    agent.model.eval()
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        path = [env.current_node]

        while not done and steps < max_steps:
            neighbors = list(env.graph.neighbors(env.current_node))
            valid_actions = list(range(len(neighbors)))
            action = agent.act(state, valid_actions, env, eval_mode=True)
            next_state, reward, done, info = env.step(action)

            state = next_state
            total_reward += reward
            path.append(env.current_node)
            steps += 1

        total_rewards.append(total_reward)
        print(f"Episode {episode+1}: Total reward: {total_reward:.2f}, Path length: {len(path)}")

    mean_reward = np.mean(total_rewards)
    print(f"Evaluation completed. Average reward: {mean_reward:.2f}")
    return mean_reward


def train_cycle(env, agent, episodes, max_steps, cycle_num):
    best_reward = float('-inf')
    best_path = None
    # batch_size = agent.batch_size
    rewards_history = []
    goal_reached_count = 0

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        path = [env.current_node]
        steps = 0
        done = False
        # agent.episode_count += 1

        while not done and steps < max_steps:
            neighbors = list(env.graph.neighbors(env.current_node))
            valid_actions = list(range(len(neighbors)))
            action = agent.act(state, valid_actions, env)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            # if len(agent.memory) >= batch_size:
            #     agent.train()

            state = next_state
            total_reward += reward
            path.append(env.current_node)
            steps += 1

        rewards_history.append(total_reward)

    #     if agent.epsilon > agent.epsilon_min:
    #         agent.epsilon *= agent.epsilon_decay

    #     if env.current_node == env.goal_node and total_reward > best_reward:
    #         best_reward = total_reward
    #         best_path = path.copy()
    #         print(f"Episode {episode+1}: Goal reached! Reward: {total_reward:.2f}, Steps: {steps}, Distance to goal: {info['dist_to_goal']:.2f}")

    #     if done and env.current_node != env.goal_node:
    #         print(f"Episode {episode+1}: Terminated early. Steps: {steps}, Reward: {total_reward:.2f}, Distance to goal: {info['dist_to_goal']:.2f}, Reason: {'Max steps' if steps >= max_steps else 'No neighbors'}")

    # return best_path, best_reward

            # Track convergence
        if env.current_node == env.goal_node:
            goal_reached_count += 1
            
            if total_reward > best_reward:
                best_reward = total_reward
                best_path = path.copy()
                
        # Early stopping if converged
        if episode > 100 and goal_reached_count > 10:
            avg_last_50 = np.mean(rewards_history[-50:])
            avg_prev_50 = np.mean(rewards_history[-100:-50])
            
            # If performance is stable or decreasing, stop early
            if avg_last_50 <= avg_prev_50 * 1.02:  # Less than 2% improvement
                print(f"Early stopping at episode {episode} - performance converged")
                break
                
    # Plot training curve
    plt.figure()
    plt.plot(rewards_history)
    plt.title(f"Training Rewards - Cycle {cycle_num}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(f"training_progress/reward_curve_cycle_{cycle_num}.png")
    plt.close()
    
    return best_path, best_reward, rewards_history

def save_path_plot(env, path, reward, title, filename):
    fig = plt.figure(figsize=(10, 8))
    env.render_path(path)
    plt.title(f"{title}\nReward: {reward:.2f}")
    plt.savefig(filename)
    plt.close(fig)

# def main():
#     os.makedirs("training_progress", exist_ok=True)
#     plt.ioff()

#     print("Loading OSM graph...")
#     G = load_graph()
#     if not G or not G.nodes:
#         print("Error: Graph is empty or failed to load!")
#         return

#     print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

#     training_cycles = 20
#     episodes_per_cycle = 200
#     max_steps_per_episode = 1000  # Increased from 500

#     overall_best_path = None
#     overall_best_reward = float('-inf')
#     overall_best_agent = None  # Lưu agent tốt nhất

#     for cycle in range(training_cycles):
#         print(f"\n=== Training Cycle {cycle+1}/{training_cycles} ===")
#         start, goal = sample_start_goal(G)
#         print(f"Start: {start}, Goal: {goal}")

#         try:
#             direct_path = nx.shortest_path(G, start, goal, weight='length')
#             print(f"Direct path length: {len(direct_path)} nodes")
#         except:
#             print("Could not calculate direct path length")

#         env = OSMGraphEnv(G, start, goal)
#         env.reward_scale = 50
#         env.goal_reward = 500  # Match step reward
#         env.step_penalty = 0.01

#         agent = DQNAgent(
#             state_size=8,
#             action_size=env.action_space.n,
#             lr=0.0001,
#             epsilon_start=1.0,
#             epsilon_end=0.01,
#             epsilon_decay=0.995
#         )

#         start_time = time.time()
#         best_path, best_reward = train_cycle(env, agent, episodes_per_cycle, max_steps_per_episode, cycle + 1)
#         elapsed_time = time.time() - start_time

#         print(f"Cycle {cycle+1} completed in {elapsed_time:.2f}s")
#         if best_path:
#             print(f"Best reward: {best_reward:.2f}, Path length: {len(best_path)}")
#             save_path_plot(
#                 env,
#                 best_path,
#                 best_reward,
#                 f"Best Path for Cycle {cycle+1}",
#                 os.path.join("training_progress", f"cycle_{cycle+1}_best_path.png")
#             )
#         else:
#             print("No path found to goal")

#         # So sánh reward để tìm agent tốt nhất
#         if best_path and best_reward > overall_best_reward:
#             overall_best_reward = best_reward
#             overall_best_path = best_path.copy()
#             overall_best_agent = agent  # Lưu agent tốt nhất

#         # Save agent after each training cycle
#         agent_save_path = os.path.join("training_progress", f"cycle_{cycle+1}_dqn_agent.pth")
#         torch.save(agent.model.state_dict(), agent_save_path)
#         print(f"Agent for Cycle {cycle+1} saved to {agent_save_path}")

#         # Evaluate the agent after the training cycle
#         print(f"Evaluating agent after cycle {cycle+1}...")
#         evaluate_agent(agent, env, episodes=10, max_steps=500)

#     # Sau khi huấn luyện xong, render path tốt nhất từ agent tốt nhất
#     if overall_best_path:
#         print("\nRendering final best path...")
#         save_path_plot(
#             env,
#             overall_best_path,
#             overall_best_reward,
#             "Best Path Found Overall",
#             os.path.join("training_progress", "final_best_path_1.png")
#         )
#         print(f"Final best reward: {overall_best_reward:.2f}, Path length: {len(overall_best_path)}")
#         print("Final best path saved to 'training_progress/final_best_path_1.png'")

#         # Đánh giá agent tốt nhất
#         print(f"Evaluating the overall best agent...")
#         evaluate_agent(overall_best_agent, env, episodes=10, max_steps=500)
#     else:
#         print("No successful paths found during training")

# if __name__ == "__main__":
#     main()
def federated_averaging(global_model, local_models, weights=None):
    """
    Weighted federated averaging, với weights là mức độ đóng góp của mỗi mô hình
    """
    if weights is None:
        weights = [1.0/len(local_models)] * len(local_models)
    
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        # Weighted average
        weighted_sum = torch.zeros_like(global_dict[key], dtype=torch.float32)
        for i, model in enumerate(local_models):
            weighted_sum += weights[i] * model.state_dict()[key].float()
        global_dict[key] = weighted_sum
    
    global_model.load_state_dict(global_dict)
    return global_model

def knowledge_distillation(global_model, local_models, env, episodes=100):
    """
    Knowledge distillation để chuyển kiến thức từ các mô hình cục bộ
    """
    teacher_ensemble = local_models
    student = global_model
    
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    for episode in range(episodes):
        state = env.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Teacher ensemble prediction
        with torch.no_grad():
            teacher_outputs = torch.stack([
                teacher(state_tensor) for teacher in teacher_ensemble
            ]).mean(dim=0)
        
        # Student prediction
        student_output = student(state_tensor)
        
        # Distillation loss
        loss = loss_fn(student_output, teacher_outputs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if episode % 10 == 0:
            print(f"Distillation episode {episode}, Loss: {loss.item():.4f}")
    
    return student

# Định nghĩa hàm save_path_plot
def save_path_plot(env, path, reward, title, filename):
    fig = plt.figure(figsize=(10, 8))
    env.render_path(path)
    plt.title(f"{title}\nReward: {reward:.2f}")
    plt.savefig(filename)
    plt.close(fig)

def main():
    os.makedirs("training_progress", exist_ok=True)
    plt.ioff()

    place_names = [
        'District 1, Ho Chi Minh City, Vietnam',
        'District 3, Ho Chi Minh City, Vietnam',
        'District 4, Ho Chi Minh City, Vietnam',
    ]
    training_cycles = len(place_names)
    episodes_per_cycle = 400
    max_steps_per_episode = 1000
    
    local_models = []
    local_performances = []
    
    # Train local models
    for cycle in range(training_cycles):
        print(f"\n=== Training Cycle {cycle+1}/{training_cycles} ===")
        G = load_graph(place_names[cycle])
        
        for pair_idx in range(3):  # Train on 3 different routes in each district
            start, goal = sample_start_goal(G)
            env = OSMGraphEnv(G, start, goal)
            
            agent = DQNAgent(
                state_size=8,
                action_size=env.action_space.n,
                lr=0.0005,
                epsilon_start=0.9,
                epsilon_end=0.05,
                epsilon_decay=0.995
            )
            
            train_cycle(env, agent, episodes_per_cycle, max_steps_per_episode, cycle + 1)
            
            eval_reward = evaluate_agent(agent, env, episodes=10)
            local_performances.append(eval_reward)
            local_models.append(agent.model)
    
    # Weight models by performance
    total_perf = sum(max(0, perf) for perf in local_performances) or 1
    weights = [max(0, perf)/total_perf for perf in local_performances]
    
    # Federated Averaging
    global_model = DQNNetwork(state_size=8, action_size=env.action_space.n)
    global_model = federated_averaging(global_model, local_models, weights)

    # Knowledge Distillation
    print("\n=== Starting Knowledge Distillation ===")
    G_distill = load_graph('District 5, Ho Chi Minh City, Vietnam')
    start, goal = sample_start_goal(G_distill)
    env_distill = OSMGraphEnv(G_distill, start, goal)
    global_model = knowledge_distillation(global_model, local_models, env_distill, episodes=100)

    # Finetune on a validation environment
    print("\n=== Starting Fine-tuning ===")
    G_finetune = load_graph('District 6, Ho Chi Minh City, Vietnam')
    start, goal = sample_start_goal(G_finetune)
    env_finetune = OSMGraphEnv(G_finetune, start, goal)

    finetune_agent = DQNAgent(
        state_size=8,
        action_size=env_finetune.action_space.n,
        lr=0.0001,
        epsilon_start=0.3,
        epsilon_end=0.01,
        epsilon_decay=0.99
    )
    
    finetune_agent.model.load_state_dict(global_model.state_dict())
    finetune_agent.update_target_model()
    
    train_cycle(env_finetune, finetune_agent, episodes=150, max_steps=1000, cycle_num=0)
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    for place_name in place_names:
        G_eval = load_graph(place_name)
        start, goal = sample_start_goal(G_eval)
        env_eval = OSMGraphEnv(G_eval, start, goal)
        evaluate_agent(finetune_agent, env_eval, episodes=20, max_steps=1000)

if __name__ == '__main__':
    main()
