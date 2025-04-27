from envs.osmEnv import OSMGraphEnv
from envs.until_osm import load_graph, sample_start_goal
from agents.DQN import DQNAgent
import networkx as nx
import matplotlib.pyplot as plt
import os
import time
import torch 
import numpy as np

def evaluate_agent(agent, env, episodes=10, max_steps=300):  # Giảm max_steps xuống
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
    batch_size = agent.batch_size

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        path = [env.current_node]
        steps = 0
        done = False
        agent.episode_count += 1

        while not done and steps < max_steps:
            neighbors = list(env.graph.neighbors(env.current_node))
            valid_actions = list(range(len(neighbors)))
            action = agent.act(state, valid_actions, env)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) >= batch_size:
                agent.train()

            state = next_state
            total_reward += reward
            path.append(env.current_node)
            steps += 1

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if env.current_node == env.goal_node and total_reward > best_reward:
            best_reward = total_reward
            best_path = path.copy()
            print(f"Episode {episode+1}: Goal reached! Reward: {total_reward:.2f}, Steps: {steps}, Distance to goal: {info['dist_to_goal']:.2f}")

        if done and env.current_node != env.goal_node:
            print(f"Episode {episode+1}: Terminated early. Steps: {steps}, Reward: {total_reward:.2f}, Distance to goal: {info['dist_to_goal']:.2f}, Reason: {'Max steps' if steps >= max_steps else 'No neighbors'}")

    return best_path, best_reward

def save_path_plot(env, path, reward, title, filename):
    fig = plt.figure(figsize=(10, 8))
    env.render_path(path)
    plt.title(f"{title}\nReward: {reward:.2f}")
    plt.savefig(filename)
    plt.close(fig)

def main():
    os.makedirs("training_progress", exist_ok=True)
    plt.ioff()

    print("Loading OSM graph...")
    G = load_graph()
    if not G or not G.nodes:
        print("Error: Graph is empty or failed to load!")
        return

    print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

    training_cycles = 20
    episodes_per_cycle = 200
    max_steps_per_episode = 1000  # Increased from 500

    overall_best_path = None
    overall_best_reward = float('-inf')
    overall_best_agent = None  # Lưu agent tốt nhất

    for cycle in range(training_cycles):
        print(f"\n=== Training Cycle {cycle+1}/{training_cycles} ===")
        start, goal = sample_start_goal(G)
        print(f"Start: {start}, Goal: {goal}")

        try:
            direct_path = nx.shortest_path(G, start, goal, weight='length')
            print(f"Direct path length: {len(direct_path)} nodes")
        except:
            print("Could not calculate direct path length")

        env = OSMGraphEnv(G, start, goal)
        env.reward_scale = 50
        env.goal_reward = 500  # Match step reward
        env.step_penalty = 0.01

        agent = DQNAgent(
            state_size=8,
            action_size=env.action_space.n,
            lr=0.0001,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995
        )

        start_time = time.time()
        best_path, best_reward = train_cycle(env, agent, episodes_per_cycle, max_steps_per_episode, cycle + 1)
        elapsed_time = time.time() - start_time

        print(f"Cycle {cycle+1} completed in {elapsed_time:.2f}s")
        if best_path:
            print(f"Best reward: {best_reward:.2f}, Path length: {len(best_path)}")
            save_path_plot(
                env,
                best_path,
                best_reward,
                f"Best Path for Cycle {cycle+1}",
                os.path.join("training_progress", f"cycle_{cycle+1}_best_path.png")
            )
        else:
            print("No path found to goal")

        # So sánh reward để tìm agent tốt nhất
        if best_path and best_reward > overall_best_reward:
            overall_best_reward = best_reward
            overall_best_path = best_path.copy()
            overall_best_agent = agent  # Lưu agent tốt nhất

        # Save agent after each training cycle
        agent_save_path = os.path.join("training_progress", f"cycle_{cycle+1}_dqn_agent.pth")
        torch.save(agent.model.state_dict(), agent_save_path)
        print(f"Agent for Cycle {cycle+1} saved to {agent_save_path}")

        # Evaluate the agent after the training cycle
        print(f"Evaluating agent after cycle {cycle+1}...")
        evaluate_agent(agent, env, episodes=10, max_steps=500)

    # Sau khi huấn luyện xong, render path tốt nhất từ agent tốt nhất
    if overall_best_path:
        print("\nRendering final best path...")
        save_path_plot(
            env,
            overall_best_path,
            overall_best_reward,
            "Best Path Found Overall",
            os.path.join("training_progress", "final_best_path_1.png")
        )
        print(f"Final best reward: {overall_best_reward:.2f}, Path length: {len(overall_best_path)}")
        print("Final best path saved to 'training_progress/final_best_path_1.png'")

        # Đánh giá agent tốt nhất
        print(f"Evaluating the overall best agent...")
        evaluate_agent(overall_best_agent, env, episodes=10, max_steps=500)
    else:
        print("No successful paths found during training")

if __name__ == "__main__":
    main()