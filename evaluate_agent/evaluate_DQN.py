import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from envs.osmEnv import OSMGraphEnv
from envs.until_osm import load_graph, sample_start_goal
from agents.DQN import DQNAgent
import networkx as nx

def evaluate_agent(agent_path, episodes=20, max_steps=1000):
    # Load graph
    G = load_graph()
    if not G or not G.nodes:
        print("Error loading graph!")
        return

    print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

    success_count = 0
    rewards = []
    steps_list = []
    paths = []

    best_reward = float('-inf')
    best_path = None
    env_best = None

    for episode in range(episodes):
        start, goal = sample_start_goal(G)
        env = OSMGraphEnv(G, start, goal)
        env.reward_scale = 50
        env.goal_reward = 500
        env.step_penalty = 0.01

        agent = DQNAgent(state_size=8, action_size=env.action_space.n)
        agent.model.load_state_dict(torch.load(agent_path))
        agent.model.eval()

        state = env.reset()
        path = [env.current_node]
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            neighbors = list(env.graph.neighbors(env.current_node))
            valid_actions = list(range(len(neighbors)))
            action = agent.act(state, valid_actions, env, eval_mode=True)  # Greedy
            next_state, reward, done, info = env.step(action)

            state = next_state
            total_reward += reward
            path.append(env.current_node)
            steps += 1

        rewards.append(total_reward)
        steps_list.append(steps)
        paths.append(path)

        if env.current_node == env.goal_node:
            success_count += 1

        if total_reward > best_reward and env.current_node == env.goal_node:
            best_reward = total_reward
            best_path = path.copy()
            env_best = env

        print(f"Episode {episode+1}: {'Success!' if env.current_node == env.goal_node else 'Fail.'} Total reward: {total_reward:.2f}, Steps: {steps}")

    # Results
    success_rate = success_count / episodes * 100
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps_list)

    print("\n=== Evaluation Results ===")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")

    # Save histogram
    os.makedirs("evaluation_results", exist_ok=True)
    plt.hist(rewards, bins=10, color='skyblue', edgecolor='black')
    plt.title("Reward Distribution")
    plt.xlabel("Total Reward")
    plt.ylabel("Frequency")
    plt.savefig("evaluation_results/reward_histogram.png")
    plt.close()

    # Save best path plot
    if best_path and env_best:
        env_best.render_path(best_path)
        plt.title(f"Best Path (Reward: {best_reward:.2f})")
        plt.savefig("evaluation_results/best_path.png")
        plt.close()

    # Save report
    with open("evaluation_results/evaluation_report.txt", "w") as f:
        f.write(f"Success Rate: {success_rate:.2f}%\n")
        f.write(f"Average Reward: {avg_reward:.2f}\n")
        f.write(f"Average Steps: {avg_steps:.2f}\n")

if __name__ == "__main__":
    # Ví dụ file agent
    evaluate_agent("training_progress/cycle_5_dqn_agent.pth", episodes=20)
