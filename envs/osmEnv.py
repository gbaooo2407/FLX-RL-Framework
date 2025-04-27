import gym
import networkx as nx
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np

class OSMGraphEnv(gym.Env):
    def __init__(self, graph, start_node, goal_node):
        super(OSMGraphEnv, self).__init__()
        self.graph = graph
        self.start_node = start_node
        self.goal_node = goal_node
        self.current_node = start_node
        self.max_steps = 1000  # Increased from 500
        self.reward_scale = 50
        self.goal_reward = 500  # Increased from 200
        self.step_penalty = 0.01  # Reduced from 0.1
        self.max_step_penalty = 10
        self.observation_space = spaces.Discrete(len(self.graph.nodes))
        self.action_space = spaces.Discrete(10)
        # Calculate max distance for normalization
        x_coords = [self.graph.nodes[n]['x'] for n in self.graph.nodes]
        y_coords = [self.graph.nodes[n]['y'] for n in self.graph.nodes]
        self.max_distance = np.sqrt((max(x_coords) - min(x_coords))**2 + (max(y_coords) - min(y_coords))**2) or 1.0

    def step(self, action):
        neighbors = list(self.graph.neighbors(self.current_node))
        if len(neighbors) == 0:
            return self.get_state(), -10, True, {}

        action = action % len(neighbors)
        next_node = neighbors[action]

        # Reward calculation
        current_pos = np.array([self.graph.nodes[self.current_node]['x'], self.graph.nodes[self.current_node]['y']])
        next_pos = np.array([self.graph.nodes[next_node]['x'], self.graph.nodes[next_node]['y']])
        goal_pos = np.array([self.graph.nodes[self.goal_node]['x'], self.graph.nodes[self.goal_node]['y']])

        dist_current_goal = np.linalg.norm(current_pos - goal_pos)
        dist_next_goal = np.linalg.norm(next_pos - goal_pos)

        # Normalized rewards
        reward = (dist_current_goal - dist_next_goal) / self.max_distance * self.reward_scale  # Distance reduction
        reward -= (dist_next_goal / self.max_distance) * 10  # Stronger potential-based penalty
        reward -= self.step_penalty  # Small step penalty

        self.current_node = next_node
        self.steps += 1

        done = False
        if self.current_node == self.goal_node:
            reward += self.goal_reward
            done = True
        elif self.steps >= self.max_steps:
            reward -= self.max_step_penalty
            done = True

        return self.get_state(), reward, done, {'dist_to_goal': dist_next_goal}

    def get_state(self):
        current = self.graph.nodes[self.current_node]
        goal = self.graph.nodes[self.goal_node]
        current_x = float(current.get('x', 0.0))
        current_y = float(current.get('y', 0.0))
        goal_x = float(goal.get('x', 0.0))
        goal_y = float(goal.get('y', 0.0))
        dx = goal_x - current_x
        dy = goal_y - current_y
        angle = np.arctan2(dy, dx)
        distance = np.sqrt(dx*dx + dy*dy) / self.max_distance  # Normalize distance
        state = np.array([
            current_x / self.max_distance,  # Normalize coordinates
            current_y / self.max_distance,
            goal_x / self.max_distance,
            goal_y / self.max_distance,
            np.cos(angle),
            np.sin(angle),
            distance,
            self.steps / self.max_steps
        ], dtype=np.float32)
        if not np.all(np.isfinite(state)):
            print(f"Warning: Invalid state detected: {state}")
            return np.zeros(8, dtype=np.float32)
        return state

    def reset(self):
        self.current_node = self.start_node
        self.steps = 0
        return self.get_state()

    def render_path(self, path):
        """Render the path with debugging information to diagnose blank figures"""
        # First, verify we have a valid path
        if not path or len(path) == 0:
            print("Warning: Empty path provided to render_path")
            return
            
        print(f"Rendering path with {len(path)} nodes")
        print(f"First node: {path[0]}, Last node: {path[-1]}")
        
        # Check if all nodes exist in the graph
        invalid_nodes = [node for node in path if node not in self.graph.nodes]
        if invalid_nodes:
            print(f"Warning: Path contains {len(invalid_nodes)} nodes not in graph: {invalid_nodes[:5]}...")
            # Filter out invalid nodes
            path = [node for node in path if node in self.graph.nodes]
            if not path:
                print("No valid nodes to render!")
                return
        
        # Extract positions
        pos = {}
        missing_coords = []
        for node in self.graph.nodes:
            try:
                x = self.graph.nodes[node].get('x')
                y = self.graph.nodes[node].get('y')
                if x is None or y is None:
                    missing_coords.append(node)
                    continue
                pos[node] = (x, y)
            except Exception as e:
                print(f"Error getting coordinates for node {node}: {e}")
        
        if missing_coords:
            print(f"Warning: {len(missing_coords)} nodes are missing coordinates")
        
        if not pos:
            print("Error: No valid node positions found in graph!")
            return
            
        # Create figure and clear it
        plt.figure(figsize=(10, 8))
        plt.clf()
        
        # Draw the graph
        print("Drawing base graph...")
        nx.draw(self.graph, pos, node_size=10, node_color='gray', edge_color='lightgray', alpha=0.5)
        
        # Draw path if it has multiple nodes
        if len(path) > 1:
            print(f"Drawing path with {len(path)-1} edges...")
            path_edges = list(zip(path[:-1], path[1:]))
            valid_edges = [(u, v) for u, v in path_edges if u in pos and v in pos]
            print(f"Valid path edges: {len(valid_edges)}/{len(path_edges)}")
            
            if valid_edges:
                nx.draw_networkx_edges(self.graph, pos, edgelist=valid_edges, 
                                    edge_color='red', width=2)
        
        # Draw start and end nodes
        start_node = path[0] if path else None
        end_node = path[-1] if path else None
        
        if start_node in pos:
            print("Drawing start node...")
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[start_node], 
                                node_color='green', node_size=100)
        
        if end_node in pos and end_node != start_node:
            print("Drawing end node...")
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[end_node], 
                                node_color='blue', node_size=100)
        
        plt.title("Agent Path")
        plt.axis('off')
        print("Rendering complete!")

    def render_step(self, current_node, path_so_far):
        plt.clf()
        pos = {node: (self.graph.nodes[node]['x'], self.graph.nodes[node]['y']) for node in self.graph.nodes}

        nx.draw(self.graph, pos, node_size=10, node_color='gray', edge_color='lightgray')

        if len(path_so_far) > 1:
            path_edges = list(zip(path_so_far[:-1], path_so_far[1:]))
            nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color='red', width=2)

        nx.draw_networkx_nodes(self.graph, pos, nodelist=[path_so_far[0]], node_color='green', node_size=100)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=[current_node], node_color='blue', node_size=100)

        plt.title(f"Agent Moving - Current: {current_node}")
        plt.axis('off')
        plt.pause(0.001) 