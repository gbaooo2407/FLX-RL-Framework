import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = []
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = 0.995
        self.target_update_freq = 10
        self.train_count = 0
        self.episode_count = 0

    def act(self, state, valid_actions, env=None, eval_mode=False):
        if not isinstance(state, np.ndarray) or state.shape != (self.state_size,):
            state = np.array(state, dtype=np.float32).reshape(self.state_size) if np.isscalar(state) else state
            print(f"Warning: State shape corrected to {state.shape}")

        # Giảm epsilon khi trong chế độ đánh giá (eval_mode)
        exploration_threshold = max(self.epsilon, 0.1 if eval_mode else 0.5)
        if random.random() < exploration_threshold and env is not None:
            # Heuristic: Choose action that minimizes distance to goal with 50% probability
            if random.random() < 0.5:
                neighbors = list(env.graph.neighbors(env.current_node))
                goal_pos = np.array([env.graph.nodes[env.goal_node]['x'], env.graph.nodes[env.goal_node]['y']])
                distances = [
                    np.linalg.norm(np.array([env.graph.nodes[neighbors[i]]['x'], env.graph.nodes[neighbors[i]]['y']]) - goal_pos)
                    for i in range(len(neighbors))
                ]
                return valid_actions[np.argmin(distances)] if distances else random.choice(valid_actions)
            return random.choice(valid_actions)

        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            q_values = self.model(state_tensor)[0]
            masked_q_values = torch.full((self.action_size,), float('-inf'))
            for action in valid_actions:
                masked_q_values[action] = q_values[action]
            chosen_action = torch.argmax(masked_q_values).item()
            return chosen_action

    # def act(self, state, valid_actions, env=None, eval_mode=False):
    #     # Kiểm tra và chuyển đổi state nếu cần
    #     if not isinstance(state, np.ndarray) or state.shape != (self.state_size,):
    #         state = np.array(state, dtype=np.float32).reshape(self.state_size) if np.isscalar(state) else state
    #         print(f"Warning: State shape corrected to {state.shape}")

    #     # Nếu đang trong chế độ đánh giá, chọn hành động tối ưu (greedy)
    #     if eval_mode:
    #         # Tính toán giá trị Q cho tất cả các hành động hợp lệ
    #         with torch.no_grad():
    #             state_tensor = torch.from_numpy(state).float().unsqueeze(0)
    #             q_values = self.model(state_tensor)[0]
    #             masked_q_values = torch.full((self.action_size,), float('-inf'))
    #             for action in valid_actions:
    #                 masked_q_values[action] = q_values[action]
    #             return torch.argmax(masked_q_values).item()

    #     # Nếu không trong chế độ đánh giá, sử dụng epsilon-greedy
    #     exploration_threshold = max(self.epsilon, 0.5 if self.episode_count < 10 else 0.0)
    #     if random.random() < exploration_threshold and env is not None:
    #         # Heuristic: Chọn hành động giảm thiểu khoảng cách tới mục tiêu với xác suất 50%
    #         if random.random() < 0.5:
    #             neighbors = list(env.graph.neighbors(env.current_node))
    #             goal_pos = np.array([env.graph.nodes[env.goal_node]['x'], env.graph.nodes[env.goal_node]['y']])
    #             distances = [
    #                 np.linalg.norm(np.array([env.graph.nodes[neighbors[i]]['x'], env.graph.nodes[neighbors[i]]['y']]) - goal_pos)
    #                 for i in range(len(neighbors))
    #             ]
    #             return valid_actions[np.argmin(distances)] if distances else random.choice(valid_actions)
    #         return random.choice(valid_actions)

        # Chọn hành động có giá trị Q cao nhất (greedy action)
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            q_values = self.model(state_tensor)[0]
            masked_q_values = torch.full((self.action_size,), float('-inf'))
            for action in valid_actions:
                masked_q_values[action] = q_values[action]
            return torch.argmax(masked_q_values).item()


    def remember(self, s, a, r, s_, done):
        s = np.array(s, dtype=np.float32).reshape(self.state_size) if not isinstance(s, np.ndarray) or s.shape != (self.state_size,) else s
        s_ = np.array(s_, dtype=np.float32).reshape(self.state_size) if not isinstance(s_, np.ndarray) or s_.shape != (self.state_size,) else s_
        expected_shape = (self.state_size,)
        if s.shape != expected_shape or s_.shape != expected_shape:
            print(f"Warning: Invalid state shape. Got s: {s.shape}, s_: {s_.shape}, expected: {expected_shape}")
            return
        self.memory.append((s, a, r, s_, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([t[0] for t in minibatch], dtype=np.float32)
        actions = torch.tensor([t[1] for t in minibatch], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor([t[2] for t in minibatch], dtype=torch.float32).unsqueeze(1)
        next_states = np.array([t[3] for t in minibatch], dtype=np.float32)
        dones = torch.tensor([t[4] for t in minibatch], dtype=torch.float32).unsqueeze(1)
        states = torch.from_numpy(states).float()
        next_states = torch.from_numpy(next_states).float()
        current_q = self.model(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.train_count += 1
        if self.train_count % self.target_update_freq == 0:
            self.update_target_model()
    
    def _build_model(self):
        # Create a deeper network for better learning
        model = DQNNetwork(self.state_size, self.action_size)
        return model
    
    def update_target_model(self):
        # Copy weights from main model to target model
        self.target_model.load_state_dict(self.model.state_dict())
    

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        # Deeper network with more capacity
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
        # Better initialization for more stable learning
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)