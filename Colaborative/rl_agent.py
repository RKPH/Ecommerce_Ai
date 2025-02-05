import numpy as np
import random
from collections import deque
import tensorflow as tf
import os


class DQNAgent:
    def __init__(
            self,
            action_space,
            model_file="dqn_model.h5",
            memory_size=2000,
            batch_size=32,
            gamma=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            learning_rate=0.001
    ):
        """
        DQN Agent với state 4 chiều: (user_id, product_id, brand_idx, category_idx)
        action_space: danh sách (numpy array) các product_id khả dĩ mà agent có thể chọn.
        """
        self.action_space = np.array(action_space, dtype=np.int32)

        # Cập nhật kích thước state = 4
        self.state_size = 4  # (user_id, product_id, brand_idx, category_idx)
        self.action_size = len(self.action_space)

        # Replay memory
        self.memory = deque(maxlen=memory_size)

        # DQN hyperparams
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate

        self.model_file = model_file

        # Build or load the model
        self.model = self.build_model()

    def build_model(self):
        """
        Xây dựng mô hình DQN với input shape = (4,).
        """
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.state_size,)),  # (4,)
            tf.keras.layers.Dense(64, activation='tanh', name="hidden_layer_1"),
            tf.keras.layers.Dense(32, activation='tanh', name="hidden_layer_2"),
            tf.keras.layers.Dense(self.action_size, activation='linear', name="output_layer")
        ])
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model

    def store_experience(self, state, action, reward, next_state):
        """
        Lưu (state, action, reward, next_state) vào replay buffer.
        state và next_state dạng (4,) numpy array.
        """
        self.memory.append((state, action, reward, next_state))

    def train(self):
        """
        Lấy batch ngẫu nhiên từ replay buffer và huấn luyện Q-network.
        """
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = []
        targets = []

        for state, action, reward, next_state in minibatch:
            # Chuyển state thành shape (1,4)
            state_arr = np.array([state], dtype=np.float32)

            # Q hiện tại
            current_qs = self.model.predict(state_arr, verbose=0)[0]

            # Index của action trong action_space
            action_idx = np.where(self.action_space == action)[0][0]

            # Tính target
            if next_state is None:
                # Trường hợp terminal state
                target_q = reward
            else:
                next_state_arr = np.array([next_state], dtype=np.float32)
                future_qs = self.model.predict(next_state_arr, verbose=0)[0]
                target_q = reward + self.gamma * np.max(future_qs)

            updated_qs = current_qs.copy()
            updated_qs[action_idx] = target_q

            states.append(state)
            targets.append(updated_qs)

        states_np = np.array(states, dtype=np.float32)  # shape = (batch_size, 4)
        targets_np = np.array(targets, dtype=np.float32)  # shape = (batch_size, action_size)

        # Train
        self.model.fit(states_np, targets_np, epochs=1, verbose=0)

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def recommend(self, state, top_n=10, candidate_actions=None):
        """
        Dựa trên Q-values, trả về top-N actions (product_id) cho state cho trước.
        state: (4,)
        candidate_actions: danh sách product_id đang cân nhắc, nếu None => dùng toàn bộ action_space.
        """
        if candidate_actions is None or len(candidate_actions) == 0:
            candidate_actions = self.action_space

        state_arr = np.array([state], dtype=np.float32)  # shape=(1,4)
        q_values = self.model.predict(state_arr, verbose=0)[0]  # shape=(action_size,)

        scores = []
        for c in candidate_actions:
            # Tìm index của c
            c_idx_list = np.where(self.action_space == c)[0]
            if len(c_idx_list) > 0:
                c_idx = c_idx_list[0]
                scores.append((c, q_values[c_idx]))
            else:
                # Nếu c không nằm trong action_space, cho điểm -inf
                scores.append((c, float('-inf')))

        # Sắp xếp theo Q-value giảm dần
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scores[:top_n]]

    def save_model(self):
        """
        Lưu model
        """
        self.model.save(self.model_file)

    def load_model(self):
        """
        Load model
        """
        self.model = tf.keras.models.load_model(self.model_file)
        print(f"Model loaded from {self.model_file}")