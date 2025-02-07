import numpy as np
import random
from collections import deque
import tensorflow as tf
import os
import logging


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
        A DQN agent that uses a two-dimensional state (user_id, item_id).
        """
        self.action_space = np.array(action_space, dtype=np.int32)
        self.state_size = 2  # Two-dimensional state: (user_id, item_id)
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
        Builds a neural network model for DQN using TensorFlow/Keras.
        """
        logging.info("Building DQN model...")
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.state_size,)),  # Input: (user_id, item_id)
            tf.keras.layers.Dense(64, activation='tanh', name="hidden_layer_1"),
            tf.keras.layers.Dense(32, activation='tanh', name="hidden_layer_2"),
            tf.keras.layers.Dense(self.action_size, activation='linear', name="output_layer")
        ])
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(reduction="sum_over_batch_size"),  # Explicit reduction value
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model

    def store_experience(self, state, action, reward, next_state):
        """
        Store (state, action, reward, next_state) in the replay buffer.
        """
        logging.info(f"Storing experience: {state} -> {action} -> {reward} -> {next_state}")
        self.memory.append((state, action, reward, next_state))

    def train(self):
        """
        Sample a batch from replay buffer and update the Q-network.
        """
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = []
        targets = []
        logging.info(f"Training on batch of size {len(minibatch)}")
        for state, action, reward, next_state in minibatch:
            state_arr = np.array([state], dtype=np.float32)  # shape=(1, 2)

            # Predict current Q-values
            current_qs = self.model.predict(state_arr, verbose=0)[0]

            # Index of the action in action_space
            action_idx = np.where(self.action_space == action)[0][0]

            # Compute target
            if next_state is None:
                target_q = reward
            else:
                next_state_arr = np.array([next_state], dtype=np.float32)
                future_qs = self.model.predict(next_state_arr, verbose=0)[0]
                target_q = reward + self.gamma * np.max(future_qs)

            # Update the Q-value for the action taken
            updated_qs = current_qs.copy()
            updated_qs[action_idx] = target_q

            states.append(state)
            targets.append(updated_qs)

        states_np = np.array(states, dtype=np.float32)
        targets_np = np.array(targets, dtype=np.float32)

        # Train on this batch
        self.model.fit(states_np, targets_np, epochs=1, verbose=0)

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def recommend(self, state, top_n=10, candidate_actions=None):
        """
        Recommend top-N actions based on Q-values for the given state (user_id, item_id).
        """
        if candidate_actions is None or len(candidate_actions) == 0:
            candidate_actions = self.action_space

        state_arr = np.array([state], dtype=np.float32)  # shape=(1, 2)
        q_values = self.model.predict(state_arr, verbose=0)[0]  # shape=(action_size,)

        scores = []
        for c in candidate_actions:
            c_idx = np.where(self.action_space == c)[0][0]
            scores.append((c, q_values[c_idx]))

        # Sort by Q-value descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scores[:top_n]]

    def save_model(self):
        """
        Saves the model to self.model_file.
        """
        logging.info(f"Saving model to {self.model_file}")
        self.model.save(self.model_file)

    def load_model(self):
        """
        Loads the model from self.model_file.
        """
        self.model = tf.keras.models.load_model(self.model_file)
        logging.info(f"Model loaded from {self.model_file}")