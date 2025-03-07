import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import keyboard  # For capturing manual inputs
from pathlib import Path  # Fix for path handling
import config.config_expert as cfg  # Import expert-specific config
from networking import RolloutWorker
from actor import TorchActorModule
from envs import GenericGymEnv  # Match the expected environment class

# Ensure the environment class matches config
ENV_CLS = GenericGymEnv


class ExpertRolloutWorker(RolloutWorker):
    """
    A rollout worker that collects expert data (manual control).
    """

    def __init__(
        self,
        env_cls,
        actor_module_cls,
        sample_compressor: callable = None,
        device="cpu",
        max_samples_per_episode=np.inf,
        model_path=cfg.MODEL_PATH_WORKER,
        obs_preprocessor: callable = None,
        crc_debug=False,
        model_path_history=cfg.MODEL_PATH_SAVE_HISTORY,
        model_history=cfg.MODEL_HISTORY,
        standalone=False,
        server_ip=None,
        server_port=cfg.PORT,
        password=cfg.PASSWORD,
        local_port=cfg.LOCAL_PORT_WORKER,
        header_size=cfg.HEADER_SIZE,
        max_buf_len=cfg.BUFFER_SIZE,
        security=cfg.SECURITY,
        keys_dir=cfg.CREDENTIALS_DIRECTORY,
        hostname=cfg.HOSTNAME
    ):
        """
        Initialize the expert rollout worker.
        """
        super().__init__(
            env_cls=env_cls,
            actor_module_cls=actor_module_cls,
            sample_compressor=sample_compressor,
            device=device,
            max_samples_per_episode=max_samples_per_episode,
            model_path=model_path,
            obs_preprocessor=obs_preprocessor,
            crc_debug=crc_debug,
            model_path_history=model_path_history,
            model_history=model_history,
            standalone=standalone,
            server_ip=server_ip,
            server_port=server_port,
            password=password,
            local_port=local_port,
            header_size=header_size,
            max_buf_len=max_buf_len,
            security=security,
            keys_dir=keys_dir,
            hostname=hostname
        )
        self.device = device
        self.expert_data = []  # Store expert demonstrations

    def get_manual_action(self):
        """
        Get user keyboard inputs for manual driving.
        Modify this function if using a controller or GUI.
        """
        action = np.zeros(cfg.ACT_BUF_LEN)  # Default: No action

        if keyboard.is_pressed('w'):  # Forward
            action[1] = 1.0
        if keyboard.is_pressed('s'):  # Brake
            action[1] = -1.0
        if keyboard.is_pressed('a'):  # Left
            action[0] = -1.0
        if keyboard.is_pressed('d'):  # Right
            action[0] = 1.0

        return action

    def run_expert_episode(self, max_samples):
        """
        Runs a manually controlled episode (expert demonstration).
        """
        obs, info = self.reset(collect_samples=True)
        ret = 0.0
        steps = 0
        done = False

        for _ in range(max_samples):
            act = self.get_manual_action()
            new_obs, rew, terminated, truncated, info = self.env.step(act)
            ret += rew
            steps += 1
            done = terminated or truncated

            # Store expert data
            self.expert_data.append((obs, act))

            if done:
                break

            obs = new_obs

        self.buffer.stat_train_return = ret
        self.buffer.stat_train_steps = steps

    def save_expert_data(self, filename=str(Path(cfg.DATASET_PATH) / "expert_data.pkl")):
        """
        Saves collected expert data to a file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self.expert_data, f)
        print(f"Expert data saved to {filename}")


class ExpertImitationModel(TorchActorModule):
    """
    Neural network model for imitation learning.
    """

    def __init__(self, device="cpu"):
        super().__init__(cfg.OBSERVATION_SPACE, cfg.ACTION_SPACE, device)
        self.fc1 = nn.Linear(cfg.IMG_WIDTH * cfg.IMG_HEIGHT, 128)  # Adjust input size based on image dimensions
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, cfg.ACT_BUF_LEN)  # Adjust output to match action space

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Output scaled between -1 and 1


def train_imitation_model(
    expert_data_file=str(Path(cfg.DATASET_PATH) / "expert_data.pkl"),
    save_path=str(Path(cfg.WEIGHTS_FOLDER) / "expert_model.pth"),
    device="cpu"
):
    """
    Trains an imitation learning model using collected expert data.
    """
    # Load expert data
    with open(expert_data_file, "rb") as f:
        expert_data = pickle.load(f)

    obs_data = np.array([x[0] for x in expert_data])
    act_data = np.array([x[1] for x in expert_data])

    # Convert to tensors
    obs_tensor = torch.tensor(obs_data, dtype=torch.float32, device=device)
    act_tensor = torch.tensor(act_data, dtype=torch.float32, device=device)

    # Create model
    model = ExpertImitationModel(device).to(device)

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.MSELoss()
    epochs = cfg.EPOCHS

    # Train model
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_actions = model(obs_tensor)
        loss = criterion(pred_actions, act_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Save trained model
    torch.save(model.state_dict(), save_path)
    print(f"Trained model saved to {save_path}")


if __name__ == "__main__":
    # Create expert worker
    worker = ExpertRolloutWorker(
        env_cls=ENV_CLS,
        actor_module_cls=ExpertImitationModel,
        device="cuda" if cfg.CUDA_TRAINING and torch.cuda.is_available() else "cpu",
        standalone=False,
        server_ip=cfg.SERVER_IP_FOR_WORKER,
        server_port=cfg.PORT
    )

    # Collect manual data
    print("Starting expert data collection...")
    num_episodes = 10  # Number of expert episodes
    for _ in range(num_episodes):
        worker.run_expert_episode(max_samples=1000)

    # Save collected expert data
    worker.save_expert_data()

    # Train an imitation model
    print("Training imitation learning model...")
    train_imitation_model(device=worker.device)
