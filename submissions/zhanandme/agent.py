import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

#actor network
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )

    def forward(self, x):
        return self.net(x)

#predator policy
class StudentAgent:
    def __init__(self):
        self.device = "cpu"

        path = Path(__file__).parent / "predator_model.pth"
        ckpt = torch.load(path, map_location="cpu")

        self.obs_dim = ckpt.get("obs_dim", 16)
        self.actor = Actor(self.obs_dim)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()

    def get_action(self, observation, agent_id: str):
        obs = np.asarray(observation, dtype=np.float32)

        #note: using pad or trim observation for safety reasons
        if obs.shape[0] < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - obs.shape[0]))
        elif obs.shape[0] > self.obs_dim:
            obs = obs[:self.obs_dim]

        with torch.no_grad():
            logits = self.actor(torch.tensor(obs).unsqueeze(0))
            action = torch.argmax(logits, dim=1).item()

        return action
