import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from pettingzoo.mpe import simple_tag_v3

"""
Training: Multi-Agent Proximal Policy Optimization (MAPPO)

This script trains 3 adversaries (predators) in the PettingZoo
`simple_tag_v3` environment to cooperatively capture 1 prey.

Architecture:
- Centralized Training, Decentralized Execution (CTDE):
  - The Critic receives the global state (concatenated predator observations).
  - The Actor receives only local observations.
- Parameter Sharing:
  - All adversaries share the same Actor network.
- Cooperative Reward:
  - Predators receive the mean team reward to encourage coordination.

Core PPO Components:
1. PPO with clipped objective for stable policy updates.
2. Generalized Advantage Estimation (GAE).
3. Running mean/variance normalization for observations and global state.
4. Optional pretrained prey policy (falls back to random behavior).

Training Details:
- Vectorized over multiple parallel environments.
- Rollouts collected for ROLLOUT_T steps.
- Multiple PPO epochs with minibatch updates.
- Entropy coefficient linearly decayed over training.

Model Saving:
- The best model is selected based on a moving average of the rollout
  mean team reward over the last 100 updates.
- Periodic checkpoints are also saved during training.
"""

#config
DEVICE = "cpu"      
NUM_ENVS = 8
ROLLOUT_T = 256
UPDATES = 6000 
MAX_CYCLES = 300        

GAMMA = 0.99    #Discount factor for future rewards
LAMBDA = 0.95   #Smoothing factor for GAE (Generalized Advantage Estimation)
CLIP = 0.12     #PPO policy clipping range to prevent destructively large updates
ACT_LR = 3e-4   #Learning rate for the Actor (policy)
CRITIC_LR = 7e-4    #Learning rate for the Critic (value function)
EPOCHS = 8  #Number of times to reuse each rollout batch for training
MINIBATCH = 1024    # Size of data samples per gradient update
ENT_START = 0.02    #Initial entropy coefficient for high exploration
ENT_END = 0.0002    #Final entropy coefficient (decayed) for exploitation
MAX_GRAD_NORM = 0.5     #Prevents "exploding gradients"

ACT_DIM = 5
N_ADV = 3
ADV_IDS = [f"adversary_{i}" for i in range(N_ADV)]
PREY_ID = "agent_0"

#normalization of inputs by calculating running mean and variance
class RunningMeanStd:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = eps

    def update(self, x):
        x = np.asarray(x, np.float64)
        bm = x.mean(axis=0)
        bv = x.var(axis=0)
        bc = x.shape[0]
        self.update_from_moments(bm, bv, bc)

    def update_from_moments(self, bm, bv, bc):
        delta = bm - self.mean
        tot = self.count + bc
        new_mean = self.mean + delta * bc/tot
        m_a = self.var * self.count
        m_b = bv * bc
        M2 = m_a + m_b + (delta ** 2) * self.count * bc/tot
        new_var = M2/tot
        self.mean, self.var, self.count = new_mean, new_var, tot

    def norm(self, x, clip=10.0):
        x = (x - self.mean)/np.sqrt(self.var + 1e-8)
        return np.clip(x, -clip, clip)

#load prey model eval if it exists
class PreyPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256, action_dim=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.network(x)

#if prey didn't exist, random actions are performed by prey
class PreyAgent:
    def __init__(self, prey_model_path: Path, obs_dim: int):
        self.use_model = False
        self.obs_dim = obs_dim
        if prey_model_path.exists():
            try:
                self.model = PreyPolicyNetwork(obs_dim=obs_dim, hidden_dim=256, action_dim=ACT_DIM)
                self.model.load_state_dict(torch.load(prey_model_path, map_location="cpu"))
                self.model.eval()
                self.use_model = True
                print(f"Loaded prey model from {prey_model_path}")
            except Exception as e:
                print(f"Could not load prey model ({e}). Using random prey.")
                self.use_model = False

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        if not self.use_model:
            return int(np.random.randint(0, ACT_DIM))
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.model(x)
        return int(torch.argmax(logits, dim=1).item())

#the same policy network is used by all predators
class SharedActor(nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, ACT_DIM),
        )

    def forward(self, obs):
        return self.net(obs)  # logits

#estimates value (V) based on the GLOBAL state (concatenated obs).
#this helps the "Credit Assignment" problem—understanding how a team effort 
#leads to a specific outcome.
class CentralCritic(nn.Module):
    def __init__(self, state_dim, hidden=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)

#generalized advantage estimation
@torch.no_grad()
def compute_gae(rew, done, val, last_val, gamma=GAMMA, lam=LAMBDA):
    """
    Computes Generalized Advantage Estimation.
    Balances the bias-variance tradeoff in advantage estimation.
    delta = r + gamma * V(s') - V(s)
    """
    T, E = rew.shape
    adv = torch.zeros(T, E, device=rew.device)
    last_gae = torch.zeros(E, device=rew.device)

    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        next_v = last_val if t == T - 1 else val[t + 1]
        delta = rew[t] + gamma * next_v * nonterm - val[t]
        last_gae = delta + gamma * lam * nonterm * last_gae
        adv[t] = last_gae

    ret = adv + val
    return adv, ret

#environment creation
def make_env():
    return simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=MAX_CYCLES,
        continuous_actions=False,
    )

def main():
    torch.set_num_threads(1)
    device = torch.device(DEVICE)
    best_avg = -1e9

    #creating envs
    envs = [make_env() for _ in range(NUM_ENVS)]
    obs_list = []
    for e in envs:
        obs, _ = e.reset()
        obs_list.append(obs)

    #detect predator obs dim dynamically
    sample_adv_obs = np.asarray(obs_list[0][ADV_IDS[0]], dtype=np.float32)
    OBS_DIM = int(sample_adv_obs.shape[0])

    #detect prey obs dim dynamically
    sample_prey_obs = np.asarray(obs_list[0][PREY_ID], dtype=np.float32)
    PREY_OBS_DIM = int(sample_prey_obs.shape[0])

    STATE_DIM = OBS_DIM * N_ADV

    print(f"Detected OBS_DIM(predator)={OBS_DIM}, PREY_OBS_DIM={PREY_OBS_DIM}, STATE_DIM={STATE_DIM}")

    actor = SharedActor(obs_dim=OBS_DIM).to(device)
    critic = CentralCritic(state_dim=STATE_DIM).to(device)

    opt_a = torch.optim.Adam(actor.parameters(), lr=ACT_LR)
    opt_c = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

    #normalizers
    obs_rms = RunningMeanStd((OBS_DIM,))
    state_rms = RunningMeanStd((STATE_DIM,))

    #optional prey model in same folder as script
    here = Path(__file__).parent
    prey = PreyAgent(here / "prey_model.pth", obs_dim=PREY_OBS_DIM)

    def concat_state(obs_dict):
        return np.concatenate([np.asarray(obs_dict[aid], dtype=np.float32) for aid in ADV_IDS], axis=0)

    def norm_obs(x):
        return obs_rms.norm(x)

    def norm_state(s):
        return state_rms.norm(s)

    avg100 = []
    global_steps = 0

    for update in range(1, UPDATES + 1):
        #entropy decay
        frac = min(1.0, update/(UPDATES * 0.7))
        ent_coef = ENT_START + frac * (ENT_END - ENT_START) #we use entropy to decide whether to explore more or less(high entropy early=exploration, low entropy later=exploitation)

        #update normalization statistics using current environment snapshot
        #provides online observation/state normalization for training stability
        sample_obs = []
        sample_state = []
        for e in range(NUM_ENVS):
            for aid in ADV_IDS:
                sample_obs.append(np.asarray(obs_list[e][aid], dtype=np.float32))
            sample_state.append(concat_state(obs_list[e]))
        obs_rms.update(np.stack(sample_obs, axis=0))
        state_rms.update(np.stack(sample_state, axis=0))

        #rollout buffers
        buf_obs = torch.zeros(ROLLOUT_T, NUM_ENVS, N_ADV, OBS_DIM, device=device)
        buf_state = torch.zeros(ROLLOUT_T, NUM_ENVS, STATE_DIM, device=device)
        buf_act = torch.zeros(ROLLOUT_T, NUM_ENVS, N_ADV, dtype=torch.long, device=device)
        buf_logp = torch.zeros(ROLLOUT_T, NUM_ENVS, N_ADV, device=device)
        buf_rew = torch.zeros(ROLLOUT_T, NUM_ENVS, device=device)
        buf_done = torch.zeros(ROLLOUT_T, NUM_ENVS, device=device)
        buf_val = torch.zeros(ROLLOUT_T, NUM_ENVS, device=device)

        #Step 1. Data collection (Rollout)
        for t in range(ROLLOUT_T):
            obs_batch = np.zeros((NUM_ENVS, N_ADV, OBS_DIM), dtype=np.float32)
            state_batch = np.zeros((NUM_ENVS, STATE_DIM), dtype=np.float32)

            for e in range(NUM_ENVS):
                for i, aid in enumerate(ADV_IDS):
                    obs_batch[e, i] = norm_obs(np.asarray(obs_list[e][aid], dtype=np.float32))
                state_batch[e] = norm_state(concat_state(obs_list[e]))

            obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=device)         # [E,A,OBS]
            state_t = torch.tensor(state_batch, dtype=torch.float32, device=device)     # [E,STATE]

            #action sampling
            with torch.no_grad():
                logits = actor(obs_t.reshape(-1, OBS_DIM))  # [E*A, ACT]
                dist = torch.distributions.Categorical(logits=logits)
                act_flat = dist.sample()
                logp = dist.log_prob(act_flat).view(NUM_ENVS, N_ADV)
                act_ea = act_flat.view(NUM_ENVS, N_ADV).cpu().numpy()
                v = critic(state_t)  # [E]

            next_obs_list = []
            team_rew = np.zeros((NUM_ENVS,), dtype=np.float32)
            done_e = np.zeros((NUM_ENVS,), dtype=np.float32)

            for e in range(NUM_ENVS):
                actions = {}
                for i, aid in enumerate(ADV_IDS):
                    actions[aid] = int(act_ea[e, i])

                prey_obs = np.asarray(obs_list[e][PREY_ID], dtype=np.float32)
                actions[PREY_ID] = prey.act(prey_obs)

                nxt_obs, rewards, terms, truncs, infos = envs[e].step(actions)

                #team reward (mean predator reward): encourages cooperative behaviour and prevents individual selfish policies
                r = 0.0
                for aid in ADV_IDS:
                    r += float(rewards.get(aid, 0.0))
                r /= N_ADV
                team_rew[e] = r

                #environment is considered finished if ANY agent terminates or truncates so this matches PettingZoo parallel environment episode handling
                d = False
                for k in terms.keys():
                    d = d or bool(terms[k]) or bool(truncs[k])
                done_e[e] = 1.0 if d else 0.0

                if d:
                    nxt_obs, _ = envs[e].reset()

                next_obs_list.append(nxt_obs)

            #store rollout
            buf_obs[t] = obs_t
            buf_state[t] = state_t
            buf_act[t] = torch.tensor(act_ea, dtype=torch.long, device=device)
            buf_logp[t] = logp
            buf_rew[t] = torch.tensor(team_rew, dtype=torch.float32, device=device)
            buf_done[t] = torch.tensor(done_e, dtype=torch.float32, device=device)
            buf_val[t] = v

            obs_list = next_obs_list
            global_steps += NUM_ENVS

        #Step 2. Compute advantages and returns for the collected rollout
        with torch.no_grad():
            last_state = np.zeros((NUM_ENVS, STATE_DIM), dtype=np.float32)
            for e in range(NUM_ENVS):
                last_state[e] = norm_state(concat_state(obs_list[e]))
            last_state_t = torch.tensor(last_state, dtype=torch.float32, device=device)
            last_val = critic(last_state_t)  # [E]

        #advantages/returns (detached)
        adv, ret = compute_gae(buf_rew, buf_done, buf_val, last_val)
        adv = (adv - adv.mean())/(adv.std() + 1e-8)

        #flatten rollout buffers into PPO training batch
        T, E, A = ROLLOUT_T, NUM_ENVS, N_ADV
        B = T * E

        obs = buf_obs.reshape(B, A, OBS_DIM)
        act = buf_act.reshape(B, A)
        old_logp = buf_logp.reshape(B, A)
        state = buf_state.reshape(B, STATE_DIM)
        ret = ret.reshape(B)
        adv = adv.reshape(B)
        advA = adv.unsqueeze(1).expand(B, A)

        idx = torch.randperm(B, device=device)

        #Step 3. Training loop with multiple epochs and mini-batches
        for _ in range(EPOCHS):
            for start in range(0, B, MINIBATCH):
                mb = idx[start:start + MINIBATCH]

                #actor update
                logits = actor(obs[mb].reshape(-1, OBS_DIM))
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act[mb].reshape(-1)).reshape(-1, A)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp[mb])
                surr1 = ratio * advA[mb]
                surr2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * advA[mb]
                pi_loss = -torch.min(surr1, surr2).mean() - ent_coef * entropy

                opt_a.zero_grad()
                pi_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
                opt_a.step()

                #critic update
                v = critic(state[mb])
                v_loss = F.mse_loss(v, ret[mb])

                opt_c.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
                opt_c.step()

        #log
        mean_team = float(buf_rew.mean().item())
        avg100.append(mean_team)
        if len(avg100) > 100:
            avg100.pop(0)

        if update % 25 == 0:
            print(
                f"Update {update:4d}/{UPDATES} | steps {global_steps:9d} | "
                f"mean_team {mean_team:+.4f} | avg100 {np.mean(avg100):+.4f} | ent {ent_coef:.4f}"
            )

        if np.mean(avg100) > best_avg:
            best_avg = np.mean(avg100)
            out = Path(__file__).parent / "best_predator_model.pth"
            torch.save({"actor": actor.state_dict(), "obs_dim": OBS_DIM}, out)
            print(f"Saved -> {out}")
        
        #save periodically
        if update % 200 == 0:
            out = Path(__file__).parent / "predator_model.pth"
            torch.save({"actor": actor.state_dict(), "obs_dim": OBS_DIM}, out)
            print(f"Saved -> {out}")
        

    #saving final model
    out = Path(__file__).parent / "predator_model.pth"
    torch.save({"actor": actor.state_dict(), "obs_dim": OBS_DIM}, out)
    print(f"Done. Saved -> {out}")

if __name__ == "__main__":
    main()
