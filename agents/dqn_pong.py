import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import numpy as np
import random
import time
from collections import deque
import jaxatari
import wandb  

# ==========================================
# 1. HELPER: FLATTEN OBSERVATION (OBJECT-CENTRIC)
# ==========================================
@jax.jit
def flatten_obs(obs):
    """Flattens structured JAXAtari observation into a 1D JAX array."""
    leaves = jax.tree_util.tree_leaves(obs)
    flat_array = jnp.concatenate([jnp.atleast_1d(leaf).flatten() for leaf in leaves])
    return flat_array.astype(jnp.float32) / 255.0  # Normalized for better gradient stability

# ==========================================
# 2. EVALUATION & VIDEO LOGGING FUNCTION
# ==========================================
def evaluate_and_log_video(env, q_network, params, global_step, eval_seed=123):
    """Runs a single episode without exploration and logs video to W&B."""
    print(f"\n--- Running Evaluation at Step {global_step} ---")
    eval_key = jax.random.PRNGKey(eval_seed)
    obs, eval_state = env.reset(eval_key)
    flat_obs = flatten_obs(obs)
    
    frames = []
    done = False
    eval_reward = 0.0
    
    while not done:
        frame = env.render(eval_state)
        frames.append(np.array(frame))
        
        q_values = q_network.apply(params, jnp.expand_dims(flat_obs, axis=0))
        action = int(q_values.argmax(axis=-1)[0])
        
        next_obs, next_eval_state, reward, done, info = env.step(eval_state, action)
        flat_obs = flatten_obs(next_obs)
        eval_state = next_eval_state
        eval_reward += float(reward)
        
    print(f"--- Evaluation Finished. Eval Reward: {eval_reward} ---\n")
    
    # Reshape for W&B: (Time, Channels, Height, Width)
    frames_np = np.array(frames) 
    frames_wandb = np.transpose(frames_np, (0, 3, 1, 2)) 
    
    # Log metrics and gameplay video to Weights & Biases
    wandb.log({
        "eval/episodic_return": eval_reward,
        "eval/gameplay_video": wandb.Video(frames_wandb, fps=30, format="gif"),
        "global_step": global_step
    })

# ==========================================
# 3. SIMPLE REPLAY BUFFER
# ==========================================
class SimpleReplayBuffer:
    def __init__(self, size, obs_dim):
        self.size = size
        self.ptr = 0
        self.current_size = 0
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size,), dtype=np.int32)
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.dones = np.zeros((size,), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.size
        self.current_size = min(self.current_size + 1, self.size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.current_size, size=batch_size)
        return (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obs[idxs],
            self.dones[idxs]
        )

# ==========================================
# 4. NEURAL NETWORK (MLP)
# ==========================================
class QNetwork(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

class CustomTrainState(TrainState):
    target_params: flax.core.FrozenDict

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

@jax.jit
def update(q_state, observations, actions, next_observations, rewards, dones, gamma=0.99):
    q_next_target = q_state.apply_fn(q_state.target_params, next_observations) 
    q_next_target = jnp.max(q_next_target, axis=-1) 
    next_q_value = rewards + (1 - dones) * gamma * q_next_target

    def mse_loss(params):
        q_pred = q_state.apply_fn(params, observations) 
        q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()] 
        return ((q_pred - next_q_value) ** 2).mean(), q_pred

    (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params)
    q_state = q_state.apply_gradients(grads=grads)
    return loss_value, q_pred, q_state

# ==========================================
# 5. MAIN TRAINING LOOP
# ==========================================
def main():
    # 100% Determinism (Seeds) for Reproducibility
    random.seed(42)
    np.random.seed(42)
    key = jax.random.PRNGKey(42)
    key, env_key, net_key = jax.random.split(key, 3)

    # Hyperparameters
    env_name = "pong"
    learning_rate = 2.5e-4
    buffer_size = 100000
    batch_size = 32
    target_network_frequency = 1000
    total_timesteps = 10000000  # 10 Million steps for full training
    learning_starts = 10000
    eval_frequency = 500000  # Evaluate and log a video every 500k steps

    # --- WANDB CONFIGURATION ---
    wandb.init(
        project="topic26-jaxatari",      
        entity="raef-tu-darmstadt",      
        name=f"DQN_Pong_10M_{int(time.time())}", 
        config={
            "env_name": env_name,
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "target_network_frequency": target_network_frequency,
            "total_timesteps": total_timesteps,
            "learning_starts": learning_starts,
            "eval_frequency": eval_frequency,
            "optimizer": "Adam",
            "modality": "Object-Centric"
        }
    )
    
    env = jaxatari.make(env_name)
    obs, env_state = env.reset(env_key)
    flat_obs = flatten_obs(obs) 
    
    action_dim = 6 
    obs_dim = flat_obs.shape[0]
    q_network = QNetwork(action_dim=action_dim)
    
    dummy_obs = jnp.expand_dims(flat_obs, axis=0) 
    params = q_network.init(net_key, dummy_obs)
    
    q_state = CustomTrainState.create(
        apply_fn=q_network.apply,
        params=params,
        target_params=params,
        tx=optax.adam(learning_rate=learning_rate),
    )

    rb = SimpleReplayBuffer(buffer_size, obs_dim)

    print(f"Starting Training on {env_name}... 10 Million Steps planned.")
    
    # Tracking variables
    episode_reward = 0
    episode_length = 0
    episode_count = 0  
    
    # Track the last 10 episodes for moving average metrics
    return_queue = deque(maxlen=10)
    length_queue = deque(maxlen=10)
    
    start_time = time.time()

    for global_step in range(total_timesteps):
        key, action_key = jax.random.split(key)
        
        # Epsilon decays slowly over the first 1,000,000 steps
        epsilon = linear_schedule(1.0, 0.05, 1000000, global_step) 
        if random.random() < epsilon:
            action = int(jax.random.randint(action_key, shape=(), minval=0, maxval=action_dim))
        else:
            q_values = q_network.apply(q_state.params, jnp.expand_dims(flat_obs, axis=0))
            action = int(q_values.argmax(axis=-1)[0])

        next_obs, next_env_state, reward, done, info = env.step(env_state, action)
        flat_next_obs = flatten_obs(next_obs)
        
        episode_reward += float(reward)
        episode_length += 1

        rb.add(flat_obs, action, float(reward), flat_next_obs, float(done))

        if done:
            episode_count += 1
            return_queue.append(episode_reward)
            length_queue.append(episode_length)
            
            avg_return = np.mean(return_queue)
            avg_length = np.mean(length_queue)
            
            if 'loss' in locals():
                current_loss = float(loss)
                sps = int(global_step / (time.time() - start_time))
                print(f"Ep: {episode_count} | Step: {global_step} | Avg Reward: {avg_return:.1f} | Eps: {epsilon:.2f} | SPS: {sps}")
                
                # Log metrics to W&B
                wandb.log({
                    "charts/avg_episodic_return": avg_return,
                    "charts/avg_episodic_length": avg_length,
                    "charts/epsilon": epsilon,
                    "charts/SPS": sps,
                    "losses/td_loss": current_loss,
                    "global_step": global_step,
                    "episode": episode_count
                })
            else:
                print(f"Ep: {episode_count} | Step: {global_step} | Avg Reward: {avg_return:.1f} | Eps: {epsilon:.2f} | (Training not started)")
                wandb.log({
                    "charts/avg_episodic_return": avg_return,
                    "charts/avg_episodic_length": avg_length,
                    "charts/epsilon": epsilon,
                    "global_step": global_step,
                    "episode": episode_count
                })
            
            episode_reward = 0
            episode_length = 0
            key, reset_key = jax.random.split(key)
            obs, next_env_state = env.reset(reset_key)
            flat_next_obs = flatten_obs(obs)

        flat_obs = flat_next_obs
        env_state = next_env_state

        if global_step > learning_starts and global_step % 4 == 0:
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = rb.sample(batch_size)
            loss, q_pred, q_state = update(q_state, b_obs, b_actions, b_next_obs, b_rewards, b_dones)

            if global_step % target_network_frequency == 0:
                q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1.0))

        # Evaluate and log video every 500k steps
        if global_step > 0 and global_step % eval_frequency == 0:
            evaluate_and_log_video(env, q_network, q_state.params, global_step)

    # One final evaluation at the end
    evaluate_and_log_video(env, q_network, q_state.params, total_timesteps)

    # Save final model weights locally
    model_path = "pong_dqn_10M_weights.msgpack"
    with open(model_path, "wb") as f:
        f.write(flax.serialization.to_bytes(q_state.params))
    
    print(f"\nTraining finished! Model saved locally to {model_path}")
    wandb.finish()

if __name__ == "__main__":
    main()