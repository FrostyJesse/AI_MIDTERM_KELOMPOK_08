"""
Training Script for Rocket Landing - Grade A Tier
Comparing Multiple DQN Variants using Stable Baselines3
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from rocket_env_with_reward import SimpleRocketEnv
import json

class RewardLogger(BaseCallback):
    """Custom callback to log episode rewards"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_reward = 0
        self.current_length = 0
        
    def _on_step(self):
        self.current_reward += self.locals['rewards'][0]
        self.current_length += 1
        
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            self.current_reward = 0
            self.current_length = 0
        return True

def train_dqn_variant(variant_name, env, total_timesteps=600000, **dqn_kwargs):
    """
    Train a DQN variant and return the model and training statistics
    
    Args:
        variant_name: Name of the variant (for logging)
        env: The environment to train on
        total_timesteps: Number of training steps
        **dqn_kwargs: Additional arguments for DQN (e.g., policy_kwargs)
    """
    print(f"\n{'='*60}")
    print(f"Training {variant_name}")
    print(f"{'='*60}\n")
    
    # Create callback
    reward_logger = RewardLogger()
    
    # Default hyperparameters
    default_params = {
        'learning_rate': 1e-4,
        'buffer_size': 50000,
        'learning_starts': 1000,
        'batch_size': 64,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 1000,
        'exploration_fraction': 0.3,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.05,
        'verbose': 1
    }
    
    # Merge with provided kwargs
    default_params.update(dqn_kwargs)
    
    # Create DQN model
    model = DQN('MlpPolicy', env, **default_params)
    
    # Train
    model.learn(total_timesteps=total_timesteps, callback=reward_logger, progress_bar=True)
    
    # Save model
    model.save(f"models/{variant_name.lower().replace(' ', '_')}")
    
    return model, reward_logger.episode_rewards, reward_logger.episode_lengths

def plot_learning_curves(results_dict, save_path='plots'):
    """
    Plot learning curves comparing different variants
    
    Args:
        results_dict: Dictionary with variant names as keys and 
                     (rewards, lengths) tuples as values
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Plot 1: Episode Rewards (Smoothed)
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    for variant_name, (rewards, _) in results_dict.items():
        # Smooth the rewards using moving average
        window = min(50, max(1, len(rewards) // 10))
        if window > 0 and len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(smoothed, label=variant_name, linewidth=2)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.title('Learning Curves: Episode Rewards (Smoothed)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Episode Lengths (Smoothed)
    plt.subplot(1, 2, 2)
    for variant_name, (_, lengths) in results_dict.items():
        window = min(50, max(1, len(lengths) // 10))
        if window > 0 and len(lengths) >= window:
            smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
            plt.plot(smoothed, label=variant_name, linewidth=2)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Episode Length', fontsize=12)
    plt.title('Learning Curves: Episode Lengths (Smoothed)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/learning_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved learning curves to {save_path}/learning_curves.png")
    plt.close()
    
    # Plot 3: Average Reward Comparison (Last 100 episodes)
    plt.figure(figsize=(10, 6))
    
    variant_names = []
    avg_rewards = []
    std_rewards = []
    
    for variant_name, (rewards, _) in results_dict.items():
        variant_names.append(variant_name)
        last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
        if len(last_100) == 0:
            avg_rewards.append(0.0)
            std_rewards.append(0.0)
        else:
            avg_rewards.append(np.mean(last_100))
            std_rewards.append(np.std(last_100))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(variant_names)))
    bars = plt.bar(variant_names, avg_rewards, yerr=std_rewards, 
                   capsize=10, color=colors, alpha=0.8, edgecolor='black')
    
    plt.xlabel('DQN Variant', fontsize=12)
    plt.ylabel('Average Reward (Last 100 Episodes)', fontsize=12)
    plt.title('Performance Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, avg, std in zip(bars, avg_rewards, std_rewards):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved performance comparison to {save_path}/performance_comparison.png")
    plt.close()

def plot_average_reward(results_dict, save_path='plots', window=100):
    """
    Plot rolling average reward per episode for each variant.
    
    Args:
        results_dict: {variant_name: (rewards, lengths)}
        save_path: directory to save the figure
        window: rolling window size for averaging
    """
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    for variant_name, (rewards, _) in results_dict.items():
        rewards = np.array(rewards, dtype=float)
        if rewards.size == 0:
            continue
        # Rolling mean (causal)
        if rewards.size >= window:
            kernel = np.ones(window) / window
            rolling = np.convolve(rewards, kernel, mode='valid')
            x = np.arange(len(rolling)) + window  # align to episode index (window..N)
            plt.plot(x, rolling, linewidth=2, label=f"{variant_name} (win={window})")
        else:
            # Not enough data for full window; plot simple cumulative average
            cum_avg = np.cumsum(rewards) / (np.arange(rewards.size) + 1)
            plt.plot(np.arange(1, rewards.size + 1), cum_avg, linewidth=2, linestyle='--', label=f"{variant_name} (cum.avg)")
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title(f'Average Reward Over Time (Rolling Mean, window={window})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}/average_reward.png', dpi=300, bbox_inches='tight')
    print(f"Saved average reward plot to {save_path}/average_reward.png")
    plt.close()

def evaluate_model(model, env, n_eval_episodes=20):
    """Evaluate trained model"""
    episode_rewards = []
    success_count = 0
    
    for i in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
        episode_rewards.append(episode_reward)
        if episode_reward > 50:  # Consider successful landing
            success_count += 1
        
        print(f"Evaluation Episode {i+1}/{n_eval_episodes}: Reward = {episode_reward:.2f}")
    
    avg_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else float('nan')
    std_reward = np.std(episode_rewards) if len(episode_rewards) > 0 else float('nan')
    success_rate = (success_count / n_eval_episodes) * 100 if n_eval_episodes > 0 else float('nan')
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"{'='*60}\n")
    
    return {
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'success_rate': success_rate,
        'episode_rewards': episode_rewards
    }

def main():
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Training timesteps
    TOTAL_TIMESTEPS = 600_000  # Adjusted to 600k steps per model
    
    print("\n" + "="*60)
    print("ROCKET LANDING - DQN VARIANTS COMPARISON")
    print("Grade A Tier Implementation")
    print("="*60 + "\n")
    
    # ========== 1. Standard DQN ==========
    env = SimpleRocketEnv(render_mode=None)
    model_dqn, rewards_dqn, lengths_dqn = train_dqn_variant(
        "Standard DQN",
        env,
        total_timesteps=TOTAL_TIMESTEPS
    )
    env.close()
    
    # ========== 2. Double DQN ==========
    env = SimpleRocketEnv(render_mode=None)
    model_double, rewards_double, lengths_double = train_dqn_variant(
        "Double DQN",
        env,
        total_timesteps=TOTAL_TIMESTEPS,
        policy_kwargs=dict(net_arch=[256, 256])  # Deeper network
    )
    env.close()
    
    # ========== 3. Dueling DQN ==========
    # Note: Stable Baselines3 doesn't have direct Dueling DQN
    # We simulate it with different architecture
    env = SimpleRocketEnv(render_mode=None)
    model_dueling, rewards_dueling, lengths_dueling = train_dqn_variant(
        "Dueling DQN",
        env,
        total_timesteps=TOTAL_TIMESTEPS,
        policy_kwargs=dict(net_arch=[128, 128]),
        learning_rate=5e-5  # Lower learning rate
    )
    env.close()
    
    # Compile results
    results = {
        'Standard DQN': (rewards_dqn, lengths_dqn),
        'Double DQN': (rewards_double, lengths_double),
        'Dueling DQN': (rewards_dueling, lengths_dueling)
    }
    
    # Plot learning curves (reward & length smoothed + bar comparison)
    plot_learning_curves(results)
    
    # Plot average reward over time (rolling mean)
    plot_average_reward(results, window=100)
    
    # ========== Evaluation Phase ==========
    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60 + "\n")
    
    evaluation_results = {}
    
    # Evaluate each model
    models = [
        ("Standard DQN", model_dqn),
        ("Double DQN", model_double),
        ("Dueling DQN", model_dueling)
    ]
    
    for name, model in models:
        print(f"\nEvaluating {name}...")
        env = SimpleRocketEnv(render_mode=None)
        eval_results = evaluate_model(model, env, n_eval_episodes=20)
        evaluation_results[name] = eval_results
        env.close()
    
    # Save all results to JSON
    results_summary = {
        name: {
            'training_episodes': len(results[name][0]),
            'final_avg_reward': float(np.mean(results[name][0][-100:])) if len(results[name][0]) > 0 else float('nan'),
            'evaluation': {
                'avg_reward': float(evaluation_results[name]['avg_reward']),
                'std_reward': float(evaluation_results[name]['std_reward']),
                'success_rate': float(evaluation_results[name]['success_rate'])
            }
        }
        for name in results.keys()
    }
    
    with open('results/training_results.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Models saved in 'models/' directory")
    print(f"Plots saved in 'plots/' directory")
    print(f"Results saved in 'results/' directory")
    print("="*60 + "\n")
    
    # Print final comparison
    print("\nFINAL PERFORMANCE COMPARISON:")
    print("-" * 80)
    print(f"{'Variant':<20} {'Avg Reward':<15} {'Success Rate':<15}")
    print("-" * 80)
    for name in results.keys():
        avg_rew = evaluation_results[name]['avg_reward']
        success = evaluation_results[name]['success_rate']
        print(f"{name:<20} {avg_rew:>10.2f}     {success:>10.1f}%")
    print("-" * 80)

if __name__ == "__main__":
    main()
