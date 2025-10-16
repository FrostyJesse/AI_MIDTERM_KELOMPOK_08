"""
Visualization Script for Trained Rocket Landing Agent
Manual Keyboard Control Mode
"""

import os
import numpy as np
from rocket_env_with_reward import SimpleRocketEnv
import pygame

def manual_control(n_episodes=5):
    """
    Manual keyboard control of rocket
    
    Controls:
        LEFT ARROW / A  : Move rocket left
        RIGHT ARROW / D : Move rocket right
        UP ARROW / W    : Apply thrust (reduce fall speed)
        DOWN ARROW / S  : No action / coast
        Q               : Quit
        R               : Restart episode
    
    Args:
        n_episodes: Number of episodes to play
    """
    env = SimpleRocketEnv(render_mode='human')
    
    print("\n" + "="*60)
    print("MANUAL KEYBOARD CONTROL MODE")
    print("="*60)
    print("\nKONTROL:")
    print("  LEFT ARROW / A   : Gerak ke kiri")
    print("  RIGHT ARROW / D  : Gerak ke kanan")
    print("  UP ARROW / W     : Thrust (kurangi kecepatan jatuh)")
    print("  DOWN ARROW / S   : Tidak ada aksi")
    print("  R                : Ulang episode")
    print("  Q                : Keluar")
    print("="*60 + "\n")
    
    episode = 0
    
    while episode < n_episodes:
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        print("Gunakan keyboard untuk mengendalikan roket...")
        
        while not done:
            # Capture keyboard input
            action = None
            restart_episode = False
            quit_game = False
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_game = True
                    break
                
                elif event.type == pygame.KEYDOWN:
                    # Check for directional keys and alternatives
                    if event.key in [pygame.K_LEFT, pygame.K_a]:
                        action = 1  # Move left
                    elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                        action = 2  # Move right
                    elif event.key in [pygame.K_UP, pygame.K_w]:
                        action = 3  # Apply thrust
                    elif event.key in [pygame.K_DOWN, pygame.K_s]:
                        action = 0  # No action
                    elif event.key == pygame.K_r:
                        restart_episode = True
                        break
                    elif event.key == pygame.K_q:
                        quit_game = True
                        break
            
            if quit_game:
                env.close()
                print("\nGame dihentikan oleh user.")
                return
            
            if restart_episode:
                print("Episode direstart...")
                break
            
            # Default to no action if no key pressed
            if action is None:
                action = 0
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated
            
            # Render
            env.render()
            pygame.time.delay(50)  # Control game speed
        
        if not restart_episode:
            print(f"\nEpisode {episode + 1} selesai:")
            print(f"  Steps: {step}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Coins: {env.coins}")
            episode += 1
            
            # Wait a bit before next episode
            pygame.time.wait(1000)
    
    env.close()
    print("\nSemua episode selesai!")

if __name__ == "__main__":
    manual_control(n_episodes=5)