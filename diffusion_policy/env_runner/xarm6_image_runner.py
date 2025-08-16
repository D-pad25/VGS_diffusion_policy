from typing import Dict, Any
import numpy as np
import torch
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

class XArm6ImageRunner(BaseImageRunner):
    def __init__(self, 
                 output_dir,
                 n_train,
                 n_train_vis,
                 train_start_seed,
                 n_test,
                 n_test_vis,
                 max_steps,
                 n_obs_steps,
                 n_action_steps,
                 fps,
                 past_action,
                 n_envs,
                 **kwargs):
        # Store parameters for testing
        self.output_dir = output_dir
        self.n_train = n_train
        self.n_train_vis = n_train_vis
        self.train_start_seed = train_start_seed
        self.n_test = n_test
        self.n_test_vis = n_test_vis
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.past_action = past_action
        self.n_envs = n_envs
        
        # Initialize your xArm6 environment here
        # self.env = XArm6Env()  # You'll need to create this or import from your openpi repo
        
        # For now, we'll create a placeholder environment
        self.env = None
        print("Warning: XArm6 environment not initialized. Please implement or import your xArm6 environment.")

    def _get_obs(self) -> Dict[str, Any]:
        # Get observations from your xArm6 environment
        # This should match your exact data structure
        
        if self.env is None:
            # Return dummy observations for testing (224x224 after downsampling)
            obs = {
                'base_rgb': np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
                'wrist_rgb': np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
                'state': np.random.rand(7).astype(np.float32),  # 6 joints + 1 gripper
            }
        else:
            # Implement actual environment observation using your exact keys
            obs = self.env.get_observation()
            # Ensure we have the right keys for the diffusion policy
            if 'base_rgb' not in obs:
                obs['base_rgb'] = obs.get('image', np.zeros((224, 224, 3), dtype=np.uint8))
            if 'wrist_rgb' not in obs:
                obs['wrist_rgb'] = obs.get('wrist_image', np.zeros((224, 224, 3), dtype=np.uint8))
        return obs

    def _reset_to(self, state: Dict[str, Any]):
        # Reset environment to specific state
        # Implement based on your environment
        if self.env is not None:
            # self.env.reset_to_state(state)
            pass
        else:
            print("Warning: Environment not initialized, cannot reset")

    def _step(self, action: np.ndarray) -> Dict[str, Any]:
        # Execute action in environment
        # Return observation, reward, done, info
        
        if self.env is None:
            # Return dummy step for testing
            obs = self._get_obs()
            reward = 0.0
            done = False
            info = {}
        else:
            # Implement actual environment step
            # obs, reward, done, info = self.env.step(action)
            obs = self._get_obs()
            reward = 0.0
            done = False
            info = {}
        
        return obs, reward, done, info

    def _get_obs_action_sequence(self, policy: BaseImagePolicy, 
                                obs: Dict[str, Any], 
                                goal: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get observation-action sequence from policy
        """
        # Convert obs to tensor format expected by policy
        obs_tensor = dict_apply(obs, lambda x: torch.from_numpy(x).unsqueeze(0))
        
        # Get action from policy
        with torch.no_grad():
            action = policy.predict_action(obs_tensor)
        
        # Convert back to numpy
        action = action.squeeze(0).cpu().numpy()
        
        return {
            'obs': obs,
            'action': action
        }

    def _get_obs_action_sequence_batch(self, policy: BaseImagePolicy, 
                                      obs: Dict[str, Any], 
                                      goal: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get observation-action sequence from policy for batch processing
        """
        return self._get_obs_action_sequence(policy, obs, goal)


def test():
    """
    Test the XArm6ImageRunner
    """
    runner = XArm6ImageRunner(
        output_dir="./test_output",
        n_train=1,
        n_train_vis=1,
        train_start_seed=0,
        n_test=1,
        n_test_vis=1,
        max_steps=10,
        n_obs_steps=2,
        n_action_steps=8,
        fps=10,
        past_action=False,
        n_envs=1
    )
    
    # Test observation
    obs = runner._get_obs()
    print(f"Observation keys: {obs.keys()}")
    print(f"Base RGB shape: {obs['base_rgb'].shape}")
    print(f"Wrist RGB shape: {obs['wrist_rgb'].shape}")
    print(f"State shape: {obs['state'].shape}")
    
    # Test step
    action = np.random.rand(7)
    obs, reward, done, info = runner._step(action)
    print(f"Step result - reward: {reward}, done: {done}")


if __name__ == "__main__":
    test() 