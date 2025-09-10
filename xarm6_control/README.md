# xarm6_control

This module provides real-time control and observation utilities for the xArm6 robot, enabling integration with Stanford Diffusion policy. It allows the xArm6 to run inference-based actions from pre-trained models using real sensor inputs and live camera feeds.

---

## 🔧 Features

- Retrieve joint states and camera observations from the xArm6 robot
- Format observations for OpenPI policies (e.g. pi0, xarm6_policy)
- Send actions to the xArm6 controller via `position`, `velocity`, or `servo` mode
- Support for real-world deployment and debug-friendly logging

---

## Data Transformation

## Training


### Weights and Biases
The following line exists in the train basch script:
```bash
source ~/.wandb_secrets
```

This line gets the following enviroment variables:

```bash
cat <<'EOF' > ~/.wandb_secrets
# W&B authentication (private)
export WANDB_API_KEY=xxxxxxxx (can be found at [text](https://wandb.ai/authorize))
export WANDB_MODE=online
export WANDB_PROJECT=agrivla
EOF
```

## 🚀 Running a Policy on xArm6

### 1. Run the Policy Server on the HPC - NOT YET INTEGRATED

#### 1.1 ssh into the HPC

```bash
ssh n10813934@aqua.qut.edu.au
```

#### 1.2 Activate the virtual environment

```bash
conda activate robodiff
```

#### 1.3 Export the cache location to access checkpoints etc.

If you want it to grab the latest check point:
```bash
CKPT=$(ls -t /home/n10813934/gitRepos/VGS_diffusion_policy/data/outputs/2025.09.07/15.13.59_train_xarm6_diffusion_unet_image_pretrained_real_xarm_image/checkpoints/*.ckpt 2>/dev/null | head -n 1);
```

Otherwise if you want it to load a spesific checkpoint
 ```bash
CKPT=$(/path/to/checkpoint.ckpt);
```

#### 1.4 Set up the gripper server
```bash
python -m xarm6_control.gripper_server_async_v2
```
#### 1.5 Set up the camera nodes
```bash
python -m xarm6_control.zmq_core.launch_camera_nodes
```
#### 1.6 Run the policy.

##### Available Args
> - --ckpt: str
> - --remote_host: str = "localhost"
> - --remote_port: int = 8000         
> - --wrist_camera_port: int = 5000
> - --base_camera_port: int = 5001
> - --max_steps: int = 5000
> - --prompt: str = "Pick a ripe, red tomato and drop it in the blue bucket." # NOT USED YET
> - --mock: bool = False
> - --control_hz: float = 30.0
> - --step_through_instructions: bool = False
> - --delta_threshold: float = 0.25 # degrees per joint       
> - --log_dir: str = os.path.expanduser("~/diffusion_logs")
> - --save: bool = False

##### If you want to run a mock case

```bash
python -m xarm6_control.run_xarm --ckpt "$CKPT" --mock --max_steps 60 --step_through_instructions
```

##### If you want to run on xarm6, stepping through instructions

```bash
python -m xarm6_control.run_xarm --ckpt "$CKPT" --step_through_instructions
```

##### If you want to run on xarm6, the real deal

```bash
python -m xarm6_control.run_xarm --ckpt "$CKPT"
```