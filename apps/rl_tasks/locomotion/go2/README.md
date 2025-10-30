# Go2 RL Locomotion - Observation Extraction

This module extracts observations from the Unitree Go2 robot for use with RL locomotion policies.

## Files

- **`get_obs.py`**: Core observation extraction functions
- **`example_usage.py`**: Complete example showing how to use the observation extraction with a policy
- **`infos.txt`**: Documentation of the observation structure (49 dimensions)

## Observation Structure (49 dimensions)

| Index | Dimension | Description | Source |
|-------|-----------|-------------|--------|
| 0-2 | 3 | Base linear velocity (body frame) | SportModeState |
| 3-5 | 3 | Base angular velocity (gyroscope) | LowState IMU |
| 6-8 | 3 | Gravity direction (body frame) | LowState IMU (quaternion) |
| 9-11 | 3 | Command velocity (forward, lateral, yaw) | External commands |
| 12 | 1 | Height command | External commands |
| 13-24 | 12 | Joint positions (relative to default) | LowState motors |
| 25-36 | 12 | Joint velocities | LowState motors |
| 37-48 | 12 | Previous actions | Action buffer |

## Usage

### Basic Usage

```python
from get_obs import get_observation_with_high_state, ObservationBuffer
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_

# Initialize
ChannelFactoryInitialize(1, "lo")
obs_buffer = ObservationBuffer()

# Subscribe to messages
def callback(low_msg: LowState_, high_msg: SportModeState_):
    # Get observation
    obs = get_observation_with_high_state(low_msg, high_msg, obs_buffer)
    
    # Use with your policy
    # actions = policy(obs)
    # obs_buffer.update_actions(actions)
```

### With Policy Controller

```python
from example_usage import PolicyController

# Initialize
controller = PolicyController()

# Set commands
controller.set_commands(
    vel_x=0.5,    # Forward 0.5 m/s
    vel_y=0.0,    # No lateral movement
    vel_yaw=0.0,  # No rotation
    height=0.3    # 0.3m height
)

# The controller automatically handles observation extraction and motor control
```

## Key Features

### ObservationBuffer
Maintains state between observations:
- Previous actions (for history)
- Command velocities
- Height command
- Default joint positions
- Latest velocity from SportModeState

### Functions

**`get_observation(low_state_msg, obs_buffer)`**
- Extracts observations from LowState only
- Velocity will be zeros if not updated via `update_high_state_velocity()`

**`get_observation_with_high_state(low_state_msg, high_state_msg, obs_buffer)`** ⭐ RECOMMENDED
- Extracts observations using both LowState and SportModeState
- Provides accurate velocity from SportModeState
- More reliable than integration methods

## Motor Order (Go2)

The Go2 has 12 motors in this order:
- 0-2: FR (Front Right) - hip, thigh, calf
- 3-5: FL (Front Left) - hip, thigh, calf
- 6-8: RR (Rear Right) - hip, thigh, calf
- 9-11: RL (Rear Left) - hip, thigh, calf

## Default Standing Pose

```python
default_pos = [
    0.0, 0.0, 0.0, 0.0,      # All hips: 0 rad
    0.8, 0.8, 0.8, 0.8,      # All thighs: 0.8 rad
    -1.5, -1.5, -1.5, -1.5   # All calves: -1.5 rad
]
```

## Topics

- **`rt/lowstate`**: LowState messages (500 Hz) - motor states, IMU
- **`rt/sportmodestate`**: SportModeState messages (500 Hz) - velocity, position
- **`rt/lowcmd`**: LowCmd messages (500 Hz) - motor commands

## Notes

- **Linear Velocity**: Obtained from SportModeState for accuracy
- **Gravity Direction**: Computed from IMU quaternion using inverse rotation
- **Joint Positions**: Stored relative to default pose
- Use `domain_id=1, interface="lo"` for local simulation
- Control loop should run at ~500 Hz (dt=0.002s)

## Example: Loading PyTorch Policy

```python
import torch

# Load policy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = torch.jit.load("policy.pt", map_location=device)
policy.eval()

# In your callback
def callback(low_msg, high_msg):
    obs = get_observation_with_high_state(low_msg, high_msg, obs_buffer)
    
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        actions = policy(obs_tensor).squeeze(0).numpy()
    
    obs_buffer.update_actions(actions)
    return actions
```
