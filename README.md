# ACIT4630 Semester Project - RL+IL Hybrid model

Content
- [Instructions and Setup](#instructions--setup)
    - [Required Applications](#1-required-applications)
    - [Library Setup](#2-library-setup)
    - [Training Models](#3-training-models)
    - [Running Existing Models](#4-running-existing-models)
- [Extra - Imitation Learning](#extra---imitation-learning)
    - [Collect own Imitation Data](#collect-imitation-data-yourself)
    - [Running IL Model](#run-the-il-model)
- [Data Analysis Tools](#data-analysis-tools)
- [Config Template](#config-template)

Link to the TMRL library [README](/Environment_README.md)

## Instructions & Setup
Before using the TMRL library, there are a few external applications you need to install and some initial setup steps to follow.
This section covers all of that in detail.

### 1. Required Applications

#### Ubisoft Account & TrackMania
- Create a Ubisoft account via the [official website](https://www.ubisoft.com/).
- Download and install **Ubisoft Connect (Uplay)** from [here](https://ubisoftconnect.com/).
- Use Ubisoft Connect to install **TrackMania**.

*Note: TrackMania is available on Steam as well, but it still requires a Ubisoft account and Ubisoft Connect to run.*

#### Openplanet Client
- Download the Openplanet client for TrackMania from [openplanet.dev](https://openplanet.dev/download).
- This is a required tool that enables custom scripting and external interaction with TrackMania during training and evaluation.

### 2. Library Setup
Once the applications above are ready, you can move on to setting up the TMRL library itself.

#### Run setup.bat
Start by running the setup.bat script in your project directory.
This script will sequentially execute:
``` bash
python setup.py --install

python setup.py --build

python tmrl/__main__.py --install
```

These commands will create **two important folders**:

##### Egg Folder
**Path:** `C:\Users\[Your_User]\AppData\Local\Programs\Python\Python313\Lib\site-packages\tmrl-0.7.0-py3.13.egg\tmrl`

**Contents**
- A clone of the tmrl library, including files such as `networking.py`, `__main__.py`, and others.
- These files must exactly match the versions in your main TMRL project folder.

:warning: If they are not identical, you will likely run into runtime errors or inconsistencies during execution.

##### Model Data Folder
**Path:** `C:\Users\[Your_user]\TmrlData`

**Purpose:**
- This folder stores all model-related data, including:
    - Trained weights
    - Reward function settings
    - Checkpoints
    - Configuration files

### 3. Training Models
All model weights, logs, and configuration are stored in the TmrlData folder located at:
```
C:\Users\[Your_user]\TmrlData
```
If you want to start training from scratch, you need to:
1. **Delete** the existing TmrlData folder.
2. Run `newModel.bat` to generate a clean one.
3. Open the config file at `TmrlData/config/config.json`, and **manually update its contents** using the provided config template ([see bottom of this document](#config-template))
>[!IMPORTANT]
> Make sure the `"RUN_NAME"` field is set to something **unique**

#### Pure Reinforcement Learning (RL) Models
To train a model using pure reinforcement learning run `Train RL-model.bat`

The batch file will
- Start the server `python tmrl/__main__.py --server` on port 55555
- Once the server is up, it automatically launches:
    - A worker: `python tmrl/__main__.py --rl-worker`
    - A trainer: `python tmrl/__main__.py --trainer`

These components work together to generate rollouts and update the RL model in real-time.

#### Hybrid Models (RL + Imitation Learning)
Hybrid models combine reinforcement learning with imitation learning, and each implemented variant has its own batch file.

To train a hybrid model:
-  Run the batch file corresponding to the model variant, for example:
    - `Train Hybrid-model-P1.bat`

This script behaves like the RL trainer but starts with a few key steps:
- Deletes any existing `bc_model.pth` file.
- Copies the desired IL model (f.ex. P1) and renames it to `bc_model.pth`
- And then it runs `--worker` instead of `--rl-worker`

### 4. Running Existing Models
If you want to **resume training or continue from a previously trained model:**
1. Navigate to the `All Models TmrlData` folder — this contains saved `TmrlData` directories from earlier test runs.
2. Pick the model folder you want to use.
3. **Delete or rename** your current *TmrlData* folder (This one: `C:\Users\[Your_user]\TmrlData`)
4. **Copy and paste** the chosen model folder into the same location and rename it to *TmrlData*.

Once replaced:

Start the server and training using the **corresponding hybrid batch file**, e.g.:
- `Train Hybrid-model-P1.bat`
> [!NOTE]
> Even though the batch script will run regardless of which model you choose, using the correct one ensures you're loading both the correct **RL state** (from the TmrlData folder) and the matching **IL model** (`bc_model.pth` for the same permutation in the project folder).

#### Running Without Further Training
If you simply want to **run the model without continuing training**:
- Launch the hybrid batch script as usual.
- Once the trainer window appears, **just close it**.
- The server and worker will continue to run the existing model in inference mode.

## Extra - Imitation Learning
If you want to train a pure imitation learning model or create a hybrid model that combines your custom IL with reinforcement learning, this section explains how to collect driving data, train the model, and run it in both training and evaluation modes.

### Collect Imitation Data Yourself
To start, you need to record your own driving data from TrackMania.
This data will be used to train the IL model.

#### 1. **Run the following batch file**: `Collect-IL-Data.bat`
This runs:
- `python tmrl/__main__.py --serve`
- `python tmrl/__main__.py --imitation`

This will:
- Collect LiDAR readings, velocity data, and your manual driving inputs.
- Save each step to a file called “demonstration_data.csv”
> [!NOTE]
> 50000 lines/steps is approximately 60-70 laps

#### 2. **Run the following batch file to train the IL Model**: `Train-IL-model.bat`
This runs `csvModifier.py` which filters `demonstration_data.csv` and give you `demonstration_filtered.csv`\
(removes the idle steps. Otherwise the model won’t be able to start).

Then runs `IL-nn.py` to create `bc_model.pth`

> [!TIP]
> You should make a copy of `bc_model.pth` which you call something unique like `bc_model_custom.pth`, this way when `bc_model.pth` is overwritten from running `Train Hybrid-model-P3.bat` (or something), you still have the model saved.

### Run the IL Model
At this point, your imitation learning model is ready to use.

#### Option 1: Train Hybrid Model Using IL
Run: “Train Hybrid-model.bat”
- This will begin training using the `bc_model.pth` file.
- If you’ve trained your own IL model, **back it up and rename** it to avoid it being overwritten by another batch file (e.g., `bc_model_custom.pth`).

#### Option 2: Run IL-Only Mode
To run the IL on its own, similar to how we tested the IL models with the 30-laps test, run `Imitation-Worker.bat`.
- This will use the current `bc_model.pth` file
- Starts the server `python tmrl/__main__.py --server`
- And then `python tmrl/__main__.py --imitation-worker`

This will produce `IL-rew.json` which saves the reward at the end of each lap and can be used in `IL_model_analysis.py` to map how far the IL model got through multiple runs.

## Data Analysis Tools
We've created several graphing tools to visualize and compare model performance. These scripts were used for generating report figures and internal analysis:
- `Box_plot_results.py` \
→ Generates box plots to compare different model groups.
- `Display-ComparativeData.py`\
→ Plots side-by-side comparisons of various model runs.
- `Display-GoalData.py`
→ Plots a single model’s performance over time.
- `IL_model_analysis.py`
→ (*Tool-specific functionality; likely supports IL evaluation and visualization.*)

### Time Alignment for 24-Hour Analysis
Our analytics scripts are designed to isolate performance data across 24-hour periods, using timestamps from goal_timestamps.json. To make this work
> [!IMPORTANT]
> You must provide the **correct training start time** (e.g., 2025-05-01 18:07).

## Config Template
```json
{
  "RUN_NAME": "SAC_4_lidar_Examinator",
  "RESET_TRAINING": false,
  "BUFFERS_MAXLEN": 500000,
  "RW_MAX_SAMPLES_PER_EPISODE": 1000,
  "CUDA_TRAINING": true,
  "CUDA_INFERENCE": false,
  "VIRTUAL_GAMEPAD": true,
  "DCAC": false,
  "LOCALHOST_WORKER": true,
  "LOCALHOST_TRAINER": true,
  "PUBLIC_IP_SERVER": "0.0.0.0",
  "PASSWORD": "==>TMRL@UseASecurePasswordHere!<==",
  "TLS": false,
  "TLS_HOSTNAME": "default",
  "TLS_CREDENTIALS_DIRECTORY": "",
  "NB_WORKERS": -1,
  "WANDB_PROJECT": "ImitationReinforcment-TrainingData",
  "WANDB_ENTITY": "sofiehk-oslomet",
  "WANDB_KEY": "78b365b6e95165722322a2d64629035428c95098",
  "PORT": 55555,
  "LOCAL_PORT_SERVER": 55556,
  "LOCAL_PORT_TRAINER": 55557,
  "LOCAL_PORT_WORKER": 55558,
  "BUFFER_SIZE": 536870912,
  "HEADER_SIZE": 12,
  "SOCKET_TIMEOUT_CONNECT_TRAINER": 300.0,
  "SOCKET_TIMEOUT_ACCEPT_TRAINER": 300.0,
  "SOCKET_TIMEOUT_CONNECT_ROLLOUT": 300.0,
  "SOCKET_TIMEOUT_ACCEPT_ROLLOUT": 300.0,
  "SOCKET_TIMEOUT_COMMUNICATE": 30.0,
  "SELECT_TIMEOUT_OUTBOUND": 30.0,
  "ACK_TIMEOUT_WORKER_TO_SERVER": 300.0,
  "ACK_TIMEOUT_TRAINER_TO_SERVER": 300.0,
  "ACK_TIMEOUT_SERVER_TO_WORKER": 300.0,
  "ACK_TIMEOUT_SERVER_TO_TRAINER": 7200.0,
  "RECV_TIMEOUT_TRAINER_FROM_SERVER": 7200.0,
  "RECV_TIMEOUT_WORKER_FROM_SERVER": 600.0,
  "WAIT_BEFORE_RECONNECTION": 10.0,
  "LOOP_SLEEP_TIME": 1.0,
  "MAX_EPOCHS": 1000,
  "ROUNDS_PER_EPOCH": 100,
  "TRAINING_STEPS_PER_ROUND": 200,
  "MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP": 4.0,
  "ENVIRONMENT_STEPS_BEFORE_TRAINING": 1000,
  "UPDATE_MODEL_INTERVAL": 200,
  "UPDATE_BUFFER_INTERVAL": 200,
  "SAVE_MODEL_EVERY": 0,
  "MEMORY_SIZE": 1000000,
  "BATCH_SIZE": 256,
  "ALG": {
    "ALGORITHM": "SAC",
    "LEARN_ENTROPY_COEF":false,
    "LR_ACTOR":0.00001,
    "LR_CRITIC":0.00005,
    "LR_ENTROPY":0.0003,
    "GAMMA":0.995,
    "POLYAK":0.995,
    "TARGET_ENTROPY":-0.5,
    "ALPHA":0.01,
    "REDQ_N":10,
    "REDQ_M":2,
    "REDQ_Q_UPDATES_PER_POLICY_UPDATE":20,
    "OPTIMIZER_ACTOR": "adam",
    "OPTIMIZER_CRITIC": "adam",
    "BETAS_ACTOR": [0.997, 0.997],
    "BETAS_CRITIC": [0.997, 0.997],
    "L2_ACTOR": 0.0,
    "L2_CRITIC": 0.0
  },
  "ENV": {
    "RTGYM_INTERFACE": "TM20LIDAR",
    "WINDOW_WIDTH": 958,
    "WINDOW_HEIGHT": 488,
    "IMG_WIDTH": 64,
    "IMG_HEIGHT": 64,
    "IMG_GRAYSCALE": true,
    "SLEEP_TIME_AT_RESET": 1.5,
    "IMG_HIST_LEN": 4,
    "RTGYM_CONFIG": {
      "time_step_duration": 0.05,
      "start_obs_capture": 0.04,
      "time_step_timeout_factor": 1.0,
      "act_buf_len": 2,
      "benchmark": false,
      "wait_on_done": true,
      "ep_max_length": 1000,
      "interface_kwargs": {"save_replays": false}
    },
    "REWARD_CONFIG": {
        "END_OF_TRACK": 100.0,
        "CONSTANT_PENALTY": 0.0,
        "CHECK_FORWARD": 500,
        "CHECK_BACKWARD": 10,
        "FAILURE_COUNTDOWN": 10,
        "MIN_STEPS": 70,
        "MAX_STRAY": 100.0
    }
  },
  "__VERSION__": "0.6.0"
}

```