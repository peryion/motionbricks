# Ping-Pong Reference RL Training

This note describes the current MotionBricks -> table_tennis training flow for
the G1 ping-pong FK reference task.

## What Runs

There are two independent processes:

1. **Reference producer** in this MotionBricks repo.
   It samples ping-pong commands, solves FK/IK keyframes, asks MotionBricks to
   generate full-body clips, and writes IsaacLab-readable `.npz` files into a
   `ready/` directory.

2. **RL training** in `/home/zhipy/project/table_tennis`.
   It samples reference clips from that `ready/` directory. The training task is:

   ```text
   PingPong-One-Step-G1-FB-MotionBricks
   ```

The producer currently exports MotionBricks' 30fps output onto a 50Hz time grid
with nearest-frame qpos sampling. Per-frame `time_to_hit_s` is regenerated on
the 50Hz grid, so the countdown matches the RL control rate.

## 1. Generate an Initial Buffer

Start with a warmup buffer before launching RL. This avoids training starting
from an empty motion directory.

```bash
cd /home/zhipy/project/GR00T-WholeBodyControl/motionbricks

python scripts/generate_pingpong_reference_buffer.py \
  --output_dir out/reference_buffer/pingpong_fk \
  --num_clips 128
```

Generated files appear in:

```text
out/reference_buffer/pingpong_fk/ready
```

The config lives at:

```text
configs/pingpong_g1.yaml
```

Important fields:

```yaml
reference_buffer:
  output_fps: 50
  episode_duration_s: 10.0
  include_frozen_gaps: 1
  max_ready_files: 2048
```

Do not mix old 30Hz `.npz` files with new 50Hz files in the same `ready/`
directory. Use a fresh output directory if needed.

## 2. Optionally Keep the Producer Running

Keeping the producer alive is useful when you want the policy to keep seeing
newly generated motions during training:

```bash
cd /home/zhipy/project/GR00T-WholeBodyControl/motionbricks

python scripts/generate_pingpong_reference_buffer.py \
  --output_dir out/reference_buffer/pingpong_fk \
  --continuous
```

You do **not** have to keep it running. A fixed offline buffer is valid too. The
tradeoff is:

```text
fixed buffer:
  simpler and repeatable, but less motion variety

continuous producer:
  more variety, but needs one extra process/GPU and the buffer changes over time
```

When `max_ready_files` is reached, the producer prunes old files and writes new
ones. The RL loader refreshes the motion directory periodically and loads newly
added files incrementally.

## 3. Start RL Training

From the table_tennis repo:

```bash
cd /home/zhipy/project/table_tennis

python scripts/rsl_rl/train.py \
  --task=PingPong-One-Step-G1-FB-MotionBricks \
  --motion_file /home/zhipy/project/GR00T-WholeBodyControl/motionbricks/out/reference_buffer/pingpong_fk/ready \
  --headless \
  --logger wandb \
  --log_project_name table_tennis \
  --run_name G1_FB_motionbricks_reference
```

The same training config is also available in VSCode launch config:

```text
train-G1-FB-MotionBricks-reference
```

For playback after training, use:

```text
play-G1-FB-MotionBricks-reference
```

and replace `REPLACE_WITH_RUN_DIR` / `model_XXXXX.pt` in `.vscode/launch.json`.

## Recommended Workflow

For a normal run:

```text
Terminal 1:
  generate 128-512 warmup clips

Terminal 2:
  start --continuous producer

Terminal 3:
  start RL training
```

If GPU memory is tight, skip Terminal 2 and train from the fixed warmup buffer.
That is the safest first training test.

