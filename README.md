# Odyssey Project: Robot Haptic Feedback

MuJoCo simulations for checking contact-force estimates on a Franka Emika Panda arm. The project runs a small set of scenarios, logs ground-truth contact forces against Jacobian-based estimates, and writes comparison plots for each run.

## Scenarios

- `push_block`: moves the gripper into a free block and compares block contact force against the virtual force estimate.
- `hit_floor`: lowers the gripper toward the floor and compares floor contact force against the virtual force estimate.
- `peg_in_hole`: adds a peg, socket, IK target, and optional keyboard teleoperation for insertion practice.

## Setup

Activate your `odyssey` environment, where `python` and `mjpython` should resolve to the MuJoCo-capable environment. If you are setting up from scratch instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Franka model is loaded from:

```text
mujoco_menagerie/franka_emika_panda/scene.xml
```

On macOS, MuJoCo viewer workflows that use the passive viewer, including interactive peg control and video recording, may need `mjpython` instead of regular `python3`.

## Usage

Run the default scenario:

```bash
python3 main.py
```

Run a specific scenario:

```bash
python3 main.py --scenario push_block
python3 main.py --scenario hit_floor
python3 main.py --scenario peg_in_hole
```

Run the interactive peg-in-hole task:

```bash
mjpython main.py --scenario peg_in_hole --interactive
```

Enable live force feedback during interactive peg insertion:

```bash
mjpython main.py --scenario peg_in_hole --interactive --force-feedback
```

Choose the force-feedback visual:

```bash
mjpython main.py --scenario peg_in_hole --interactive --force-feedback --force-visual arrow
mjpython main.py --scenario peg_in_hole --interactive --force-feedback --force-visual ring
mjpython main.py --scenario peg_in_hole --interactive --force-feedback --force-visual both
```

`arrow` draws a red/orange vector at the strongest peg contact point, pointing in the world-space force direction applied to the peg. `ring` draws a red/orange ring at the strongest contact surface, and `both` draws both overlays. The size of each overlay uses a log scale from roughly `10 N` to `1000 N`, so mid-range forces remain visually distinguishable without huge spikes dominating the view.

Make the peg and socket walls semi-transparent when inspecting internal contacts:

```bash
mjpython main.py --scenario peg_in_hole --interactive --force-feedback --force-visual both --peg-alpha 0.45 --socket-alpha 0.45
```

Enable the experimental impedance cushion during interactive peg insertion:

```bash
mjpython main.py --scenario peg_in_hole --interactive --force-feedback --force-visual both --contact-cushion
```

The cushion activates after contact force crosses `--cushion-threshold` (`100 N` by default). While active, the arm position servos are commanded to the current joint positions to cancel the servo spring, and a torque-limited Cartesian spring/damper is applied through `J.T @ wrench`. You can tune it with `--impedance-kp`, `--impedance-dp`, `--impedance-kr`, `--impedance-dr`, and `--impedance-torque-limit`.

Record a video of a run:

```bash
mjpython main.py --scenario push_block --record-video
mjpython main.py --scenario peg_in_hole --interactive --record-video
mjpython main.py --scenario peg_in_hole --interactive --record-video --record-force-feedback --force-visual both
```

`--record-force-feedback` includes the same visual feedback geoms in the saved video: the green idle marker before contact, plus the selected red/orange arrow, ring, or both during contact.

Show CLI options:

```bash
python3 main.py --help
```

## Outputs

Each run writes artifacts under `results/<scenario>/`:

- `force_verification_log.csv`: raw force samples.
- `force_verification_log_filtered.csv`: samples after anomaly filtering.
- `force_comparison_raw.png`: all measured and estimated force samples.
- `force_comparison_filtered.png`: filtered measured and estimated force samples.
- `force_comparison_contact_only_raw.png`: contact-only raw comparison.
- `force_comparison_contact_only_filtered.png`: contact-only filtered comparison.
- `run_recording.mp4`: video output when `--record-video` is used.

Reference outputs are checked in under `sample_results/`.

## Project Layout

```text
main.py                         CLI entrypoint
franka_force/config.py          Paths, scenario names, video defaults
franka_force/env.py             Shared MuJoCo environment and viewer orchestration
franka_force/recording.py       Offscreen MP4 recording helper
franka_force/plotting.py        CSV filtering and plot generation
franka_force/scenarios/         Scenario-specific model, control, and contact logic
```

`FrankaForceEnv` delegates scenario-specific behavior through the scenario registry in `franka_force/scenarios/__init__.py`, so adding a new scenario should usually mean adding one scenario module and registering it there.

The CSV files also include cushion state, cushion scale, impedance torque norm, and the strongest contact-force vector components.

## Control Experiments

The `--contact-cushion` mode is reactive, not predictive: it only engages after MuJoCo reports contact force. Without a proximity sensor or a simulation-only distance check, the controller cannot slow before first contact.

The experimental cushion uses the impedance idea:

```text
tau_impedance = J.T @ (K * (X_target - X_current) - D * Xdot_current)
```

This first version keeps normal IK as the default and switches to torque-level impedance only after the force threshold is exceeded. It is useful for comparing force graphs with and without cushioning, but the gains should be treated as tuning parameters rather than final control values.

## Development Checks

Compile the Python files:

```bash
python3 -m compileall main.py franka_force
```

Check the command-line interface without launching MuJoCo:

```bash
python3 main.py --help
```
