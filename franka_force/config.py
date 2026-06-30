from pathlib import Path

MODEL_PATH = Path("mujoco_menagerie/franka_emika_panda/scene.xml")
RESULTS_DIR = Path("results")
SCENARIOS = ("hit_floor", "push_block", "peg_in_hole")
FORCE_VISUAL_MODES = ("arrow", "ring", "both")

VIDEO_FPS = 30
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 540
VIDEO_CAPTURE_EVERY = 2
