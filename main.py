import argparse

from franka_force import FORCE_VISUAL_MODES, SCENARIOS


def parse_args():
    parser = argparse.ArgumentParser(description="Franka force verification scenarios")
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="push_block",
        help="Simulation scenario to run",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable keyboard control (peg_in_hole only; requires mjpython on macOS)",
    )
    parser.add_argument(
        "--force-feedback",
        action="store_true",
        help="Enable live force visual overlay (peg_in_hole + --interactive only)",
    )
    parser.add_argument(
        "--force-visual",
        choices=FORCE_VISUAL_MODES,
        default="arrow",
        help="Force feedback visual to show when --force-feedback is enabled",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Save run_recording.mp4 in results/<scenario>/ (uses passive viewer; mjpython on macOS)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from franka_force import FrankaForceEnv

    env = FrankaForceEnv(
        scenario=args.scenario,
        interactive=args.interactive,
        force_feedback=args.force_feedback,
        force_visual=args.force_visual,
        record_video=args.record_video,
    )
    env.run()
