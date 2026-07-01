import argparse

from franka_force import FORCE_VISUAL_MODES, SCENARIOS
from franka_force.config import (
    DEFAULT_CUSHION_THRESHOLD,
    DEFAULT_IMPEDANCE_DP,
    DEFAULT_IMPEDANCE_DR,
    DEFAULT_IMPEDANCE_KP,
    DEFAULT_IMPEDANCE_KR,
    DEFAULT_IMPEDANCE_TORQUE_LIMIT,
    DEFAULT_PEG_ALPHA,
    DEFAULT_SOCKET_ALPHA,
)


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
        help="Force feedback visual to show when live or recorded feedback is enabled",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Save run_recording.mp4 in results/<scenario>/ (uses passive viewer; mjpython on macOS)",
    )
    parser.add_argument(
        "--record-force-feedback",
        action="store_true",
        help="Include force feedback overlay geoms in --record-video output (peg_in_hole only)",
    )
    parser.add_argument(
        "--contact-cushion",
        action="store_true",
        help="Enable experimental impedance cushion (peg_in_hole + --interactive only)",
    )
    parser.add_argument(
        "--cushion-threshold",
        type=float,
        default=DEFAULT_CUSHION_THRESHOLD,
        help="Contact force threshold in newtons that activates --contact-cushion",
    )
    parser.add_argument(
        "--impedance-kp",
        type=float,
        default=DEFAULT_IMPEDANCE_KP,
        help="Cartesian translational stiffness for --contact-cushion",
    )
    parser.add_argument(
        "--impedance-dp",
        type=float,
        default=DEFAULT_IMPEDANCE_DP,
        help="Cartesian translational damping for --contact-cushion",
    )
    parser.add_argument(
        "--impedance-kr",
        type=float,
        default=DEFAULT_IMPEDANCE_KR,
        help="Cartesian rotational stiffness for --contact-cushion",
    )
    parser.add_argument(
        "--impedance-dr",
        type=float,
        default=DEFAULT_IMPEDANCE_DR,
        help="Cartesian rotational damping for --contact-cushion",
    )
    parser.add_argument(
        "--impedance-torque-limit",
        type=float,
        default=DEFAULT_IMPEDANCE_TORQUE_LIMIT,
        help="Per-joint torque clamp for --contact-cushion",
    )
    parser.add_argument(
        "--peg-alpha",
        type=float,
        default=DEFAULT_PEG_ALPHA,
        help="Peg opacity for peg_in_hole, from 0.0 transparent to 1.0 opaque",
    )
    parser.add_argument(
        "--socket-alpha",
        type=float,
        default=DEFAULT_SOCKET_ALPHA,
        help="Socket wall opacity for peg_in_hole, from 0.0 transparent to 1.0 opaque",
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
        record_force_feedback=args.record_force_feedback,
        contact_cushion=args.contact_cushion,
        cushion_threshold=args.cushion_threshold,
        impedance_kp=args.impedance_kp,
        impedance_dp=args.impedance_dp,
        impedance_kr=args.impedance_kr,
        impedance_dr=args.impedance_dr,
        impedance_torque_limit=args.impedance_torque_limit,
        peg_alpha=args.peg_alpha,
        socket_alpha=args.socket_alpha,
    )
    env.run()
