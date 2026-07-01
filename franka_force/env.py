import csv

import mujoco
import mujoco.viewer
import numpy as np

from .config import (
    DEFAULT_CUSHION_THRESHOLD,
    DEFAULT_IMPEDANCE_DP,
    DEFAULT_IMPEDANCE_DR,
    DEFAULT_IMPEDANCE_KP,
    DEFAULT_IMPEDANCE_KR,
    DEFAULT_IMPEDANCE_TORQUE_LIMIT,
    DEFAULT_PEG_ALPHA,
    DEFAULT_SOCKET_ALPHA,
    FORCE_VISUAL_MODES,
    MODEL_PATH,
    RESULTS_DIR,
)
from .plotting import plot_force_comparison
from .recording import VideoRecorder
from .scenarios import SCENARIOS, get_scenario


class FrankaForceEnv:
    def __init__(
        self,
        scenario="hit_floor",
        interactive=False,
        force_feedback=False,
        force_visual="arrow",
        record_video=False,
        record_force_feedback=False,
        contact_cushion=False,
        cushion_threshold=DEFAULT_CUSHION_THRESHOLD,
        impedance_kp=DEFAULT_IMPEDANCE_KP,
        impedance_dp=DEFAULT_IMPEDANCE_DP,
        impedance_kr=DEFAULT_IMPEDANCE_KR,
        impedance_dr=DEFAULT_IMPEDANCE_DR,
        impedance_torque_limit=DEFAULT_IMPEDANCE_TORQUE_LIMIT,
        peg_alpha=DEFAULT_PEG_ALPHA,
        socket_alpha=DEFAULT_SOCKET_ALPHA,
    ):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}. Choose from {SCENARIOS}")
        if force_visual not in FORCE_VISUAL_MODES:
            raise ValueError(f"Unknown force visual: {force_visual}. Choose from {FORCE_VISUAL_MODES}")

        self.scenario = scenario
        self.scenario_impl = get_scenario(scenario)
        self.interactive = interactive
        self.force_feedback = force_feedback
        self.force_visual = force_visual
        self.record_video = record_video
        self.record_force_feedback = record_force_feedback
        self.contact_cushion = contact_cushion
        self.cushion_threshold = cushion_threshold
        self.impedance_kp = impedance_kp
        self.impedance_dp = impedance_dp
        self.impedance_kr = impedance_kr
        self.impedance_dr = impedance_dr
        self.impedance_torque_limit = impedance_torque_limit
        self.peg_alpha = peg_alpha
        self.socket_alpha = socket_alpha

        if force_feedback and not interactive:
            raise ValueError("force_feedback requires interactive=True")
        if record_force_feedback and not record_video:
            raise ValueError("record_force_feedback requires record_video=True")
        if record_force_feedback and scenario != "peg_in_hole":
            raise ValueError("record_force_feedback is only supported for peg_in_hole")
        if interactive and not self.scenario_impl.supports_interactive:
            raise ValueError("interactive mode is only supported for peg_in_hole")
        if contact_cushion and (scenario != "peg_in_hole" or not interactive):
            raise ValueError("contact_cushion requires scenario='peg_in_hole' and interactive=True")
        if cushion_threshold <= 0.0:
            raise ValueError("cushion_threshold must be positive")
        if impedance_kp < 0.0 or impedance_dp < 0.0 or impedance_kr < 0.0 or impedance_dr < 0.0:
            raise ValueError("impedance gains and damping values must be non-negative")
        if impedance_torque_limit <= 0.0:
            raise ValueError("impedance_torque_limit must be positive")
        if not 0.0 <= peg_alpha <= 1.0:
            raise ValueError("peg_alpha must be between 0.0 and 1.0")
        if not 0.0 <= socket_alpha <= 1.0:
            raise ValueError("socket_alpha must be between 0.0 and 1.0")

        self.results_dir = RESULTS_DIR / scenario
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.results_dir / "force_verification_log.csv"
        self.telemetry_filtered_path = self.results_dir / "force_verification_log_filtered.csv"
        self.plot_raw_path = self.results_dir / "force_comparison_raw.png"
        self.plot_filtered_path = self.results_dir / "force_comparison_filtered.png"
        self.plot_contact_raw_path = self.results_dir / "force_comparison_contact_only_raw.png"
        self.plot_contact_filtered_path = self.results_dir / "force_comparison_contact_only_filtered.png"
        self.video_path = self.results_dir / "run_recording.mp4"

        # Flag samples where ground truth diverges sharply from the Jacobian estimate.
        self.anomaly_ratio_high = 5.0
        self.anomaly_ratio_low = 0.2
        self.anomaly_abs_max = 1000.0
        self.min_est_for_ratio = 5.0

        # Storage lists for timeline telemetry
        self.time_history = []
        self.true_force_history = []
        self.estimated_force_history = []
        self.in_contact_history = []
        self.anomaly_history = []
        self.cushion_active_history = []
        self.cushion_scale_history = []
        self.impedance_tau_norm_history = []
        self.contact_force_vector_history = []

        self.step_counter = 0
        self.downsample_factor = 10
        self.latest_f_est = 0.0
        self.latest_f_true = 0.0
        self.latest_in_contact = False
        self.latest_contact_pos = None
        self.latest_contact_frame = None
        self.latest_contact_force = 0.0
        self.latest_contact_force_vector = np.zeros(3)
        self.latest_contact_arrow_pos = None
        self.latest_contact_arrow_vector = np.zeros(3)
        self.latest_contact_arrow_force = 0.0
        self.cushion_active = False
        self.cushion_scale = 0.0
        self.impedance_tau_norm = 0.0

        self.scenario_impl.initialize_state(self)

        # Telemetry CSV Setup
        self.log_file = open(self.telemetry_path, mode="w", newline="")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow([
            "Time (s)",
            "Ground Truth (N)",
            "Jacobian Estimate (N)",
            "In Contact",
            "Is Anomaly",
            "Cushion Active",
            "Cushion Scale",
            "Impedance Tau Norm",
            "Contact Force X (N)",
            "Contact Force Y (N)",
            "Contact Force Z (N)",
        ])

        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)

        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        self.hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.scenario_impl.resolve_ids(self)
        self.scenario_impl.after_model_init(self)

    def _build_model(self):
        """Load scene.xml from disk (so includes resolve), then inject scenario extras."""
        spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
        self.scenario_impl.augment_model_spec(self, spec)
        return spec.compile()

    def _apply_control_policy(self):
        """Factory Method: Changes how the arm moves depending on the task goal."""
        self.scenario_impl.apply_control(self)

    def _get_active_gripper_body_ids(self):
        """Returns the IDs of the active tool center contact surfaces."""
        gripper_names = ["hand", "left_finger", "right_finger"]
        return [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in gripper_names]

    def _contact_force_in_world(self, contact_idx):
        """Contact force in world frame (MuJoCo contact frame -> global)."""
        contact = self.data.contact[contact_idx]
        c_forces = np.zeros(6)
        mujoco.mj_contactForce(self.model, self.data, contact_idx, c_forces)
        frame = contact.frame.reshape(3, 3)
        return frame.T @ c_forces[:3]

    def _has_target_contact(self, gripper_ids):
        for i in range(self.data.ncon):
            if self.scenario_impl.is_target_contact(self, self.data.contact[i], gripper_ids):
                return True
        return False

    def _calculate_ground_truth_force(self, gripper_ids):
        """Sum target contact forces in world frame, then return magnitude."""
        force_world = np.zeros(3)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if not self.scenario_impl.is_target_contact(self, contact, gripper_ids):
                continue
            force_world += self._contact_force_in_world(i)
        return float(np.linalg.norm(force_world))

    def _hand_jacobian_arm_cols(self):
        """6x7 spatial Jacobian at the hand body origin (arm joints only)."""
        jac_p = np.zeros((3, self.model.nv))
        jac_r = np.zeros((3, self.model.nv))
        mujoco.mj_jac(
            self.model, self.data, jac_p, jac_r,
            self.data.xpos[self.hand_body_id], self.hand_body_id,
        )
        return np.vstack([jac_p, jac_r])[:, :7]

    def _estimate_virtual_force(self):
        """Map constraint (contact) joint forces to Cartesian force at the hand."""
        tau_contact = self.data.qfrc_constraint[:7]
        if not np.any(np.abs(tau_contact) > 1e-9):
            return 0.0

        J = self._hand_jacobian_arm_cols()
        wrench_estimated = np.linalg.pinv(J.T) @ tau_contact
        return float(np.linalg.norm(wrench_estimated[:3]))

    def _sample_forces(self):
        return self.scenario_impl.sample_forces(self)

    def _is_anomaly(self, f_true, f_est, in_contact):
        """Heuristic outlier flag for multi-contact spikes and vector-cancellation dips."""
        if f_true > self.anomaly_abs_max:
            return True
        if not in_contact:
            return f_true > 1.0
        if f_est < self.min_est_for_ratio:
            return f_true > 50.0

        ratio = f_true / max(f_est, 1e-9)
        return ratio > self.anomaly_ratio_high or ratio < self.anomaly_ratio_low

    def _record_telemetry(self):
        """Sample ground-truth and estimated forces."""
        if self.step_counter % self.downsample_factor == 0:
            in_contact, f_true, f_est = self._sample_forces()
            is_anomaly = self._is_anomaly(f_true, f_est, in_contact)

            self.time_history.append(self.data.time)
            self.true_force_history.append(f_true)
            self.estimated_force_history.append(f_est)
            self.in_contact_history.append(in_contact)
            self.anomaly_history.append(is_anomaly)
            self.cushion_active_history.append(self.cushion_active)
            self.cushion_scale_history.append(self.cushion_scale)
            self.impedance_tau_norm_history.append(self.impedance_tau_norm)
            contact_force_vector = np.asarray(self.latest_contact_force_vector, dtype=float).copy()
            self.contact_force_vector_history.append(contact_force_vector)

            self.log_writer.writerow([
                self.data.time,
                f_true,
                f_est,
                int(in_contact),
                int(is_anomaly),
                int(self.cushion_active),
                self.cushion_scale,
                self.impedance_tau_norm,
                contact_force_vector[0],
                contact_force_vector[1],
                contact_force_vector[2],
            ])

        self.step_counter += 1

    def _update_live_force(self):
        in_contact, f_true, f_est = self._sample_forces()
        self.latest_in_contact = in_contact
        self.latest_f_true = f_true
        self.latest_f_est = f_est

    def _force_feedback_magnitude(self):
        return max(self.latest_f_est, self.latest_f_true)

    def _force_feedback_overlay_enabled(self):
        return self.force_feedback or self.record_force_feedback

    def _apply_control_policy_callback(self, model, data):
        self._apply_control_policy()

    def _passive_callback(self, model, data):
        self._record_telemetry()

    def _run_passive_viewer(self, interactive=False):
        if interactive:
            self.scenario_impl.print_controls(self)
            self.scenario_impl.start_interactive(self)

        recorder = None
        if self.record_video:
            recorder = VideoRecorder(self.model, self.video_path)
            recorder.start()
            print(f"Recording video → {self.video_path.resolve()}")

        mujoco.set_mjcb_control(self._apply_control_policy_callback)
        substeps = 3 if interactive else 1
        key_callback = None
        if interactive:
            key_callback = lambda keycode: self.scenario_impl.viewer_key_callback(self, keycode)

        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                key_callback=key_callback,
                show_left_ui=False,
                show_right_ui=False,
            ) as viewer:
                while viewer.is_running():
                    if interactive:
                        self.scenario_impl.before_interactive_step(self, self.model.opt.timestep)

                    for _ in range(substeps):
                        mujoco.mj_step(self.model, self.data)

                    if interactive or self.record_force_feedback:
                        self._update_live_force()
                    self._record_telemetry()

                    if interactive:
                        self.scenario_impl.update_interactive_viewer(self, viewer)

                    if recorder is not None:
                        overlay_callback = None
                        if self.record_force_feedback:
                            overlay_callback = (
                                lambda scene: self.scenario_impl.update_recording_scene(self, scene)
                            )
                        recorder.capture(self.data, viewer.cam, overlay_callback=overlay_callback)

                    viewer.sync()
        except RuntimeError as exc:
            if "mjpython" in str(exc).lower():
                hint = (
                    "Passive viewer (required for --record-video) needs `mjpython` on macOS.\n"
                    "  Run: mjpython main.py --scenario peg_in_hole --interactive --record-video"
                )
                if not interactive:
                    hint = (
                        "Video recording uses the passive viewer and needs `mjpython` on macOS.\n"
                        "  Run: mjpython main.py --scenario push_block --record-video"
                    )
                raise RuntimeError(hint) from exc
            raise
        finally:
            if recorder is not None:
                recorder.close()
            if interactive:
                self.scenario_impl.stop_interactive(self)
            mujoco.set_mjcb_control(None)

    def _run_standard_viewer(self):
        mujoco.set_mjcb_control(self._apply_control_policy_callback)
        mujoco.set_mjcb_passive(self._passive_callback)
        try:
            mujoco.viewer.launch(self.model, self.data)
        except RuntimeError as exc:
            raise RuntimeError(
                "MuJoCo viewer failed to start. Try: (1) close any stuck "
                "simulator windows, (2) run from Terminal.app instead of an "
                "embedded shell, (3) run `pkill -f 'python.*main.py'`, then "
                "retry."
            ) from exc
        finally:
            mujoco.set_mjcb_passive(None)
            mujoco.set_mjcb_control(None)

    def _run_interactive_viewer(self):
        self._run_passive_viewer(interactive=True)

    def run(self):
        print(f"Booting up environment factory running: [{self.scenario.upper()}]")

        if self.interactive:
            self._run_interactive_viewer()
        elif self.record_video:
            self._run_passive_viewer(interactive=False)
        else:
            self._run_standard_viewer()

        self.log_file.close()
        self.plot_comparison()

    def plot_comparison(self):
        plot_force_comparison(self)

    def __del__(self):
        try:
            if hasattr(self, "log_file") and not self.log_file.closed:
                self.log_file.close()
        except Exception:
            pass
