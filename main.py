import argparse
import csv
import threading
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None

MODEL_PATH = Path('mujoco_menagerie/franka_emika_panda/scene.xml')
RESULTS_DIR = Path('results')
SCENARIOS = ("hit_floor", "push_block", "peg_in_hole")
VIDEO_FPS = 30
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 540
VIDEO_CAPTURE_EVERY = 2


class VideoRecorder:
    """Stream offscreen frames to mp4 (uses the same camera as the passive viewer)."""

    def __init__(self, model, path, fps=VIDEO_FPS, width=VIDEO_WIDTH, height=VIDEO_HEIGHT):
        self.path = Path(path)
        self.fps = fps
        self.capture_every = VIDEO_CAPTURE_EVERY
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(model, height, width)
        self._writer = None
        self._frame_counter = 0
        self._saved_frames = 0

    def start(self):
        try:
            import imageio
        except ImportError as exc:
            raise RuntimeError(
                "Video recording requires imageio. Install with: pip install imageio imageio-ffmpeg"
            ) from exc

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(
            str(self.path),
            fps=self.fps,
            macro_block_size=1,
        )

    def capture(self, data, camera):
        if self._writer is None:
            return

        self._frame_counter += 1
        if (self._frame_counter - 1) % self.capture_every != 0:
            return

        self.renderer.update_scene(data, camera=camera)
        frame = self.renderer.render()
        self._writer.append_data(frame)
        self._saved_frames += 1

    def close(self):
        if self._writer is None:
            return

        self._writer.close()
        self._writer = None
        if self._saved_frames == 0:
            print("No video frames captured; skipping video save.")
            if self.path.exists():
                self.path.unlink()
            return

        print(
            f"Saved run video ({self._saved_frames} frames @ {self.fps} fps) "
            f"to {self.path.resolve()}"
        )


class FrankaForceEnv:
    def __init__(self, scenario="hit_floor", interactive=False, force_feedback=False, record_video=False):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}. Choose from {SCENARIOS}")

        self.scenario = scenario
        self.interactive = interactive
        self.force_feedback = force_feedback
        self.record_video = record_video

        if force_feedback and not interactive:
            raise ValueError("force_feedback requires interactive=True")
        if interactive and scenario != "peg_in_hole":
            raise ValueError("interactive mode is only supported for peg_in_hole")

        self.results_dir = RESULTS_DIR / scenario
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.results_dir / 'force_verification_log.csv'
        self.telemetry_filtered_path = self.results_dir / 'force_verification_log_filtered.csv'
        self.plot_raw_path = self.results_dir / 'force_comparison_raw.png'
        self.plot_filtered_path = self.results_dir / 'force_comparison_filtered.png'
        self.plot_contact_raw_path = self.results_dir / 'force_comparison_contact_only_raw.png'
        self.plot_contact_filtered_path = self.results_dir / 'force_comparison_contact_only_filtered.png'
        self.video_path = self.results_dir / 'run_recording.mp4'

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

        self.step_counter = 0
        self.downsample_factor = 10
        self.latest_f_est = 0.0
        self.latest_f_true = 0.0
        self.latest_in_contact = False

        if scenario == "peg_in_hole":
            self.target_pos = np.zeros(3)
            self.target_roll = 0.0
            self.teleop_speed = 0.10
            self.roll_speed = 0.8
            self.gripper_closed = False
            self._teleop_lock = threading.Lock()
            self._move_cmd = np.zeros(3)
            self._roll_cmd = 0.0
            self._keyboard_listener = None
            self._peg_home_q = np.array([0.0, 0.229, 0.0, -1.80, 0.0, 2.25, 0.80])
            self._peg_down = np.array([0.0, 0.0, -1.0])

        # Telemetry CSV Setup
        self.log_file = open(self.telemetry_path, mode="w", newline="")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow([
            "Time (s)", "Ground Truth (N)", "Jacobian Estimate (N)", "In Contact", "Is Anomaly"
        ])

        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)

        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        self.hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self.scenario == "push_block":
            self.block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        elif self.scenario == "peg_in_hole":
            self.peg_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom")
            self.ik_target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ik_target")
            self.ik_target_mocap_id = self.model.body_mocapid[self.ik_target_body_id]
            if self.interactive:
                self._boost_peg_actuators()
            self._init_peg_home_pose()

    def _build_model(self):
        """Load scene.xml from disk (so includes resolve), then inject scenario extras."""
        spec = mujoco.MjSpec.from_file(str(MODEL_PATH))

        if self.scenario == "push_block":
            body = spec.worldbody.add_body(name="target_block", pos=[0.55, 0.0, 0.03])
            body.add_freejoint()
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.08, 0.08, 0.06],
                mass=5.0,
                rgba=[1, 0, 0, 1],
                condim=3,
                friction=[1, 0.005, 0.0001],
            )
        elif self.scenario == "peg_in_hole":
            hand_body = spec.body("hand")
            hand_body.add_geom(
                name="peg_geom",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.012, 0.05],
                pos=[0, 0, 0.10],
                rgba=[0.8, 0.8, 0.8, 1],
                mass=0.2,
                condim=3,
            )

            socket_base = spec.worldbody.add_body(name="socket", pos=[0.50, 0.0, 0.0])
            wall_thick = 0.015
            wall_len = 0.05
            wall_height = 0.04
            hole_gap = 0.016

            socket_base.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[wall_thick, wall_len, wall_height],
                pos=[-hole_gap - wall_thick, 0, wall_height],
                rgba=[0.4, 0.4, 0.4, 1],
            )
            socket_base.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[wall_thick, wall_len, wall_height],
                pos=[hole_gap + wall_thick, 0, wall_height],
                rgba=[0.4, 0.4, 0.4, 1],
            )
            socket_base.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[wall_len, wall_thick, wall_height],
                pos=[0, -hole_gap - wall_thick, wall_height],
                rgba=[0.4, 0.4, 0.4, 1],
            )
            socket_base.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[wall_len, wall_thick, wall_height],
                pos=[0, hole_gap + wall_thick, wall_height],
                rgba=[0.4, 0.4, 0.4, 1],
            )

            ik_target = spec.worldbody.add_body(name="ik_target", mocap=True)
            ik_target.add_geom(
                name="ik_target_geom",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.015],
                rgba=[0.1, 0.9, 0.2, 0.55],
                contype=0,
                conaffinity=0,
            )

            self._add_floor_compass(spec, origin=[0.38, 0.0, 0.0])

        return spec.compile()

    def _add_floor_compass(self, spec, origin):
        """World-frame N/E/S/W arrows on the floor (teleop uses world +X/+Y, not camera axes)."""
        base = spec.worldbody.add_body(name="floor_compass", pos=list(origin))
        z = 0.002
        arm = 0.11
        thick = 0.007
        decal = dict(contype=0, conaffinity=0)

        # +X = East (Right arrow in teleop)
        base.add_geom(
            name="compass_e",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[arm / 2, thick, 0.001],
            pos=[arm / 2, 0, z],
            rgba=[0.95, 0.25, 0.2, 0.95],
            **decal,
        )
        base.add_geom(
            name="compass_e_tip",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[thick * 1.4, thick * 2.2, 0.001],
            pos=[arm - thick, 0, z],
            rgba=[0.95, 0.25, 0.2, 0.95],
            **decal,
        )
        # -X = West (Left arrow)
        base.add_geom(
            name="compass_w",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[arm / 2, thick, 0.001],
            pos=[-arm / 2, 0, z],
            rgba=[0.55, 0.15, 0.12, 0.85],
            **decal,
        )
        # +Y = North (Up arrow)
        base.add_geom(
            name="compass_n",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[thick, arm / 2, 0.001],
            pos=[0, arm / 2, z],
            rgba=[0.2, 0.35, 0.95, 0.95],
            **decal,
        )
        base.add_geom(
            name="compass_n_tip",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[thick * 2.2, thick * 1.4, 0.001],
            pos=[0, arm - thick, z],
            rgba=[0.2, 0.35, 0.95, 0.95],
            **decal,
        )
        # -Y = South (Down arrow)
        base.add_geom(
            name="compass_s",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[thick, arm / 2, 0.001],
            pos=[0, -arm / 2, z],
            rgba=[0.15, 0.55, 0.85, 0.85],
            **decal,
        )
        base.add_geom(
            name="compass_hub",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.014],
            pos=[0, 0, z + 0.001],
            rgba=[0.95, 0.95, 0.95, 0.9],
            **decal,
        )
        label_z = 0.004
        label_size = 0.018
        for name, pos, rgba in (
            ("compass_label_e", [arm + 0.03, 0, label_z], [0.95, 0.25, 0.2, 1]),
            ("compass_label_w", [-arm - 0.03, 0, label_z], [0.55, 0.15, 0.12, 1]),
            ("compass_label_n", [0, arm + 0.03, label_z], [0.2, 0.35, 0.95, 1]),
            ("compass_label_s", [0, -arm - 0.03, label_z], [0.15, 0.55, 0.85, 1]),
        ):
            base.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[label_size, label_size, 0.001],
                pos=pos,
                rgba=rgba,
                **decal,
            )

    def _apply_control_policy(self):
        """Factory Method: Changes how the arm moves depending on the task goal"""
        if self.scenario == "hit_floor":
            self.data.ctrl[1] = 0.229
            self.data.ctrl[3] = -2.20
            self.data.ctrl[5] = 2.30
            self.data.ctrl[6] = 0.80
            self.data.ctrl[7] = 255

        elif self.scenario == "push_block":
            if self.data.time < 1.5:
                self.data.ctrl[1] = 0.229
                self.data.ctrl[3] = -1.80
                self.data.ctrl[5] = 1.87
                self.data.ctrl[6] = 0.80
                self.data.ctrl[7] = 255
            elif self.data.time < 3.0:
                progress = (self.data.time - 1.5) / 1.5
                self.data.ctrl[1] = 0.229
                self.data.ctrl[3] = -1.80 + progress * (-2.37 - (-1.80))
                self.data.ctrl[5] = 1.87 + progress * (2.25 - 1.87)
                self.data.ctrl[6] = 0.80
                self.data.ctrl[7] = 255
            elif self.data.time < 6.0:
                progress = (self.data.time - 3.0) / 3.0
                self.data.ctrl[3] = -2.37 + progress * (-2.05 - (-2.37))
                self.data.ctrl[1] = 0.229 + progress * (0.420 - 0.229)
                self.data.ctrl[5] = 2.25
                self.data.ctrl[6] = 0.80
                self.data.ctrl[7] = 255
            else:
                self.data.ctrl[1] = 0.420
                self.data.ctrl[3] = -2.05
                self.data.ctrl[5] = 2.25
                self.data.ctrl[6] = 0.80
                self.data.ctrl[7] = 255

        elif self.scenario == "peg_in_hole":
            self._apply_peg_ik_control()

    def _init_peg_home_pose(self):
        """Place the arm over the socket and sync the IK target to the current hand pose."""
        self.data.qpos[:7] = self._peg_home_q
        mujoco.mj_forward(self.model, self.data)
        for i in range(7):
            self.data.ctrl[i] = self._peg_home_q[i]
        self.data.ctrl[7] = 0.0
        with self._teleop_lock:
            self.target_pos = self.data.xpos[self.hand_body_id].copy()
        self._sync_target_marker()

    def _boost_peg_actuators(self):
        """Stiffen arm servos so interactive IK targets are tracked quickly."""
        for i in range(7):
            self.model.actuator_gainprm[i, 0] *= 10.0
            self.model.actuator_biasprm[i, 1] *= 10.0
            self.model.actuator_biasprm[i, 2] *= 10.0

    def _target_hand_rotmat(self, roll):
        """Hand frame with peg axis (+Z) pointing down; roll spins peg about vertical."""
        z_des = self._peg_down
        x_des = np.array([np.cos(roll), np.sin(roll), 0.0])
        y_des = np.cross(z_des, x_des)
        return np.column_stack([x_des, y_des, z_des])

    def _orientation_error(self, current_rot, target_rot):
        rot_err = target_rot @ current_rot.T
        return 0.5 * np.array([
            rot_err[2, 1] - rot_err[1, 2],
            rot_err[0, 2] - rot_err[2, 0],
            rot_err[1, 0] - rot_err[0, 1],
        ])

    def _solve_peg_ik(self, target_pos, target_roll):
        """6-DOF iterative IK: reach target_pos with peg axis pointing down."""
        saved_qpos = self.data.qpos.copy()
        q_cmd = saved_qpos[:7].copy()
        target_rot = self._target_hand_rotmat(target_roll)
        dls_lambda = 0.025
        max_dq = 0.10
        pos_step_cap = 0.18
        ori_step_cap = 0.35
        pos_weight = 1.0
        ori_weight = 2.5

        try:
            for _ in range(24):
                self.data.qpos[:7] = q_cmd
                mujoco.mj_kinematics(self.model, self.data)
                ee = self.data.xpos[self.hand_body_id].copy()
                current_rot = self.data.xmat[self.hand_body_id].reshape(3, 3).copy()

                pos_err = target_pos - ee
                ori_err = self._orientation_error(current_rot, target_rot)
                pos_norm = np.linalg.norm(pos_err)
                ori_norm = np.linalg.norm(ori_err)
                if pos_norm < 3e-3 and ori_norm < 0.04:
                    break

                if pos_norm > 1e-6:
                    pos_step = pos_err / pos_norm * min(pos_norm, pos_step_cap)
                else:
                    pos_step = np.zeros(3)
                if ori_norm > 1e-6:
                    ori_step = ori_err / ori_norm * min(ori_norm, ori_step_cap)
                else:
                    ori_step = np.zeros(3)

                task_err = np.concatenate([pos_weight * pos_step, ori_weight * ori_step])
                jac_p = np.zeros((3, self.model.nv))
                jac_r = np.zeros((3, self.model.nv))
                mujoco.mj_jac(
                    self.model, self.data, jac_p, jac_r, ee, self.hand_body_id,
                )
                j_arm = np.vstack([pos_weight * jac_p[:, :7], ori_weight * jac_r[:, :7]])
                dq = j_arm.T @ np.linalg.solve(
                    j_arm @ j_arm.T + dls_lambda ** 2 * np.eye(6),
                    task_err,
                )
                q_cmd += np.clip(dq, -max_dq, max_dq)
        finally:
            self.data.qpos[:] = saved_qpos
            mujoco.mj_kinematics(self.model, self.data)

        return q_cmd

    def _solve_peg_ik_pos_only(self, target_pos):
        """3-DOF fallback for non-interactive peg mode."""
        saved_qpos = self.data.qpos.copy()
        q_cmd = saved_qpos[:7].copy()
        dls_lambda = 0.025
        max_dq = 0.10
        cart_step_cap = 0.18

        try:
            for _ in range(20):
                self.data.qpos[:7] = q_cmd
                mujoco.mj_kinematics(self.model, self.data)
                ee = self.data.xpos[self.hand_body_id].copy()
                error = target_pos - ee
                error_norm = np.linalg.norm(error)
                if error_norm < 3e-3:
                    break

                step_error = error / max(error_norm, 1e-6) * min(error_norm, cart_step_cap)
                jac_p = np.zeros((3, self.model.nv))
                jac_r = np.zeros((3, self.model.nv))
                mujoco.mj_jac(
                    self.model, self.data, jac_p, jac_r, ee, self.hand_body_id,
                )
                j_arm = jac_p[:, :7]
                dq = j_arm.T @ np.linalg.solve(
                    j_arm @ j_arm.T + dls_lambda ** 2 * np.eye(3),
                    step_error,
                )
                q_cmd += np.clip(dq, -max_dq, max_dq)
        finally:
            self.data.qpos[:] = saved_qpos
            mujoco.mj_kinematics(self.model, self.data)

        return q_cmd

    def _apply_peg_ik_control(self):
        """IK toward target pose (interactive: 6-DOF peg-down + roll)."""
        with self._teleop_lock:
            target_pos = self.target_pos.copy()
            target_roll = self.target_roll
            gripper_closed = self.gripper_closed

        if self.interactive:
            q_des = self._solve_peg_ik(target_pos, target_roll)
        else:
            q_des = self._solve_peg_ik_pos_only(target_pos)

        for i in range(7):
            self.data.ctrl[i] = q_des[i]
        self.data.ctrl[7] = 255.0 if gripper_closed else 0.0

    def _get_active_gripper_body_ids(self):
        """Returns the IDs of the active tool center contact surfaces."""
        gripper_names = ["hand", "left_finger", "right_finger"]
        return [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in gripper_names]

    def _is_target_contact(self, contact, gripper_ids):
        """True when this contact is gripper↔floor or gripper↔block (scenario-specific)."""
        body1 = self.model.geom_bodyid[contact.geom1]
        body2 = self.model.geom_bodyid[contact.geom2]
        gripper_on_1 = body1 in gripper_ids
        gripper_on_2 = body2 in gripper_ids

        if self.scenario == "hit_floor":
            floor_on_1 = contact.geom1 == self.floor_geom_id
            floor_on_2 = contact.geom2 == self.floor_geom_id
            return (gripper_on_1 and floor_on_2) or (gripper_on_2 and floor_on_1)

        block_on_1 = body1 == self.block_id
        block_on_2 = body2 == self.block_id
        return (gripper_on_1 and block_on_2) or (gripper_on_2 and block_on_1)

    def _has_peg_contact(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == self.peg_geom_id or contact.geom2 == self.peg_geom_id:
                return True
        return False

    def _contact_force_in_world(self, contact_idx):
        """Contact force in world frame (MuJoCo contact frame → global)."""
        contact = self.data.contact[contact_idx]
        c_forces = np.zeros(6)
        mujoco.mj_contactForce(self.model, self.data, contact_idx, c_forces)
        frame = contact.frame.reshape(3, 3)
        return frame.T @ c_forces[:3]

    def _has_target_contact(self, gripper_ids):
        for i in range(self.data.ncon):
            if self._is_target_contact(self.data.contact[i], gripper_ids):
                return True
        return False

    def _calculate_ground_truth_force(self, gripper_ids):
        """Sum target contact forces in world frame, then return magnitude."""
        force_world = np.zeros(3)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if not self._is_target_contact(contact, gripper_ids):
                continue
            force_world += self._contact_force_in_world(i)
        return float(np.linalg.norm(force_world))

    def _calculate_peg_ground_truth_force(self):
        force_world = np.zeros(3)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 != self.peg_geom_id and contact.geom2 != self.peg_geom_id:
                continue
            force_world += self._contact_force_in_world(i)
        return float(np.linalg.norm(force_world))

    def _hand_jacobian_arm_cols(self):
        """6×7 spatial Jacobian at the hand body origin (arm joints only)."""
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
        if self.scenario == "peg_in_hole":
            in_contact = self._has_peg_contact()
            f_true = self._calculate_peg_ground_truth_force()
        else:
            gripper_ids = self._get_active_gripper_body_ids()
            in_contact = self._has_target_contact(gripper_ids)
            f_true = self._calculate_ground_truth_force(gripper_ids)

        f_est = self._estimate_virtual_force() if in_contact else 0.0
        return in_contact, f_true, f_est

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

            self.log_writer.writerow([self.data.time, f_true, f_est, int(in_contact), int(is_anomaly)])

        self.step_counter += 1

    def _update_live_force(self):
        in_contact, f_true, f_est = self._sample_forces()
        self.latest_in_contact = in_contact
        self.latest_f_true = f_true
        self.latest_f_est = f_est

    def _force_feedback_magnitude(self):
        return max(self.latest_f_est, self.latest_f_true)

    def _sync_target_marker(self):
        with self._teleop_lock:
            target = self.target_pos.copy()
        self.data.mocap_pos[self.ik_target_mocap_id] = target
        self.data.mocap_quat[self.ik_target_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])

    def _nudge_target(self, dx=0.0, dy=0.0, dz=0.0):
        step = 0.02
        with self._teleop_lock:
            self.target_pos[0] += dx * step
            self.target_pos[1] += dy * step
            self.target_pos[2] += dz * step

    def _set_gripper(self, closed):
        with self._teleop_lock:
            self.gripper_closed = closed

    def _nudge_roll(self, delta):
        step = 0.08
        with self._teleop_lock:
            self.target_roll += delta * step

    def _adjust_roll_cmd(self, delta):
        with self._teleop_lock:
            self._roll_cmd = np.clip(self._roll_cmd + delta, -1.0, 1.0)

    def _apply_teleop_motion(self, dt):
        with self._teleop_lock:
            move_cmd = self._move_cmd.copy()
            roll_cmd = self._roll_cmd
        if np.any(move_cmd != 0) or roll_cmd != 0:
            with self._teleop_lock:
                if np.any(move_cmd != 0):
                    self.target_pos += move_cmd * self.teleop_speed * dt
                if roll_cmd != 0:
                    self.target_roll += roll_cmd * self.roll_speed * dt

    def _adjust_move_cmd(self, axis, delta):
        with self._teleop_lock:
            self._move_cmd[axis] = np.clip(self._move_cmd[axis] + delta, -1.0, 1.0)

    def _viewer_key_callback(self, keycode):
        """Nudge the IK target from the MuJoCo window (one step per key press)."""
        if self.scenario != "peg_in_hole" or not self.interactive:
            return

        # GLFW key codes — avoid I/J/K/U; those toggle MuJoCo debug overlays.
        if keycode == 265:      # Up arrow -> North (+Y)
            self._nudge_target(dy=1.0)
        elif keycode == 264:    # Down arrow -> South (-Y)
            self._nudge_target(dy=-1.0)
        elif keycode == 262:    # Right arrow -> East (+X)
            self._nudge_target(dx=1.0)
        elif keycode == 263:    # Left arrow -> West (-X)
            self._nudge_target(dx=-1.0)
        elif keycode == 266:    # Page Up -> +Z
            self._nudge_target(dz=1.0)
        elif keycode == 267:    # Page Down -> -Z
            self._nudge_target(dz=-1.0)
        elif keycode in (57,):  # 9 raise (Z+)
            self._nudge_target(dz=1.0)
        elif keycode in (56,):  # 8 lower (Z-)
            self._nudge_target(dz=-1.0)
        elif keycode in (44,):  # , open gripper
            self._set_gripper(False)
        elif keycode in (46,):  # . close gripper
            self._set_gripper(True)
        elif keycode in (54,):  # 6 roll CCW
            self._nudge_roll(-1.0)
        elif keycode in (55,):  # 7 roll CW
            self._nudge_roll(1.0)

    def _start_pynput_teleop(self):
        if pynput_keyboard is None:
            print("Note: install pynput for smoother hold-to-move teleop (pip install pynput).")
            return

        def on_press(key):
            try:
                if key == pynput_keyboard.Key.up:
                    self._adjust_move_cmd(1, 1.0)
                elif key == pynput_keyboard.Key.down:
                    self._adjust_move_cmd(1, -1.0)
                elif key == pynput_keyboard.Key.right:
                    self._adjust_move_cmd(0, 1.0)
                elif key == pynput_keyboard.Key.left:
                    self._adjust_move_cmd(0, -1.0)
                elif key == pynput_keyboard.Key.page_up:
                    self._adjust_move_cmd(2, 1.0)
                elif key == pynput_keyboard.Key.page_down:
                    self._adjust_move_cmd(2, -1.0)
                elif hasattr(key, "char") and key.char == "9":
                    self._adjust_move_cmd(2, 1.0)
                elif hasattr(key, "char") and key.char == "8":
                    self._adjust_move_cmd(2, -1.0)
                elif hasattr(key, "char") and key.char == ",":
                    self._set_gripper(False)
                elif hasattr(key, "char") and key.char == ".":
                    self._set_gripper(True)
                elif hasattr(key, "char") and key.char == "6":
                    self._adjust_roll_cmd(-1.0)
                elif hasattr(key, "char") and key.char == "7":
                    self._adjust_roll_cmd(1.0)
            except Exception:
                pass

        def on_release(key):
            try:
                if key == pynput_keyboard.Key.up:
                    self._adjust_move_cmd(1, -1.0)
                elif key == pynput_keyboard.Key.down:
                    self._adjust_move_cmd(1, 1.0)
                elif key == pynput_keyboard.Key.right:
                    self._adjust_move_cmd(0, -1.0)
                elif key == pynput_keyboard.Key.left:
                    self._adjust_move_cmd(0, 1.0)
                elif key == pynput_keyboard.Key.page_up:
                    self._adjust_move_cmd(2, -1.0)
                elif key == pynput_keyboard.Key.page_down:
                    self._adjust_move_cmd(2, 1.0)
                elif hasattr(key, "char") and key.char == "9":
                    self._adjust_move_cmd(2, -1.0)
                elif hasattr(key, "char") and key.char == "8":
                    self._adjust_move_cmd(2, 1.0)
                elif hasattr(key, "char") and key.char == "6":
                    self._adjust_roll_cmd(1.0)
                elif hasattr(key, "char") and key.char == "7":
                    self._adjust_roll_cmd(-1.0)
            except Exception:
                pass

        self._keyboard_listener = pynput_keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
        )
        self._keyboard_listener.start()

    def _stop_pynput_teleop(self):
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        with self._teleop_lock:
            self._move_cmd[:] = 0.0
            self._roll_cmd = 0.0

    def _update_peg_hud(self, viewer):
        with self._teleop_lock:
            target = self.target_pos.copy()
            gripper = "closed" if self.gripper_closed else "open"
            moving = np.any(self._move_cmd != 0) or self._roll_cmd != 0
            roll_deg = np.degrees(self.target_roll)
        force_line = ""
        if self.force_feedback:
            f_display = self._force_feedback_magnitude()
            force_line = (
                f"force {f_display:.1f} N"
                + (" (contact)" if self.latest_in_contact else " (no contact yet)")
            )
        viewer.set_texts([
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "Arrows=N/S/E/W | 9/8=Z | 6/7=roll | ,/.=gripper | peg locked down",
                f"target ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})  roll {roll_deg:.0f}°  {gripper}"
                + ("  MOVING" if moving else "")
                + (f"  |  {force_line}" if force_line else ""),
            ),
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                "Compass on floor: blue=N (Up arrow)  red=E (Right arrow)",
                "",
            ),
        ])

    def _set_user_geom(self, geom, gtype, size, pos, mat, rgba):
        mujoco.mjv_initGeom(geom, gtype, size, pos, mat, rgba)
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR

    def _draw_force_feedback(self, viewer):
        if not self.force_feedback or viewer.user_scn is None:
            return

        f_display = self._force_feedback_magnitude()
        hand_pos = self.data.xpos[self.hand_body_id].astype(np.float64)
        identity = np.eye(3, dtype=np.float64).reshape(9, 1)
        zero3 = np.zeros((3, 1), dtype=np.float64)

        viewer.user_scn.ngeom = 0
        idx = 0

        base_pos = (hand_pos + np.array([0.0, 0.0, 0.05])).reshape(3, 1)
        if f_display <= 0.05:
            base_rgba = np.array([0.25, 0.85, 0.35, 0.9], dtype=np.float32).reshape(4, 1)
        else:
            base_rgba = np.array([0.15, 0.75, 0.25, 0.9], dtype=np.float32).reshape(4, 1)

        self._set_user_geom(
            viewer.user_scn.geoms[idx],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.020, 0.0, 0.0], dtype=np.float64).reshape(3, 1),
            base_pos,
            identity,
            base_rgba,
        )
        idx += 1

        if f_display > 0.05:
            max_threshold = 35.0
            intensity = min(f_display / max_threshold, 1.0)
            arrow_len = 0.08 + intensity * 0.28
            shaft_width = 0.020
            p1 = hand_pos + np.array([0.0, 0.0, 0.07])
            p2 = hand_pos + np.array([0.0, 0.0, 0.07 + arrow_len])
            color = np.array([intensity, 1.0 - intensity, 0.0, 0.95], dtype=np.float32).reshape(4, 1)

            arrow_geom = viewer.user_scn.geoms[idx]
            self._set_user_geom(arrow_geom, mujoco.mjtGeom.mjGEOM_ARROW, zero3, zero3, identity, color)
            mujoco.mjv_connector(arrow_geom, mujoco.mjtGeom.mjGEOM_ARROW, shaft_width, p1, p2)
            idx += 1

            self._set_user_geom(
                viewer.user_scn.geoms[idx],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.026, 0.0, 0.0], dtype=np.float64).reshape(3, 1),
                p2.reshape(3, 1),
                identity,
                color,
            )
            idx += 1

        viewer.user_scn.ngeom = idx

    def _apply_control_policy_callback(self, model, data):
        self._apply_control_policy()

    def _passive_callback(self, model, data):
        self._record_telemetry()

    def _print_peg_controls(self):
        print("\n=== INTERACTIVE PEG-IN-HOLE TASK ===")
        print("Goal: Align the peg over the socket and insert safely.")
        print()
        print("Click the MuJoCo window, then use:")
        print("  Arrow keys   : Up/Down = North/South, Left/Right = West/East")
        print("  9 / 8        : Raise / lower target (Z)")
        print("  6 / 7        : Roll peg left / right (spin about vertical)")
        print("  Page Up/Down : Also raise / lower (Z), if your keyboard has them")
        print("  , / .        : Open / close gripper")
        print()
        print("DO NOT press I, J, K, or U — those are MuJoCo debug toggles")
        print("(red collision boxes, joint axes, etc.), not robot controls.")
        print("Hold arrow keys for smooth motion if pynput is installed.")
        print("Peg orientation is locked pointing down; use 6/7 to spin it.")
        if self.force_feedback:
            print("Force feedback overlay: ON")
            print("  Green sphere above hand = waiting for contact")
            print("  Vertical green→red arrow + tip sphere = force while inserted")
            print("  (Run with --force-feedback; HUD also shows force in newtons)")
        else:
            print("Force feedback overlay: OFF")
        print()

    def _run_passive_viewer(self, interactive=False):
        if interactive:
            self._print_peg_controls()
            self._start_pynput_teleop()

        recorder = None
        if self.record_video:
            recorder = VideoRecorder(self.model, self.video_path)
            recorder.start()
            print(f"Recording video → {self.video_path.resolve()}")

        mujoco.set_mjcb_control(self._apply_control_policy_callback)
        substeps = 3 if interactive else 1

        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                key_callback=self._viewer_key_callback if interactive else None,
                show_left_ui=False,
                show_right_ui=False,
            ) as viewer:
                while viewer.is_running():
                    if interactive:
                        self._apply_teleop_motion(self.model.opt.timestep)
                        self._sync_target_marker()

                    for _ in range(substeps):
                        mujoco.mj_step(self.model, self.data)

                    if interactive:
                        self._update_live_force()
                    self._record_telemetry()

                    if interactive:
                        self._draw_force_feedback(viewer)
                        self._update_peg_hud(viewer)

                    if recorder is not None:
                        recorder.capture(self.data, viewer.cam)

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
                self._stop_pynput_teleop()
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

    def _run_interactive_peg(self):
        self._run_passive_viewer(interactive=True)

    def run(self):
        print(f"Booting up environment factory running: [{self.scenario.upper()}]")

        if self.scenario == "peg_in_hole" and self.interactive:
            self._run_interactive_peg()
        elif self.record_video:
            self._run_passive_viewer(interactive=False)
        else:
            self._run_standard_viewer()

        self.log_file.close()
        self.plot_comparison()

    def plot_comparison(self):
        if not self.time_history:
            print("No force samples recorded; skipping plot.")
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        t = np.array(self.time_history)
        f_true = np.array(self.true_force_history)
        f_est = np.array(self.estimated_force_history)
        in_contact = np.array(self.in_contact_history, dtype=bool)
        is_anomaly = np.array(self.anomaly_history, dtype=bool)
        is_clean = ~is_anomaly

        self._write_filtered_csv(t, f_true, f_est, in_contact, is_clean)

        self._save_plot(
            plt, t, f_true, f_est,
            title='Raw: Measured vs. Estimated Contact Forces',
            path=self.plot_raw_path,
        )

        if np.any(is_clean):
            self._save_plot(
                plt, t[is_clean], f_true[is_clean], f_est[is_clean],
                title='Filtered: Measured vs. Estimated Contact Forces',
                path=self.plot_filtered_path,
            )
        else:
            print("No clean samples left after filtering; skipped filtered full plot.")

        if np.any(in_contact):
            self._save_plot(
                plt, t[in_contact], f_true[in_contact], f_est[in_contact],
                title='Raw (Contact-Only): Measured vs. Estimated Contact Forces',
                path=self.plot_contact_raw_path,
            )

            contact_clean = in_contact & is_clean
            if np.any(contact_clean):
                self._save_plot(
                    plt, t[contact_clean], f_true[contact_clean], f_est[contact_clean],
                    title='Filtered (Contact-Only): Measured vs. Estimated Contact Forces',
                    path=self.plot_contact_filtered_path,
                )
            else:
                print("No clean contact samples; skipped filtered contact-only plot.")
        else:
            print("No target contacts recorded; skipped contact-only plots.")

        n_anomaly = int(np.sum(is_anomaly))
        print(f"Flagged {n_anomaly}/{len(t)} samples as anomalies "
              f"({100 * n_anomaly / len(t):.1f}%)")

    def _write_filtered_csv(self, times, true_forces, est_forces, in_contact, is_clean):
        with open(self.telemetry_filtered_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time (s)", "Ground Truth (N)", "Jacobian Estimate (N)", "In Contact"])
            for i in np.where(is_clean)[0]:
                writer.writerow([
                    times[i], true_forces[i], est_forces[i], int(in_contact[i])
                ])
        print(f"Saved filtered CSV to {self.telemetry_filtered_path.resolve()}")

    def _save_plot(self, plt, times, true_forces, est_forces, title, path):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, true_forces,
                label='Ground Truth (contact, world frame)', color='black', linewidth=2.5)
        ax.plot(times, est_forces,
                label='Jacobian Estimate (qfrc_constraint)', color='orange', linestyle='--', linewidth=2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Simulation Time (Seconds)')
        ax.set_ylabel('Force Amplitude (Newtons)')
        ax.grid(True, linestyle=':')
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved force plot to {path.resolve()}")

    def __del__(self):
        try:
            if hasattr(self, 'log_file') and not self.log_file.closed:
                self.log_file.close()
        except Exception:
            pass


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
        help="Enable live force arrow overlay (peg_in_hole + --interactive only)",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Save run_recording.mp4 in results/<scenario>/ (uses passive viewer; mjpython on macOS)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env = FrankaForceEnv(
        scenario=args.scenario,
        interactive=args.interactive,
        force_feedback=args.force_feedback,
        record_video=args.record_video,
    )
    env.run()
