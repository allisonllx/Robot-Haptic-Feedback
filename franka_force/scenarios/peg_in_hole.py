import threading

import mujoco
import numpy as np

from .base import Scenario

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None


class PegInHoleScenario(Scenario):
    name = "peg_in_hole"
    supports_interactive = True
    force_visual_min = 10.0
    force_visual_threshold = force_visual_min
    force_visual_max = 1000.0
    force_arrow_offset = np.array([0.0, -0.12, 0.14])

    def initialize_state(self, env):
        env.target_pos = np.zeros(3)
        env.target_roll = 0.0
        env.teleop_speed = 0.10
        env.roll_speed = 0.8
        env.gripper_closed = False
        env._teleop_lock = threading.Lock()
        env._move_cmd = np.zeros(3)
        env._roll_cmd = 0.0
        env._keyboard_listener = None
        env._peg_home_q = np.array([0.0, 0.229, 0.0, -1.80, 0.0, 2.25, 0.80])
        env._peg_down = np.array([0.0, 0.0, -1.0])

    def augment_model_spec(self, env, spec):
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

    def resolve_ids(self, env):
        env.peg_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom")
        env.ik_target_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "ik_target")
        env.ik_target_mocap_id = env.model.body_mocapid[env.ik_target_body_id]

    def after_model_init(self, env):
        if env.interactive:
            self._boost_peg_actuators(env)
        self._init_peg_home_pose(env)

    def apply_control(self, env):
        self._apply_peg_ik_control(env)

    def sample_forces(self, env):
        in_contact, f_true, contact_pos, contact_frame, contact_force = self._peg_contact_summary(env)
        env.latest_contact_pos = contact_pos
        env.latest_contact_frame = contact_frame
        env.latest_contact_force = contact_force
        f_est = env._estimate_virtual_force() if in_contact else 0.0
        return in_contact, f_true, f_est

    def print_controls(self, env):
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
        if env.force_feedback:
            print("Force feedback overlay: ON")
            print(f"  Visual mode: {env.force_visual}")
            print("  Green sphere above hand = waiting for contact.")
            print("  Red/orange arrow and/or contact ring scale with force.")
            print("  (Run with --force-feedback; HUD also shows force in newtons)")
        else:
            print("Force feedback overlay: OFF")
        print()

    def start_interactive(self, env):
        self._start_pynput_teleop(env)

    def stop_interactive(self, env):
        self._stop_pynput_teleop(env)

    def viewer_key_callback(self, env, keycode):
        """Nudge the IK target from the MuJoCo window (one step per key press)."""
        if not env.interactive:
            return

        # GLFW key codes — avoid I/J/K/U; those toggle MuJoCo debug overlays.
        if keycode == 265:      # Up arrow -> North (+Y)
            self._nudge_target(env, dy=1.0)
        elif keycode == 264:    # Down arrow -> South (-Y)
            self._nudge_target(env, dy=-1.0)
        elif keycode == 262:    # Right arrow -> East (+X)
            self._nudge_target(env, dx=1.0)
        elif keycode == 263:    # Left arrow -> West (-X)
            self._nudge_target(env, dx=-1.0)
        elif keycode == 266:    # Page Up -> +Z
            self._nudge_target(env, dz=1.0)
        elif keycode == 267:    # Page Down -> -Z
            self._nudge_target(env, dz=-1.0)
        elif keycode in (57,):  # 9 raise (Z+)
            self._nudge_target(env, dz=1.0)
        elif keycode in (56,):  # 8 lower (Z-)
            self._nudge_target(env, dz=-1.0)
        elif keycode in (44,):  # , open gripper
            self._set_gripper(env, False)
        elif keycode in (46,):  # . close gripper
            self._set_gripper(env, True)
        elif keycode in (54,):  # 6 roll CCW
            self._nudge_roll(env, -1.0)
        elif keycode in (55,):  # 7 roll CW
            self._nudge_roll(env, 1.0)

    def before_interactive_step(self, env, dt):
        self._apply_teleop_motion(env, dt)
        self._sync_target_marker(env)

    def update_interactive_viewer(self, env, viewer):
        self._draw_force_feedback(env, viewer)
        self._update_peg_hud(env, viewer)

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

    def _init_peg_home_pose(self, env):
        """Place the arm over the socket and sync the IK target to the current hand pose."""
        env.data.qpos[:7] = env._peg_home_q
        mujoco.mj_forward(env.model, env.data)
        for i in range(7):
            env.data.ctrl[i] = env._peg_home_q[i]
        env.data.ctrl[7] = 0.0
        with env._teleop_lock:
            env.target_pos = env.data.xpos[env.hand_body_id].copy()
        self._sync_target_marker(env)

    def _boost_peg_actuators(self, env):
        """Stiffen arm servos so interactive IK targets are tracked quickly."""
        for i in range(7):
            env.model.actuator_gainprm[i, 0] *= 10.0
            env.model.actuator_biasprm[i, 1] *= 10.0
            env.model.actuator_biasprm[i, 2] *= 10.0

    def _target_hand_rotmat(self, env, roll):
        """Hand frame with peg axis (+Z) pointing down; roll spins peg about vertical."""
        z_des = env._peg_down
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

    def _solve_peg_ik(self, env, target_pos, target_roll):
        """6-DOF iterative IK: reach target_pos with peg axis pointing down."""
        saved_qpos = env.data.qpos.copy()
        q_cmd = saved_qpos[:7].copy()
        target_rot = self._target_hand_rotmat(env, target_roll)
        dls_lambda = 0.025
        max_dq = 0.10
        pos_step_cap = 0.18
        ori_step_cap = 0.35
        pos_weight = 1.0
        ori_weight = 2.5

        try:
            for _ in range(24):
                env.data.qpos[:7] = q_cmd
                mujoco.mj_kinematics(env.model, env.data)
                ee = env.data.xpos[env.hand_body_id].copy()
                current_rot = env.data.xmat[env.hand_body_id].reshape(3, 3).copy()

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
                jac_p = np.zeros((3, env.model.nv))
                jac_r = np.zeros((3, env.model.nv))
                mujoco.mj_jac(
                    env.model, env.data, jac_p, jac_r, ee, env.hand_body_id,
                )
                j_arm = np.vstack([pos_weight * jac_p[:, :7], ori_weight * jac_r[:, :7]])
                dq = j_arm.T @ np.linalg.solve(
                    j_arm @ j_arm.T + dls_lambda ** 2 * np.eye(6),
                    task_err,
                )
                q_cmd += np.clip(dq, -max_dq, max_dq)
        finally:
            env.data.qpos[:] = saved_qpos
            mujoco.mj_kinematics(env.model, env.data)

        return q_cmd

    def _solve_peg_ik_pos_only(self, env, target_pos):
        """3-DOF fallback for non-interactive peg mode."""
        saved_qpos = env.data.qpos.copy()
        q_cmd = saved_qpos[:7].copy()
        dls_lambda = 0.025
        max_dq = 0.10
        cart_step_cap = 0.18

        try:
            for _ in range(20):
                env.data.qpos[:7] = q_cmd
                mujoco.mj_kinematics(env.model, env.data)
                ee = env.data.xpos[env.hand_body_id].copy()
                error = target_pos - ee
                error_norm = np.linalg.norm(error)
                if error_norm < 3e-3:
                    break

                step_error = error / max(error_norm, 1e-6) * min(error_norm, cart_step_cap)
                jac_p = np.zeros((3, env.model.nv))
                jac_r = np.zeros((3, env.model.nv))
                mujoco.mj_jac(
                    env.model, env.data, jac_p, jac_r, ee, env.hand_body_id,
                )
                j_arm = jac_p[:, :7]
                dq = j_arm.T @ np.linalg.solve(
                    j_arm @ j_arm.T + dls_lambda ** 2 * np.eye(3),
                    step_error,
                )
                q_cmd += np.clip(dq, -max_dq, max_dq)
        finally:
            env.data.qpos[:] = saved_qpos
            mujoco.mj_kinematics(env.model, env.data)

        return q_cmd

    def _apply_peg_ik_control(self, env):
        """IK toward target pose (interactive: 6-DOF peg-down + roll)."""
        with env._teleop_lock:
            target_pos = env.target_pos.copy()
            target_roll = env.target_roll
            gripper_closed = env.gripper_closed

        if env.interactive:
            q_des = self._solve_peg_ik(env, target_pos, target_roll)
        else:
            q_des = self._solve_peg_ik_pos_only(env, target_pos)

        for i in range(7):
            env.data.ctrl[i] = q_des[i]
        env.data.ctrl[7] = 255.0 if gripper_closed else 0.0

    def _peg_contact_summary(self, env):
        force_world = np.zeros(3)
        strongest_force = 0.0
        strongest_pos = None
        strongest_frame = None

        for i in range(env.data.ncon):
            contact = env.data.contact[i]
            if contact.geom1 != env.peg_geom_id and contact.geom2 != env.peg_geom_id:
                continue

            contact_force = env._contact_force_in_world(i)
            contact_force_mag = float(np.linalg.norm(contact_force))
            force_world += contact_force

            if contact_force_mag > strongest_force:
                strongest_force = contact_force_mag
                strongest_pos = contact.pos.copy()
                strongest_frame = contact.frame.reshape(3, 3).copy()

        return (
            strongest_pos is not None,
            float(np.linalg.norm(force_world)),
            strongest_pos,
            strongest_frame,
            strongest_force,
        )

    def _sync_target_marker(self, env):
        with env._teleop_lock:
            target = env.target_pos.copy()
        env.data.mocap_pos[env.ik_target_mocap_id] = target
        env.data.mocap_quat[env.ik_target_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])

    def _nudge_target(self, env, dx=0.0, dy=0.0, dz=0.0):
        step = 0.02
        with env._teleop_lock:
            env.target_pos[0] += dx * step
            env.target_pos[1] += dy * step
            env.target_pos[2] += dz * step

    def _set_gripper(self, env, closed):
        with env._teleop_lock:
            env.gripper_closed = closed

    def _nudge_roll(self, env, delta):
        step = 0.08
        with env._teleop_lock:
            env.target_roll += delta * step

    def _adjust_roll_cmd(self, env, delta):
        with env._teleop_lock:
            env._roll_cmd = np.clip(env._roll_cmd + delta, -1.0, 1.0)

    def _apply_teleop_motion(self, env, dt):
        with env._teleop_lock:
            move_cmd = env._move_cmd.copy()
            roll_cmd = env._roll_cmd
        if np.any(move_cmd != 0) or roll_cmd != 0:
            with env._teleop_lock:
                if np.any(move_cmd != 0):
                    env.target_pos += move_cmd * env.teleop_speed * dt
                if roll_cmd != 0:
                    env.target_roll += roll_cmd * env.roll_speed * dt

    def _adjust_move_cmd(self, env, axis, delta):
        with env._teleop_lock:
            env._move_cmd[axis] = np.clip(env._move_cmd[axis] + delta, -1.0, 1.0)

    def _start_pynput_teleop(self, env):
        if pynput_keyboard is None:
            print("Note: install pynput for smoother hold-to-move teleop (pip install pynput).")
            return

        def on_press(key):
            try:
                if key == pynput_keyboard.Key.up:
                    self._adjust_move_cmd(env, 1, 1.0)
                elif key == pynput_keyboard.Key.down:
                    self._adjust_move_cmd(env, 1, -1.0)
                elif key == pynput_keyboard.Key.right:
                    self._adjust_move_cmd(env, 0, 1.0)
                elif key == pynput_keyboard.Key.left:
                    self._adjust_move_cmd(env, 0, -1.0)
                elif key == pynput_keyboard.Key.page_up:
                    self._adjust_move_cmd(env, 2, 1.0)
                elif key == pynput_keyboard.Key.page_down:
                    self._adjust_move_cmd(env, 2, -1.0)
                elif hasattr(key, "char") and key.char == "9":
                    self._adjust_move_cmd(env, 2, 1.0)
                elif hasattr(key, "char") and key.char == "8":
                    self._adjust_move_cmd(env, 2, -1.0)
                elif hasattr(key, "char") and key.char == ",":
                    self._set_gripper(env, False)
                elif hasattr(key, "char") and key.char == ".":
                    self._set_gripper(env, True)
                elif hasattr(key, "char") and key.char == "6":
                    self._adjust_roll_cmd(env, -1.0)
                elif hasattr(key, "char") and key.char == "7":
                    self._adjust_roll_cmd(env, 1.0)
            except Exception:
                pass

        def on_release(key):
            try:
                if key == pynput_keyboard.Key.up:
                    self._adjust_move_cmd(env, 1, -1.0)
                elif key == pynput_keyboard.Key.down:
                    self._adjust_move_cmd(env, 1, 1.0)
                elif key == pynput_keyboard.Key.right:
                    self._adjust_move_cmd(env, 0, -1.0)
                elif key == pynput_keyboard.Key.left:
                    self._adjust_move_cmd(env, 0, 1.0)
                elif key == pynput_keyboard.Key.page_up:
                    self._adjust_move_cmd(env, 2, -1.0)
                elif key == pynput_keyboard.Key.page_down:
                    self._adjust_move_cmd(env, 2, 1.0)
                elif hasattr(key, "char") and key.char == "9":
                    self._adjust_move_cmd(env, 2, -1.0)
                elif hasattr(key, "char") and key.char == "8":
                    self._adjust_move_cmd(env, 2, 1.0)
                elif hasattr(key, "char") and key.char == "6":
                    self._adjust_roll_cmd(env, 1.0)
                elif hasattr(key, "char") and key.char == "7":
                    self._adjust_roll_cmd(env, -1.0)
            except Exception:
                pass

        env._keyboard_listener = pynput_keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
        )
        env._keyboard_listener.start()

    def _stop_pynput_teleop(self, env):
        if env._keyboard_listener is not None:
            env._keyboard_listener.stop()
            env._keyboard_listener = None
        with env._teleop_lock:
            env._move_cmd[:] = 0.0
            env._roll_cmd = 0.0

    def _update_peg_hud(self, env, viewer):
        with env._teleop_lock:
            target = env.target_pos.copy()
            gripper = "closed" if env.gripper_closed else "open"
            moving = np.any(env._move_cmd != 0) or env._roll_cmd != 0
            roll_deg = np.degrees(env.target_roll)
        force_line = ""
        if env.force_feedback:
            f_display = env._force_feedback_magnitude()
            force_line = (
                f"force {f_display:.1f} N"
                + (" (contact)" if env.latest_in_contact else " (no contact yet)")
                + f" | visual {env.force_visual}"
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

    def _draw_force_feedback(self, env, viewer):
        if not env.force_feedback or viewer.user_scn is None:
            return

        f_display = env._force_feedback_magnitude()
        hand_pos = env.data.xpos[env.hand_body_id].astype(np.float64)
        identity = np.eye(3, dtype=np.float64).reshape(9, 1)

        viewer.user_scn.ngeom = 0
        idx = 0

        idx = self._draw_idle_marker(env, viewer, idx, hand_pos, identity, f_display)

        if f_display > self.force_visual_threshold:
            if env.force_visual in ("arrow", "both"):
                idx = self._draw_force_arrow(viewer, idx, hand_pos, identity, f_display)
            if env.force_visual in ("ring", "both"):
                idx = self._draw_contact_ring(env, viewer, idx, identity)

        viewer.user_scn.ngeom = idx

    def _draw_idle_marker(self, env, viewer, idx, hand_pos, identity, f_display):
        if not self._has_user_geom_slot(viewer, idx):
            return idx

        base_pos = self._force_gauge_origin(hand_pos).reshape(3, 1)
        if f_display <= self.force_visual_threshold:
            base_rgba = np.array([0.25, 0.85, 0.35, 0.9], dtype=np.float32).reshape(4, 1)
        else:
            base_rgba = np.array([1.0, 0.20, 0.05, 0.9], dtype=np.float32).reshape(4, 1)

        self._set_user_geom(
            viewer.user_scn.geoms[idx],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.020, 0.0, 0.0], dtype=np.float64).reshape(3, 1),
            base_pos,
            identity,
            base_rgba,
        )
        return idx + 1

    def _draw_force_arrow(self, viewer, idx, hand_pos, identity, force_magnitude):
        if not self._has_user_geom_slot(viewer, idx):
            return idx

        intensity = self._force_visual_intensity(force_magnitude)
        arrow_len = 0.06 + intensity * 0.42
        shaft_width = 0.012 + intensity * 0.018
        p1 = self._force_gauge_origin(hand_pos)
        p2 = p1 + np.array([0.0, 0.0, arrow_len])
        color = self._force_color(intensity)
        zero3 = np.zeros((3, 1), dtype=np.float64)

        arrow_geom = viewer.user_scn.geoms[idx]
        self._set_user_geom(arrow_geom, mujoco.mjtGeom.mjGEOM_ARROW, zero3, zero3, identity, color)
        mujoco.mjv_connector(arrow_geom, mujoco.mjtGeom.mjGEOM_ARROW, shaft_width, p1, p2)
        return idx + 1

    def _draw_contact_ring(self, env, viewer, idx, identity):
        if env.latest_contact_pos is None or env.latest_contact_frame is None:
            return idx

        segments = 24
        if not self._has_user_geom_slot(viewer, idx + segments - 1):
            return idx

        force_magnitude = max(env.latest_contact_force, self._force_feedback_magnitude(env))
        intensity = self._force_visual_intensity(force_magnitude)
        radius = 0.025 + intensity * 0.095
        ring_width = 0.0035 + intensity * 0.007
        color = self._force_color(intensity)
        frame = env.latest_contact_frame
        normal = self._unit_vector(frame[0], np.array([0.0, 0.0, 1.0]))
        tangent_a = self._unit_vector(frame[1], np.array([1.0, 0.0, 0.0]))
        tangent_b = np.cross(normal, tangent_a)
        if np.linalg.norm(tangent_b) < 1e-9:
            tangent_b = self._unit_vector(frame[2], np.array([0.0, 1.0, 0.0]))
        else:
            tangent_b = self._unit_vector(tangent_b, np.array([0.0, 1.0, 0.0]))
        tangent_a = self._unit_vector(np.cross(tangent_b, normal), tangent_a)
        center = env.latest_contact_pos + normal * 0.002
        zero3 = np.zeros((3, 1), dtype=np.float64)

        for segment in range(segments):
            theta1 = 2.0 * np.pi * segment / segments
            theta2 = 2.0 * np.pi * (segment + 1) / segments
            p1 = center + radius * (np.cos(theta1) * tangent_a + np.sin(theta1) * tangent_b)
            p2 = center + radius * (np.cos(theta2) * tangent_a + np.sin(theta2) * tangent_b)
            ring_geom = viewer.user_scn.geoms[idx]
            self._set_user_geom(ring_geom, mujoco.mjtGeom.mjGEOM_CAPSULE, zero3, zero3, identity, color)
            mujoco.mjv_connector(ring_geom, mujoco.mjtGeom.mjGEOM_CAPSULE, ring_width, p1, p2)
            idx += 1

        return idx

    def _force_visual_intensity(self, force_magnitude):
        if force_magnitude <= self.force_visual_min:
            return 0.0
        log_min = np.log(self.force_visual_min)
        log_max = np.log(self.force_visual_max)
        log_force = np.log(min(force_magnitude, self.force_visual_max))
        return float(np.clip((log_force - log_min) / (log_max - log_min), 0.0, 1.0))

    def _force_gauge_origin(self, hand_pos):
        return hand_pos + self.force_arrow_offset

    def _force_color(self, intensity):
        green = 0.35 * (1.0 - intensity)
        return np.array([1.0, green, 0.02, 0.95], dtype=np.float32).reshape(4, 1)

    def _unit_vector(self, value, fallback):
        value = np.asarray(value, dtype=np.float64)
        norm = np.linalg.norm(value)
        if norm < 1e-9:
            return fallback.astype(np.float64)
        return value / norm

    def _has_user_geom_slot(self, viewer, idx):
        maxgeom = getattr(viewer.user_scn, "maxgeom", None)
        if maxgeom is None:
            maxgeom = len(viewer.user_scn.geoms)
        return idx < maxgeom

    def _force_feedback_magnitude(self, env):
        return max(env.latest_f_est, env.latest_f_true)
