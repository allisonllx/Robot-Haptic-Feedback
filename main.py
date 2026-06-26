import csv
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path

MODEL_PATH = Path('mujoco_menagerie/franka_emika_panda/scene.xml')
RESULTS_DIR = Path('results')

class FrankaForceEnv:
    def __init__(self, scenario="hit_floor"):
        self.scenario = scenario
        self.results_dir = RESULTS_DIR / scenario
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_path = self.results_dir / 'force_verification_log.csv'
        self.plot_path = self.results_dir / 'force_comparison.png'
        self.plot_contact_path = self.results_dir / 'force_comparison_contact_only.png'
        
        # Storage lists for timeline telemetry
        self.time_history = []
        self.true_force_history = []
        self.estimated_force_history = []
        self.in_contact_history = []

        self.step_counter = 0
        self.downsample_factor = 10

        # Telemetry CSV Setup
        self.log_file = open(self.telemetry_path, mode="w", newline="")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow([
            "Time (s)", "Ground Truth (N)", "Jacobian Estimate (N)", "In Contact"
        ])
        
        # 1. Build the MuJoCo model for the chosen scenario
        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        
        # Cache structural IDs
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        self.hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self.scenario == "push_block":
            self.block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_block")

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
        elif self.scenario != "hit_floor":
            raise ValueError(f"Unknown scenario configuration: {self.scenario}")

        return spec.compile()

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

    def _record_telemetry(self):
        """Sample ground-truth and estimated forces."""
        if self.step_counter % self.downsample_factor == 0:
            gripper_ids = self._get_active_gripper_body_ids()
            in_contact = self._has_target_contact(gripper_ids)
            f_true = self._calculate_ground_truth_force(gripper_ids)
            f_est = self._estimate_virtual_force() if in_contact else 0.0

            self.time_history.append(self.data.time)
            self.true_force_history.append(f_true)
            self.estimated_force_history.append(f_est)
            self.in_contact_history.append(in_contact)

            self.log_writer.writerow([self.data.time, f_true, f_est, int(in_contact)])

        self.step_counter += 1

    def _apply_control_policy_callback(self, model, data):
        """Control hook: only write actuator commands."""
        self._apply_control_policy()

    def _passive_callback(self, model, data):
        """Runs after forward kinematics each step."""
        self._record_telemetry()

    def run(self):
        """Launches the window execution."""
        print(f"Booting up environment factory running: [{self.scenario.upper()}]")

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

        self._save_plot(
            plt, t, f_true, f_est,
            title='Verification Profile: Measured vs. Estimated Contact Forces',
            path=self.plot_path,
        )

        if np.any(in_contact):
            self._save_plot(
                plt, t[in_contact], f_true[in_contact], f_est[in_contact],
                title='Contact-Only: Measured vs. Estimated Contact Forces',
                path=self.plot_contact_path,
            )
            print(f"Saved contact-only plot to {self.plot_contact_path.resolve()}")
        else:
            print("No target contacts recorded; skipped contact-only plot.")

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

if __name__ == "__main__":
    env = FrankaForceEnv(scenario="push_block")  # "hit_floor" or "push_block"
    env.run()
