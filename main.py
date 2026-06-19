import csv
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path

MODEL_PATH = Path('mujoco_menagerie/franka_emika_panda/scene.xml')
PLOT_PATH = Path('force_comparison.png')

class FrankaForceEnv:
    def __init__(self, scenario="hit_floor"):
        self.scenario = scenario
        
        # Storage lists for timeline telemetry
        self.time_history = []
        self.true_force_history = []
        self.estimated_force_history = []

        self.step_counter = 0
        self.downsample_factor = 10

        # Telemetry CSV Setup
        self.log_file = open("force_verification_log.csv", mode="w", newline="")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(["Time (s)", "Ground Truth (N)", "Jacobian Estimate (N)"])
        
        # 1. Build the MuJoCo model for the chosen scenario
        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        
        # 3. Cache crucial structural IDs
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        self.hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
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
                mass=5.0, # 5.0 kg
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
            # Slam straight downwards
            self.data.ctrl[1] = 0.229
            self.data.ctrl[3] = -2.20
            self.data.ctrl[5] = 2.30
            self.data.ctrl[6] = 0.80
            self.data.ctrl[7] = 255
            
        elif self.scenario == "push_block":
            # Phase 1A: High Hover (0.0 to 1.5 seconds)
            # Stay completely elevated above the block's workspace profile
            if self.data.time < 1.5:
                self.data.ctrl[1] = 0.229
                self.data.ctrl[3] = -1.80  # 🟢 High, retracted elbow position
                self.data.ctrl[5] = 1.87
                self.data.ctrl[6] = 0.80
                self.data.ctrl[7] = 255
                
            # Phase 1B: Vertical Drop Down (1.5 to 3.0 seconds)
            # Sink down directly behind the back face of the block
            elif self.data.time < 3.0:
                progress = (self.data.time - 1.5) / 1.5
                self.data.ctrl[1] = 0.229
                # Smoothly drop elbow from -1.80 down to your stable stance of -2.37
                self.data.ctrl[3] = -1.80 + progress * (-2.37 - (-1.80))
                self.data.ctrl[5] = 1.87 + progress * (2.25 - 1.87)
                self.data.ctrl[6] = 0.80
                self.data.ctrl[7] = 255
                
            # Phase 2: Coordinated Flat Horizontal Push (3.0 to 6.0 seconds)
            # Push forward while leaning the shoulder down to keep the path flat
            elif self.data.time < 6.0:
                progress = (self.data.time - 3.0) / 3.0
                self.data.ctrl[3] = -2.37 + progress * (-2.05 - (-2.37))
                self.data.ctrl[1] = 0.229 + progress * (0.420 - 0.229)
                self.data.ctrl[5] = 2.25
                self.data.ctrl[6] = 0.80
                self.data.ctrl[7] = 255
                
            # Phase 3: Hold the plateau force (After 6.0 seconds)
            else:
                self.data.ctrl[1] = 0.420
                self.data.ctrl[3] = -2.05
                self.data.ctrl[5] = 2.25
                self.data.ctrl[6] = 0.80
                self.data.ctrl[7] = 255

    def _get_active_gripper_body_ids(self):
        """Returns the IDs of the active tool center contact surfaces"""
        gripper_names = ["hand", "left_finger", "right_finger"]
        return [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in gripper_names]
    
    def _calculate_ground_truth_force(self, gripper_ids):
        """Extracts direct oracle physical normal force (F_true)"""
        total_true_force = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            
            is_touching = False
            if self.scenario == "hit_floor":
                if body1 in gripper_ids or body2 in gripper_ids:
                    is_touching = True
            elif self.scenario == "push_block":
                is_touching = (body1 in gripper_ids and body2 == self.block_id) or \
                              (body2 in gripper_ids and body1 == self.block_id)
                              
            if is_touching:
                c_forces = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, c_forces)
                total_true_force += c_forces[0]
        return total_true_force

    def _estimate_virtual_force(self):
        """Step 2 & 3: Math pipeline to compute F_estimated from joint torques"""
        # A. Grab raw motor efforts from the 7 arm joints
        tau_measured = self.data.qfrc_actuator[:7]
        
        # B. Extract ONLY gravity, Coriolis, and centrifugal torques
        tau_passive_bias = self.data.qfrc_bias[:7]

        # C. Inertial torques M(q) @ qacc (use mj_mulM; mj_fullM expects data.qM, not data)
        tau_inertia = np.zeros(self.model.nv)
        mujoco.mj_mulM(self.model, self.data, tau_inertia, self.data.qacc)
        tau_inertia = tau_inertia[:7]
        
        # D. Isolate the external torque component
        tau_ext = tau_measured - (tau_passive_bias + tau_inertia)
        
        # E. Extract the 6x7 operational space Jacobian for the Hand body position
        jac_p = np.zeros((3, self.model.nv))
        jac_r = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jac_p, jac_r, self.data.xpos[self.hand_body_id], self.hand_body_id)
        J = np.vstack([jac_p, jac_r])[:, :7]
        
        # F. Map joint torques to Cartesian coordinates using the Pseudo-Inverse of J^T
        J_T_pinv = np.linalg.pinv(J.T)
        wrench_estimated = J_T_pinv @ tau_ext
        
        # G. Return the magnitude of the linear spatial force vector (X, Y, Z)
        linear_forces = wrench_estimated[:3]
        return np.linalg.norm(linear_forces)

    def _record_telemetry(self):
        """Sample ground-truth and estimated forces (call after mj_step, not inside callbacks)."""
        if self.step_counter % self.downsample_factor == 0:
            gripper_ids = self._get_active_gripper_body_ids()
            f_true = self._calculate_ground_truth_force(gripper_ids)
            f_est = self._estimate_virtual_force()

            self.time_history.append(self.data.time)
            self.true_force_history.append(f_true)
            self.estimated_force_history.append(f_est)

            self.log_writer.writerow([self.data.time, f_true, f_est])

        self.step_counter += 1

    def _apply_control_policy_callback(self, model, data):
        """Control hook: only write actuator commands, no dynamics/Jacobian calls."""
        self._apply_control_policy()

    def _passive_callback(self, model, data):
        """Runs after forward kinematics each step; safe place to read forces/Jacobians."""
        self._record_telemetry()

    def run(self):
        """Launches the window execution"""
        print(f"Booting up environment factory running: [{self.scenario.upper()}]")

        # Optional automated motion; comment out to keep manual drag-only control.
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
            self.log_file.close()

        self.plot_comparison()

    def plot_comparison(self):
        if not self.time_history:
            print("No force samples recorded; skipping plot.")
            return

        # Import after the viewer closes to avoid macOS GUI/GLFW conflicts.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.time_history, self.true_force_history,
                label='Ground Truth (mj_contactForce)', color='black', linewidth=2.5)
        ax.plot(self.time_history, self.estimated_force_history,
                label='Virtual Estimate (Jacobian Math)', color='orange', linestyle='--', linewidth=2)

        ax.set_title('Verification Profile: Measured vs. Estimated Contact Forces', fontsize=12, fontweight='bold')
        ax.set_xlabel('Simulation Time (Seconds)')
        ax.set_ylabel('Force Amplitude (Newtons)')
        ax.grid(True, linestyle=':')
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=150)
        plt.close(fig)
        print(f"Saved force plot to {PLOT_PATH.resolve()}")

    def __del__(self):
        """Fallback safety wrapper to close open file streams if environment catches an exception"""
        try:
            if hasattr(self, 'log_file') and not self.log_file.closed:
                self.log_file.close()
        except:
            pass

if __name__ == "__main__":
    env = FrankaForceEnv(scenario="hit_floor") # "hit_floor" or "push_block"
    env.run()