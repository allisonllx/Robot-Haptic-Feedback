import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = Path('mujoco_menagerie/franka_emika_panda/scene.xml')

class FrankaForceEnv:
    def __init__(self, scenario="hit_floor"):
        self.scenario = scenario
        
        # Storage lists for timeline telemetry
        self.time_history = []
        self.force_history = []
        self.step_counter = 0
        self.downsample_factor = 10
        
        # 1. Build the MuJoCo model for the chosen scenario
        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        
        # 3. Cache crucial structural IDs
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        if self.scenario == "push_block":
            self.block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_block")

    def _build_model(self):
        """Load scene.xml from disk (so includes resolve), then inject scenario extras."""
        spec = mujoco.MjSpec.from_file(str(MODEL_PATH))

        if self.scenario == "push_block":
            body = spec.worldbody.add_body(name="target_block", pos=[0.4, 0.0, 0.05])
            body.add_freejoint()
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.04, 0.04, 0.05],
                mass=2.0,
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
            self.data.ctrl[1] = -1.0
            self.data.ctrl[2] = 0.5
            
        elif self.scenario == "push_block":
            # Phase 1: Get in position behind the box, Phase 2: Drive forward
            if self.data.time < 2.0:
                self.data.ctrl[1] = 0.5   
                self.data.ctrl[3] = -1.5  
            else:
                self.data.ctrl[3] = -2.2  

    def _extract_contact_forces(self):
        """Factory Method: Broadened to capture any end-effector assembly collision"""
        total_force = 0.0
        
        # Cache all body IDs that belong to the gripper/hand assembly group
        # This covers whatever part of the hand or wrist slams into the object
        gripper_body_names = ["link7", "hand", "left_finger", "right_finger"]
        gripper_body_ids = []
        for name in gripper_body_names:
            b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if b_id != -1: # Ensure the name exists in this specific XML configuration
                gripper_body_ids.append(b_id)
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            
            if self.scenario == "hit_floor":
                # Check if ANY part of our hand assembly group hits the floor
                if body1 in gripper_body_ids or body2 in gripper_body_ids:
                    c_forces = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, c_forces)
                    total_force += c_forces[0]
                    
            elif self.scenario == "push_block":
                # Check if the collision is between our red block AND any part of the hand assembly group
                is_touching = (body1 in gripper_body_ids and body2 == self.block_id) or \
                              (body2 in gripper_body_ids and body1 == self.block_id)
                              
                if is_touching:
                    c_forces = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, c_forces)
                    total_force += c_forces[0]
                    
        return total_force

    def controller_callback(self, model, data):
        """The master loop runner tied directly to MuJoCo's internal heartbeats"""
        # Execute the movement steps
        # self._apply_control_policy()
        
        # Log metrics based on downsampled intervals
        if self.step_counter % self.downsample_factor == 0:
            force_value = self._extract_contact_forces()
            self.time_history.append(self.data.time)
            self.force_history.append(force_value)
            
        self.step_counter += 1

    def run(self):
        """Launches the window execution"""
        print(f"Booting up environment factory running: [{self.scenario.upper()}]")
        
        mujoco.set_mjcb_control(self.controller_callback)
        mujoco.viewer.launch(self.model, self.data)
        
        self.plot_results()

    def plot_results(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.time_history, self.force_history, color='purple', linewidth=2)
        plt.title(f'Force Estimation Timeline - Scenario: {self.scenario}', fontsize=12, fontweight='bold')
        plt.xlabel('Seconds')
        plt.ylabel('Normal Force (N)')
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    env = FrankaForceEnv(scenario="push_block") # "hit_floor" or "push_block"
    env.run()