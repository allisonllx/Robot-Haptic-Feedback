class Scenario:
    name = ""
    supports_interactive = False

    def initialize_state(self, env):
        pass

    def augment_model_spec(self, env, spec):
        pass

    def resolve_ids(self, env):
        pass

    def after_model_init(self, env):
        pass

    def apply_control(self, env):
        raise NotImplementedError

    def sample_forces(self, env):
        raise NotImplementedError

    def print_controls(self, env):
        pass

    def start_interactive(self, env):
        pass

    def stop_interactive(self, env):
        pass

    def viewer_key_callback(self, env, keycode):
        pass

    def before_interactive_step(self, env, dt):
        pass

    def update_interactive_viewer(self, env, viewer):
        pass

    def update_recording_scene(self, env, scene):
        pass


class TargetContactScenario(Scenario):
    def is_target_contact(self, env, contact, gripper_ids):
        raise NotImplementedError

    def sample_forces(self, env):
        gripper_ids = env._get_active_gripper_body_ids()
        in_contact = env._has_target_contact(gripper_ids)
        f_true = env._calculate_ground_truth_force(gripper_ids)
        f_est = env._estimate_virtual_force() if in_contact else 0.0
        return in_contact, f_true, f_est
