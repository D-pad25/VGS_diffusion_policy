from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

class XArm6LowdimRunner(BaseLowdimRunner):
    def __init__(self, output_dir=None):
        self.output_dir = output_dir

    def run(self, *args, **kwargs):
        print("[XArm6LowdimRunner] run() called — this is a dummy runner for training only")
        return {}
