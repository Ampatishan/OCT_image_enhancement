import unittest
import torch
from hydra import initialize, compose
from models import get_model
class TestUNETRHydra(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.img_size = 512

    def test_unetr_from_hydra_config(self):
        # Initialize Hydra config context
        with initialize(config_path="../configs", job_name="test_unetr"):
            cfg = compose(config_name="unetr_config")

            # Instantiate the model using Hydra
            model = get_model(cfg.model)

            # Run a forward pass
            x = torch.randn(self.batch_size, cfg.model.param.in_channels, self.img_size, self.img_size)
            out = model(x)

            # Assert output shape
            self.assertEqual(out.shape, (self.batch_size, cfg.model.param.out_channels, self.img_size, self.img_size))


if __name__ == '__main__':
    unittest.main()
