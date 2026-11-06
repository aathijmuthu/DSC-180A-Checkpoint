"""Scientific invariants, independent of remote downloads or CUDA."""
import tempfile
import unittest
from pathlib import Path

import torch

from checkpoint.common import (ROOT, cell_source, definitions, frechet, gaussian,
                               restore_rng, rng_state, save_torch, setup)
from checkpoint.mnist import denoising_loss, lab_module


class Experiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup('cpu')
        cls.mg = lab_module()

    def test_fid_identity_symmetry_and_translation(self):
        torch.manual_seed(21)
        x = torch.randn(100, 6, dtype=torch.double)
        mu, cov = gaussian(x)
        self.assertAlmostEqual(frechet(mu, cov, mu, cov), 0, places=8)
        shift = torch.arange(6, dtype=torch.double) / 5
        translated = gaussian(x + shift)
        self.assertAlmostEqual(frechet(mu, cov, *translated), float(shift.square().sum()), places=8)
        other = gaussian(torch.randn(150, 6, dtype=torch.double) * 1.5)
        self.assertAlmostEqual(frechet(mu, cov, *other), frechet(*other, mu, cov), places=8)

    def test_null_guidance_is_independent_of_request(self):
        torch.manual_seed(0)
        net = self.mg.UNet(ch=8).eval().requires_grad_(False)
        a, _ = self.mg.sample(net, 4, torch.zeros(4, dtype=torch.long), 'cpu', steps=2, w=0, seed=7, batch=2)
        b, _ = self.mg.sample(net, 4, torch.full((4,), 7), 'cpu', steps=2, w=0, seed=7, batch=2)
        self.assertTrue(torch.equal(a, b))
        c, _ = self.mg.sample(net, 4, torch.zeros(4, dtype=torch.long), 'cpu', steps=2, w=1, seed=7)
        self.assertFalse(torch.allclose(a, c))

    def test_training_label_dropout_updates_null_embedding(self):
        torch.manual_seed(1)
        net = self.mg.UNet(ch=8)
        loss = denoising_loss(net, torch.zeros(2, 1, 28, 28), torch.tensor([1, 7]), dropout=1.)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(net.cemb.weight.grad[10].abs().sum()), 0.)
        self.assertEqual(float(net.cemb.weight.grad[:10].abs().sum()), 0.)

    def test_edm_sampler_lands_on_constant_clean_prediction(self):
        ns = dict(torch=torch, DEVICE=torch.device('cpu'))
        exec(definitions(cell_source(ROOT / 'dsc180-d32-cifar10-edm-lab/cifar10_edm_lab.ipynb', 7)), ns)
        sample = ns['edm_sample'](lambda x, s: torch.full_like(x, .25), n=2, seed=7)
        self.assertTrue(torch.allclose(sample, torch.full_like(sample, .25), atol=1e-6))

    def test_checkpoint_restores_optimizer_and_next_random_draw(self):
        torch.manual_seed(2)
        net = torch.nn.Linear(3, 1)
        opt = torch.optim.Adam(net.parameters())
        opt.zero_grad(); net(torch.ones(1, 3)).sum().backward(); opt.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            save_torch(path, dict(net=net.state_dict(), optimizer=opt.state_dict(), rng=rng_state()))
            expected = torch.randn(3)
            recovered = torch.load(path, weights_only=False)
            restored = torch.nn.Linear(3, 1)
            restored_opt = torch.optim.Adam(restored.parameters())
            restored.load_state_dict(recovered['net'])
            restored_opt.load_state_dict(recovered['optimizer'])
            restore_rng(recovered['rng'])
            self.assertTrue(torch.equal(expected, torch.randn(3)))
            for a, b in zip(net.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(len(restored_opt.state), len(opt.state))


if __name__ == '__main__':
    unittest.main()
