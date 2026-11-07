"""Scientific invariants, independent of remote downloads or CUDA."""
import tempfile
import unittest
import json
from argparse import Namespace
from pathlib import Path

import torch

from checkpoint.common import (ROOT, cell_source, definitions, frechet, gaussian,
                               notebook_code_hash, restore_rng, rng_state, save_torch, setup)
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

    def test_cifar_figures_save_with_current_matplotlib(self):
        import matplotlib.pyplot as plt
        from checkpoint.cifar import Lab
        old_show = plt.show
        try:
            with tempfile.TemporaryDirectory() as directory:
                lab = Lab(Namespace(out=directory))
                plt.subplots()[1].plot([0, 1], [0, 1])
                plt.show()
                self.assertTrue((Path(directory) / 'figure.png').is_file())
        finally:
            plt.show = old_show

    def test_cifar_sweep_recovers_completed_weights_after_failure(self):
        import json
        import matplotlib.pyplot as plt
        from checkpoint.cifar import Lab
        old_show = plt.show
        try:
            with tempfile.TemporaryDirectory() as directory:
                lab = Lab(Namespace(out=directory, batch=2, n_eval=20))
                real = torch.randn(20, 3, generator=torch.Generator().manual_seed(3)).double()
                labels = torch.arange(20) % 10
                calls, feature_calls, fail = [], [], [True]
                def generate(w):
                    calls.append(w)
                    if w == 2 and fail[0]:
                        raise RuntimeError('simulated interruption')
                    return real + w, labels
                def features(images):
                    feature_calls.append(len(images))
                    return images
                lab.ns.update(train_images=real, train_labels=labels, W=[0, 1, 2],
                              gaussian=gaussian, frechet=frechet, features=features,
                              generate=generate, classify=lambda images: torch.zeros(len(images), dtype=torch.long),
                              fid_class=lambda feats, requested: 0.)
                lab.ns['MU_REF'], lab.ns['SIGMA_REF'] = gaussian(real)
                lab.cell = lambda *args, **kwargs: None
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    lab.sweep()
                saved = json.loads((Path(directory) / 'cfg_sweep_n20.json').read_text())
                self.assertEqual(set(saved['results']), {'0', '1'})
                self.assertFalse(saved['complete'])
                completed_feature_calls = len(feature_calls)
                fail[0] = False
                lab.sweep()
                saved = json.loads((Path(directory) / 'cfg_sweep_n20.json').read_text())
                self.assertTrue(saved['complete'])
                self.assertEqual(len(feature_calls), completed_feature_calls + 1)
                self.assertAlmostEqual(saved['results']['2']['fid'], 12., places=7)
                self.assertEqual(calls, [0, 1, 2, 2, 0])
        finally:
            plt.show = old_show

    def test_notebook_answers_do_not_invalidate_equivalent_sweep_code(self):
        from checkpoint.cifar import (NOTEBOOK, SWEEP_CELLS, ORIGINAL_NOTEBOOK_HASH,
                                      ORIGINAL_SWEEP_CODE_HASH, cached_protocol)
        digest = notebook_code_hash(NOTEBOOK, SWEEP_CELLS)
        self.assertEqual(digest, ORIGINAL_SWEEP_CODE_HASH)
        notebook = json.loads(NOTEBOOK.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'lab.ipynb'
            notebook['cells'][0]['source'].append('\nAn additional written answer.')
            notebook['cells'][17]['source'].append('\n# Extra exercise commentary')
            path.write_text(json.dumps(notebook))
            self.assertEqual(notebook_code_hash(path, SWEEP_CELLS), digest)
            notebook['cells'][15]['source'].append('\nW = [0, 1]')
            path.write_text(json.dumps(notebook))
            self.assertNotEqual(notebook_code_hash(path, SWEEP_CELLS), digest)
        saved = dict(batch=32, n_eval=1000, steps=18, seed_base=1000,
                     notebook_sha256=ORIGINAL_NOTEBOOK_HASH)
        current = dict(batch=32, n_eval=1000, steps=18, seed_base=1000,
                       notebook_code_sha256=digest, fid_device='cuda')
        self.assertEqual(cached_protocol(saved, current, {'device': 'cuda'}), current)
        self.assertNotEqual(cached_protocol(saved, current, {'device': 'cpu'}), current)
        changed = dict(current, notebook_code_sha256='changed implementation')
        self.assertNotEqual(cached_protocol(saved, changed, {'device': 'cuda'}), changed)

    def test_rejected_sweep_cache_preserves_historical_manifest(self):
        import matplotlib.pyplot as plt
        from checkpoint.cifar import Lab, ORIGINAL_NOTEBOOK_HASH
        previous_show = plt.show
        try:
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                historical = {'device': 'cuda', 'git_commit': 'original experiment'}
                (out / 'manifest_sweep.json').write_text(json.dumps(historical))
                (out / 'sweep_protocol.json').write_text(json.dumps(dict(
                    batch=32, n_eval=1000, steps=18, seed_base=1000,
                    notebook_sha256=ORIGINAL_NOTEBOOK_HASH)))
                (out / 'class_reference.pt').touch()
                lab = Lab(Namespace(out=directory, task='sweep', batch=32, n_eval=1000))
                lab.pending_sweep_manifest = {'device': 'cpu'}
                lab.ns['FID_DEVICE'] = 'cpu'
                with self.assertRaisesRegex(ValueError, 'Sampling configuration changed'):
                    lab.sweep()
                self.assertEqual(json.loads((out / 'manifest_sweep.json').read_text()), historical)
        finally:
            plt.show = previous_show


if __name__ == '__main__':
    unittest.main()
