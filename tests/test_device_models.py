import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, 'optional torch dependency')
class DeviceModels(unittest.TestCase):
    def test_resolver(self):
        from mamrl.devices import resolve_device
        self.assertEqual(resolve_device('cpu').type, 'cpu')
        with self.assertRaises(ValueError): resolve_device('npu')
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError): resolve_device('cuda')

    def test_batched_outputs_gradients_and_portable_checkpoint(self):
        from mamrl.action_set import ActionSetActorCritic
        from mamrl.environment import Observation
        from mamrl.recurrent import save_checkpoint, load_checkpoint
        m=ActionSetActorCritic(8,seed=7)
        observations=[Observation(0,1,0.5,False), Observation(2,0,-1,True)]
        hidden=torch.randn(2,3,8)
        logits, values, next_hidden=m.forward_observations(observations,hidden,n=3)
        for i,obs in enumerate(observations):
            l,v,h=m.forward_observation(obs,hidden[i],n=3)
            self.assertTrue(torch.allclose(l,logits[i],atol=1e-6))
            self.assertTrue(torch.allclose(v,values[i],atol=1e-6))
            self.assertTrue(torch.allclose(h,next_hidden[i],atol=1e-6))
        (logits.sum()+values.sum()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'p.pt';save_checkpoint(m,path); loaded=load_checkpoint(path,device='cpu')
            self.assertTrue(all(torch.equal(v,loaded.state_dict()[k]) for k,v in m.state_dict().items()))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_forward_and_portable_checkpoint(self):
        from mamrl.action_set import ActionSetActorCritic
        from mamrl.environment import Observation
        from mamrl.recurrent import save_checkpoint,load_checkpoint
        m=ActionSetActorCritic(8,seed=7).to('cuda')
        logits,_,_=m.forward_observation(Observation(0,1,0.,False),m.initial_hidden(n=3),n=3)
        self.assertEqual(logits.device.type,'cuda')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'p.pt';save_checkpoint(m,p);self.assertEqual(load_checkpoint(p).device.type,'cpu')
