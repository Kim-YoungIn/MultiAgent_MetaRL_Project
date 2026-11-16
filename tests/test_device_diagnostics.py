"""Runtime availability is distinct from installed CUDA libraries and drivers."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mamrl.diagnose_device import diagnose
import mamrl.diagnose_device as device_check


class DeviceDiagnostics(unittest.TestCase):
    def test_cpu_binary_is_identified_without_calling_cuda(self):
        torch=SimpleNamespace(__version__='2.test+cpu',__file__='torch.py',version=SimpleNamespace(cuda=None))
        with patch.object(device_check.importlib,'import_module',return_value=torch),patch.object(device_check.shutil,'which',return_value=None),patch.object(device_check.Path,'is_file',return_value=False):
            result=diagnose(smoke=True)
        self.assertEqual(result['status'],'cpu_build');self.assertFalse(result['cuda_available'])
        self.assertIsNone(result['cuda_smoke_passed'])

    def test_installed_cuda_does_not_imply_access(self):
        torch=SimpleNamespace(__version__='2.test+cu128',__file__='torch.py',version=SimpleNamespace(cuda='12.8'),
                              cuda=SimpleNamespace(is_available=lambda:False))
        with patch.object(device_check.importlib,'import_module',return_value=torch),patch.object(device_check.shutil,'which',return_value='/nvidia-smi'),patch.object(device_check.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='GPU access blocked')):
            result=diagnose(smoke=True)
        self.assertEqual(result['status'],'cuda_inaccessible')
        self.assertEqual(result['driver_probe']['stderr'],'GPU access blocked')
        self.assertIsNone(result['cuda_smoke_passed'])

    def test_missing_torch_does_not_break_core_diagnostics(self):
        with patch.object(device_check.importlib,'import_module',side_effect=ImportError),patch.object(device_check.shutil,'which',return_value=None),patch.object(device_check.Path,'is_file',return_value=False):
            result=diagnose()
        self.assertEqual(result['status'],'torch_missing')
