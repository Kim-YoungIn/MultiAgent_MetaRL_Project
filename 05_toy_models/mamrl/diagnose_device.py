"""Report the current process's PyTorch runtime, driver access and CUDA execution.

Run in the same terminal/environment as training. Restricted processes can see a
different device set and Python package build than a normal WSL terminal.
"""
import argparse
import importlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def diagnose(*, smoke=False):
    report={'python':sys.executable,'python_version':platform.python_version(),
            'platform':platform.platform(),'wsl_device_visible':Path('/dev/dxg').exists(),
            'nvidia_device_visible':Path('/dev/nvidia0').exists(),
            'torch':None,'cuda_available':False,'cuda_smoke_passed':None,'issues':[]}
    executable=shutil.which('nvidia-smi')
    if executable is None and Path('/usr/lib/wsl/lib/nvidia-smi').is_file():
        executable='/usr/lib/wsl/lib/nvidia-smi'
    if executable:
        try:
            result=subprocess.run([executable,'--query-gpu=name,driver_version,memory.total',
                                   '--format=csv,noheader'],capture_output=True,text=True,timeout=10)
            report['driver_probe']={'returncode':result.returncode,'stdout':result.stdout.strip(),
                                    'stderr':result.stderr.strip()}
            if result.returncode!=0:
                report['issues'].append('Driver query failed in this execution context; check GPU access/driver restrictions.')
        except (OSError,subprocess.TimeoutExpired) as exc:
            report['driver_probe']={'error':str(exc)}
    else:
        report['driver_probe']={'error':'nvidia-smi not found; this alone does not prove CUDA is unavailable'}
    try:
        torch=importlib.import_module('torch')
    except ImportError:
        report['status']='torch_missing'
        report['issues'].append('Install a CUDA-enabled PyTorch binary in the training environment.')
        return report
    report['torch']={'version':str(torch.__version__),'module':str(torch.__file__),
                     'cuda_runtime':torch.version.cuda}
    if torch.version.cuda is None:
        report['status']='cpu_build'
        report['issues'].append('This process loaded a CPU-only PyTorch build. A CUDA runtime cannot replace GPU/driver access.')
        return report
    try:
        report['cuda_available']=torch.cuda.is_available()
        if not report['cuda_available']:
            report['status']='cuda_inaccessible'
            report['issues'].append('CUDA runtime is installed but GPU initialization is unavailable in this process.')
            return report
        report['devices']=[{'index':i,'name':torch.cuda.get_device_name(i),
                            'capability':list(torch.cuda.get_device_capability(i)),
                            'memory_bytes':torch.cuda.get_device_properties(i).total_memory}
                           for i in range(torch.cuda.device_count())]
        report['status']='ready'
        if smoke:
            values=torch.ones((8,8),device='cuda',dtype=torch.float32)
            observed=float((values@values).sum().item())
            torch.cuda.synchronize()
            report['cuda_smoke_passed']=observed==512.0
            if not report['cuda_smoke_passed']:
                report['status']='cuda_execution_failed';report['issues'].append('CUDA matrix smoke result differs from expected 512.')
    except (RuntimeError,OSError) as exc:
        report['status']='cuda_execution_failed';report['issues'].append(str(exc))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true',help='execute a small CUDA matrix operation')
    parser.add_argument('--require-cuda',action='store_true',help='fail when this process cannot use CUDA')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=diagnose(smoke=args.smoke)
    if args.output:
        from .run_files import write_json_exclusive
        args.output.parent.mkdir(parents=True,exist_ok=True)
        write_json_exclusive(args.output,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if args.require_cuda and report['status']!='ready':return 2
    return 0


if __name__=='__main__':sys.exit(main())
