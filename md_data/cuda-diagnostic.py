import torch
import sys
import os
import subprocess

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error running command: {e}"

def check_cuda():
    print("=== PyTorch CUDA Diagnostic ===")
    
    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available through PyTorch
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # CUDA version as reported by PyTorch
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
        
        # Number of GPUs
        print(f"GPU count: {torch.cuda.device_count()}")
        
        # GPU name
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("PyTorch cannot find CUDA. Let's check the system:")
    
    # Environment variables
    print("\n=== CUDA Environment Variables ===")
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    print(f"CUDA_HOME: {cuda_home}")
    
    path = os.environ.get('PATH', '')
    cuda_in_path = False
    if path:
        path_entries = path.split(':')
        for entry in path_entries:
            if 'cuda' in entry:
                print(f"CUDA entry in PATH: {entry}")
                cuda_in_path = True
    if not cuda_in_path:
        print("No CUDA directories found in PATH")
    
    print("\nLD_LIBRARY_PATH:")
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    cuda_in_ld = False
    if ld_library_path:
        ld_entries = ld_library_path.split(':')
        for entry in ld_entries:
            if 'cuda' in entry:
                print(f"CUDA entry in LD_LIBRARY_PATH: {entry}")
                cuda_in_ld = True
    if not cuda_in_ld:
        print("No CUDA directories found in LD_LIBRARY_PATH")
    
    # Check if nvidia-smi is available
    print("\n=== NVIDIA System Management Interface ===")
    nvidia_smi = run_command("which nvidia-smi")
    if nvidia_smi and "Error" not in nvidia_smi:
        print(f"nvidia-smi found at: {nvidia_smi}")
        # Run nvidia-smi
        smi_output = run_command("nvidia-smi")
        print("nvidia-smi output:")
        print(smi_output)
    else:
        print("nvidia-smi not found in PATH or failed to run")
    
    # Check for CUDA toolkit
    print("\n=== CUDA Toolkit ===")
    nvcc = run_command("which nvcc")
    if nvcc and "Error" not in nvcc:
        print(f"nvcc found at: {nvcc}")
        nvcc_version = run_command("nvcc --version")
        print("nvcc version:")
        print(nvcc_version)
    else:
        print("nvcc not found in PATH or failed to run")
    
    # Check for libcudart
    print("\n=== CUDA Runtime Library ===")
    libcudart = run_command("ldconfig -p | grep libcudart")
    if libcudart and "Error" not in libcudart:
        print("CUDA Runtime Library found:")
        print(libcudart)
    else:
        print("CUDA Runtime Library not found with ldconfig or command failed")
    
    # Check for the pytorch installation details
    print("\n=== PyTorch Installation Details ===")
    torch_location = run_command(f"{sys.executable} -c 'import torch; print(torch.__file__)'")
    print(f"PyTorch is installed at: {torch_location}")
    
    # Check if PyTorch was built with CUDA
    torch_built_with_cuda = run_command(f"{sys.executable} -c 'import torch; print(\"Built with CUDA:\" if torch.backends.cudnn.enabled else \"Not built with CUDA\")'")
    print(f"PyTorch CUDA status: {torch_built_with_cuda}")
    
    print("\n=== Recommendations ===")
    if not torch.cuda.is_available():
        print("1. Check if your PyTorch installation includes CUDA support:")
        print("   You may need to reinstall PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu12X")
        print("   (Replace 12X with your CUDA version, e.g. 118, 121, etc.)")
        print("\n2. Make sure your NVIDIA drivers are properly installed")
        print("\n3. Check if your GPU supports CUDA")
        print("\n4. Ensure environment variables are set correctly:")
        print("   - CUDA_HOME or CUDA_PATH should point to your CUDA installation")
        print("   - PATH should include CUDA bin directory")
        print("   - LD_LIBRARY_PATH should include CUDA lib64 directory")
        print("\n5. If running in Docker, make sure the container has GPU access:")
        print("   - Use the --gpus flag: docker run --gpus all ...")
        print("   - Make sure nvidia-docker is installed if needed")
        
if __name__ == "__main__":
    check_cuda()
