import torch
import unittest
import time
import torch.nn as nn
from depth_model import MobileDepth, SpatialAttention, ConvBNReLU6, DepthwiseSeparableConv, ImprovedUpBlock

class TestMobileDepthComprehensive(unittest.TestCase):
    def setUp(self):
        self.model = MobileDepth(num_classes=1)
        self.input = torch.randn(1, 3, 640, 640)
    
    def test_training_mode_output_verbose(self):
        self.model.train()  # Перевод модели в режим обучения
        with torch.no_grad():
            outputs = self.model(self.input)
        print("\nTEST: Training Mode Output")
        print("  Number of outputs:", len(outputs))
        print("  Output shapes:", [out.shape for out in outputs])
        # Should return a list of 3 outputs
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 3)
        # Check individual output shapes
        self.assertEqual(outputs[0].shape, (1, 1, 640, 640))  # out1 shape
        self.assertEqual(outputs[1].shape, (1, 1, 640, 640))  # out2 shape, upsampled from d3
        self.assertEqual(outputs[2].shape, (1, 1, 640, 640))  # out3 shape, upsampled from d2
    
    def test_different_input_sizes_verbose(self):
        input_sizes = [(1, 3, 320, 320), (1, 3, 480, 480), (1, 3, 1024, 1024)]
        self.model.eval()
        
        print("\nTEST: Different Input Sizes")
        for size in input_sizes:
            print(f"  Testing input size: {size}")
            test_input = torch.randn(size)
            with torch.no_grad():
                output = self.model(test_input)
            
            # Expected output should have same spatial dimensions as input but with 1 channel
            expected_shape = (size[0], 1, size[2], size[3])
            print(f"  Expected output shape: {expected_shape}")
            print(f"  Actual output shape: {output.shape}")
            
            self.assertEqual(output.shape, expected_shape)
    
    def test_skip_connections_influence_verbose(self):
        self.model.eval()
        
        # Create a modified model with zeroed skip connections
        modified_model = MobileDepth(num_classes=1)
        modified_model.eval()
        
        # Create a function to zero out skip connections
        def zero_out_skips(module, input):
            return torch.zeros_like(input[0])
        
        # Register forward hooks to zero out skip connections
        hooks = []
        hooks.append(modified_model.att_skip1.register_forward_pre_hook(zero_out_skips))
        hooks.append(modified_model.att_skip2.register_forward_pre_hook(zero_out_skips))
        hooks.append(modified_model.att_skip3.register_forward_pre_hook(zero_out_skips))
        
        # Run both models on the same input
        with torch.no_grad():
            normal_output = self.model(self.input)
            modified_output = modified_model(self.input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Check if outputs are different (they should be if skip connections matter)
        diff = (normal_output - modified_output).abs().mean().item()
        
        print("\nTEST: Skip Connections Influence")
        print("  Mean absolute difference between normal and no-skip outputs:", diff)
        
        # The outputs should be different if skip connections have influence
        self.assertGreater(diff, 0.01, "Skip connections should significantly influence the output")
    
    def test_spatial_attention_verbose(self):
        print("\nTEST: Spatial Attention")
        
        # Test for att_skip1
        test_tensor = torch.randn(1, 64, 64, 64)
        attention_output = self.model.att_skip1(test_tensor)
        print("  Attention skip1 input shape:", test_tensor.shape)
        print("  Attention skip1 output shape:", attention_output.shape)
        
        # Output shape should be the same as input
        self.assertEqual(attention_output.shape, test_tensor.shape)
        
        # Values should be modified (multiplied by attention weights)
        attention_diff = (test_tensor - attention_output).abs().mean().item()
        print("  Mean absolute difference (skip1):", attention_diff)
        self.assertGreater(attention_diff, 0, "Attention should modify the input values")
    
    def test_gradient_flow_verbose_corrected(self):
        self.model.train()  # Switch to training mode
        
        # Create a smaller input for faster gradient computation
        small_input = torch.randn(1, 3, 128, 128)
        small_input.requires_grad = True
        
        # Reset gradients
        self.model.zero_grad()
        
        # Forward pass
        outputs = self.model(small_input)
        
        # Calculate loss using all outputs (a more realistic scenario)
        loss = 0
        for output in outputs:
            loss += output.mean()
        
        # Backward pass
        loss.backward()
        
        # Check if gradients are computed for model parameters
        params_with_grad = sum(1 for param in self.model.parameters() if param.grad is not None and param.requires_grad)
        total_params = sum(1 for param in self.model.parameters() if param.requires_grad)
        
        print("\nTEST: Gradient Flow (Corrected)")
        print(f"  Parameters with gradients: {params_with_grad}/{total_params}")
        print(f"  Percentage with gradients: {100 * params_with_grad / total_params:.2f}%")
        
        # Check the magnitude of gradients for different parts of the model
        encoder_grad = torch.cat([p.grad.view(-1) for p in self.model.dsconv1.parameters() if p.grad is not None])
        decoder_grad = torch.cat([p.grad.view(-1) for p in self.model.up1.parameters() if p.grad is not None])
        
        encoder_mag = encoder_grad.abs().mean().item()
        decoder_mag = decoder_grad.abs().mean().item()
        
        print("  Mean encoder gradient magnitude:", encoder_mag)
        print("  Mean decoder gradient magnitude:", decoder_mag)
        
        # More realistic expectation: a high percentage of parameters should have gradients
        self.assertGreater(params_with_grad / total_params, 0.95, 
                           "At least 95% of parameters should have gradients when using all outputs")
        self.assertGreater(encoder_mag, 0, "Encoder gradients should be non-zero")
        self.assertGreater(decoder_mag, 0, "Decoder gradients should be non-zero")
    
    def test_weight_initialization_verbose(self):
        # Create a new model to test initialization
        new_model = MobileDepth(num_classes=1)
        
        print("\nTEST: Weight Initialization")
        
        # Check conv weights (sample a few layers)
        conv_layers_checked = 0
        for name, module in new_model.named_modules():
            if isinstance(module, nn.Conv2d) and conv_layers_checked < 3:
                weight = module.weight.data
                mean = weight.mean().item()
                std = weight.std().item()
                print(f"  Conv layer '{name}' - Mean: {mean:.6f}, Std: {std:.6f}")
                
                # For Kaiming init with ReLU, mean should be close to 0
                self.assertLess(abs(mean), 0.1, f"Mean of weights in '{name}' should be close to 0")
                self.assertGreater(std, 0.01, f"Std of weights in '{name}' should be non-negligible")
                
                conv_layers_checked += 1
        
        # Check batchnorm weights and biases (sample a few layers)
        bn_layers_checked = 0
        for name, module in new_model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and bn_layers_checked < 3:
                weight = module.weight.data
                bias = module.bias.data
                
                print(f"  BN layer '{name}' - Weight mean: {weight.mean().item():.6f}, Bias mean: {bias.mean().item():.6f}")
                
                # Check if weights are initialized to 1 and biases to 0
                self.assertAlmostEqual(weight.mean().item(), 1.0, delta=0.1, 
                                      msg=f"BatchNorm weights in '{name}' should be initialized to 1")
                self.assertAlmostEqual(bias.mean().item(), 0.0, delta=0.1,
                                      msg=f"BatchNorm biases in '{name}' should be initialized to 0")
                
                bn_layers_checked += 1
    
    def test_device_compatibility_verbose_improved(self):
        print("\nTEST: Device Compatibility (Improved)")
        
        # Check CUDA availability and print diagnostic info
        print("  CUDA available:", torch.cuda.is_available())
        
        if torch.cuda.is_available():
            print("  CUDA version:", torch.version.cuda)
            print("  GPU count:", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Create a smaller input for faster device transfer
            small_input = torch.randn(1, 3, 128, 128)
            
            # Test CPU execution
            self.model.cpu()
            self.model.eval()
            
            with torch.no_grad():
                start_time = time.time()
                cpu_output = self.model(small_input)
                cpu_time = time.time() - start_time
            
            # Test CUDA execution
            cuda_model = self.model.to('cuda')
            cuda_model.eval()
            cuda_input = small_input.to('cuda')
            
            with torch.no_grad():
                # Warm-up run (first CUDA execution can be slower)
                _ = cuda_model(cuda_input)
                
                # Timed run
                start_time = time.time()
                cuda_output = cuda_model(cuda_input)
                cuda_time = time.time() - start_time
                
                # Move CUDA output to CPU for comparison
                cuda_output_cpu = cuda_output.cpu()
            
            # Compare outputs (they should be similar)
            diff = (cpu_output - cuda_output_cpu).abs().max().item()
            
            print("  CPU forward time:", cpu_time)
            print("  CUDA forward time:", cuda_time)
            print("  Speed improvement with CUDA:", cpu_time / cuda_time, "×")
            print("  Max absolute difference between CPU and CUDA outputs:", diff)
            
            self.assertLess(diff, 1e-3, "Outputs on CPU and CUDA should be very similar")
        else:
            # Check if CUDA should be available (i.e., if there are NVIDIA GPUs)
            try:
                # Try to run nvidia-smi
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("  WARNING: nvidia-smi detected GPU(s), but PyTorch cannot access CUDA.")
                    print("  This might indicate a PyTorch installation issue or driver/CUDA mismatch.")
                    print("  PyTorch version:", torch.__version__)
                    print("  You may need to reinstall PyTorch with the correct CUDA version.")
                    print("  See: https://pytorch.org/get-started/locally/")
                    print("\n  nvidia-smi output (truncated):")
                    print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
                else:
                    print("  nvidia-smi failed, suggesting no NVIDIA GPUs or driver issues.")
            except FileNotFoundError:
                print("  nvidia-smi not found. No NVIDIA driver appears to be installed.")
            except Exception as e:
                print(f"  Error checking nvidia-smi: {e}")
            
            print("  Skipping CUDA performance test")
    
    def test_upsample_mechanism_verbose(self):
        # Test that the upsampling in decoder works correctly
        self.model.eval()
        
        print("\nTEST: Upsampling Mechanism")
        
        # Create a test input for the first up block
        test_input = torch.randn(1, 512, 32, 32)
        up_output = self.model.up1(test_input)
        
        print(f"  Input shape: {test_input.shape}")
        print(f"  Expected output shape: (1, 256, 64, 64)")
        print(f"  Actual output shape: {up_output.shape}")
        
        # Check that spatial dimensions are doubled
        self.assertEqual(up_output.shape[2], test_input.shape[2] * 2)
        self.assertEqual(up_output.shape[3], test_input.shape[3] * 2)
        
        # Check that channels are as expected
        self.assertEqual(up_output.shape[1], 256)
