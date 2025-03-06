import torch
import unittest
from depth_model import MobileDepth

class TestMobileDepthVerbose(unittest.TestCase):
    def setUp(self):
        self.model = MobileDepth(num_classes=1)
        self.model.eval()  # Перевод модели в режим инференса
        self.input = torch.randn(1, 3, 640, 640)
    
    def test_output_shape_verbose(self):
        with torch.no_grad():
            output = self.model(self.input)
        expected_shape = (1, 1, 640, 640)
        print("\nTEST: Output Shape")
        print("  Input shape:           ", self.input.shape)
        print("  Expected output shape: ", expected_shape)
        print("  Actual output shape:   ", output.shape)
        self.assertEqual(output.shape, expected_shape)

    def test_output_type_verbose(self):
        with torch.no_grad():
            output = self.model(self.input)
        print("\nTEST: Output Type")
        print("  Type of output: ", type(output))
        self.assertIsInstance(output, torch.Tensor)

    def test_forward_no_exception_verbose(self):
        try:
            with torch.no_grad():
                output = self.model(self.input)
            print("\nTEST: Forward Pass")
            print("  Forward pass выполнен успешно.")
            print("  Пример выходных значений (min, max, mean):", output.min().item(), output.max().item(), output.mean().item())
        except Exception as e:
            self.fail(f"Forward pass вызвал исключение: {e}")

    def test_output_range_verbose(self):
        with torch.no_grad():
            output = self.model(self.input)
        min_val = output.min().item()
        max_val = output.max().item()
        diff = max_val - min_val
        print("\nTEST: Output Range")
        print("  Output min:      ", min_val)
        print("  Output max:      ", max_val)
        print("  Difference:      ", diff)
        self.assertTrue(diff > 0, "Диапазон значений должен быть больше нуля")

    def test_model_parameters_verbose(self):
        params = list(self.model.parameters())
        total_params = sum(p.numel() for p in params)
        print("\nTEST: Model Parameters")
        print("  Number of parameter groups: ", len(params))
        print("  Total number of parameters: ", total_params)
        self.assertTrue(len(params) > 0, "Модель не содержит параметров")
        self.assertTrue(total_params > 0, "Количество параметров модели должно быть больше 0")

if __name__ == "__main__":
    unittest.main(verbosity=2)
