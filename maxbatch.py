import torch
import torch.nn as nn
import gc

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)

def find_max_batch_size(model, input_shape, max_batch_size=1024, step=32):
    device = next(model.parameters()).device
    batch_size = step

    while batch_size <= max_batch_size:
        try:
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()

            # Create input tensor
            x = torch.randn(batch_size, *input_shape, device=device)

            # Forward pass
            output = model(x)

            # Backward pass
            loss = output.sum()
            loss.backward()

            print(f"Batch size {batch_size} successful")
            batch_size += step
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Max batch size: {batch_size - step}")
                return batch_size - step
            else:
                raise e

    print(f"Max batch size: {max_batch_size}")
    return max_batch_size

# Usage
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = SimpleModel().to(device)
max_batch = find_max_batch_size(model, input_shape=(100,))
print(f"Maximum batch size: {max_batch}")
