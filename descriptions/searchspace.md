- **class SuperResK1K3K1**: This class defines a residual block with convolutional layers having kernel sizes of 1x3x1. It is designed for applications that require smaller kernel sizes to extract local features while maintaining a residual connection for improved gradient flow.
- The `__init__(self, in_channels, out_channels, stride, bottleneck_channels, sub_layers)` parameters are:
    - `in_channels`: Number of input channels.
    - `out_channels`: Number of output channels.
    - `stride`: Stride length for the convolutional layers.
    - `bottleneck_channels`: Number of channels in the bottleneck layer for reduced dimensionality.
    - `sub_layers`: Number of sub-layers in the block.
- Function `forward` code:
```python
def forward(self, x):
    output = x
    for block in self.block_list:
        output = block(output)
    return output
```

- **class SuperResK1K5K1**: This class defines a residual block with convolutional layers having kernel sizes of 1x5x1. The larger kernel size helps capture broader feature patterns while preserving the residual structure for gradient flow.
- The `__init__(self, in_channels, out_channels, stride, bottleneck_channels, sub_layers)` parameters are:
    - `in_channels`: Number of input channels.
    - `out_channels`: Number of output channels.
    - `stride`: Stride length for the convolutional layers.
    - `bottleneck_channels`: Number of channels in the bottleneck layer.
    - `sub_layers`: Number of sub-layers in the block.
- Function `forward` code:
```python
def forward(self, x):
    output = x
    for block in self.block_list:
        output = block(output)
    return output
```

- **class SuperResK1K7K1**: This class defines a residual block with convolutional layers having kernel sizes of 1x7x1. This block is suitable for tasks requiring large receptive fields to capture more contextual information while still maintaining the residual connection.
- The `__init__(self, in_channels, out_channels, stride, bottleneck_channels, sub_layers)` parameters are:
    - `in_channels`: Number of input channels.
    - `out_channels`: Number of output channels.
    - `stride`: Stride length for the convolutional layers.
    - `bottleneck_channels`: Number of channels in the bottleneck layer.
    - `sub_layers`: Number of sub-layers in the block.
- Function `forward` code:
```python
def forward(self, x):
    output = x
    for block in self.block_list:
        output = block(output)
    return output
```

- **class SuperResK3K3**: This class defines a residual block with convolutional layers having kernel sizes of 3x3. This block type is commonly used in deep learning architectures to capture mid-range features in images, providing a balanced receptive field for various types of image data.
- The `__init__(self, in_channels, out_channels, stride, sub_layers)` parameters are:
    - `in_channels`: Number of input channels.
    - `out_channels`: Number of output channels.
    - `stride`: Stride length for the convolutional layers.
    - `sub_layers`: Number of sub-layers in the block.
- Function `forward` code:
```python
def forward(self, x):
    output = x
    for block in self.block_list:
        output = block(output)
    return output
```

- **class SuperResK5K5**: This class defines a residual block with convolutional layers having kernel sizes of 5x5. The larger kernel size is suitable for capturing more global features in images, allowing the network to better understand broader patterns.
- The `__init__(self, in_channels, out_channels, stride, sub_layers)` parameters are:
    - `in_channels`: Number of input channels.
    - `out_channels`: Number of output channels.
    - `stride`: Stride length for the convolutional layers.
    - `sub_layers`: Number of sub-layers in the block.
- Function `forward` code:
```python
def forward(self, x):
    output = x
    for block in self.block_list:
        output = block(output)
    return output
```

- **class SuperResK7K7**: This class defines a residual block with convolutional layers having kernel sizes of 7x7. This block is suitable for very large receptive fields and is typically used when the task requires capturing extensive context or patterns across larger image areas.
- The `__init__(self, in_channels, out_channels, stride, sub_layers)` parameters are:
    - `in_channels`: Number of input channels.
    - `out_channels`: Number of output channels.
    - `stride`: Stride length for the convolutional layers.
    - `sub_layers`: Number of sub-layers in the block.
- Function `forward` code:
```python
def forward(self, x):
    output = x
    for block in self.block_list:
        output = block(output)
    return output
```