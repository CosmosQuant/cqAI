import pandas as pd
import numpy as np
from src.features import (Feature, ReLU, Clip, Scale, Sign, Abs, Lag, Smooth, Crossover, 
                         apply_operators, generate_test_data)

# Generate test data and base feature
test_data = generate_test_data(100, 42)
print(f'Generated test data: {test_data.shape}')

# Create base feature
base_feature = Feature(test_data, feature_type='maratio', short=3, long=10)
base_feature.calculate()
f1 = base_feature.get_feature()
print(f'Base feature range: [{f1.min():.3f}, {f1.max():.3f}]')

# Test ReLU
relu_op = ReLU(threshold=0.0, direction='high')
f2 = relu_op.apply(f1)
print(f'ReLU: {relu_op.get_name()}')
print(f'  Positive values: {(f2 > 0).sum()}, Zero/negative: {(f2 <= 0).sum()}')

# Test Crossover
crossover_op = Crossover(threshold=0.0, direction='both')
f3 = crossover_op.apply(f1)
print(f'Crossover: {crossover_op.get_name()}')
print(f'  Signal events: {(f3 != 0).sum()}')

# Test batch processing
operators = [ReLU(threshold=0.0, direction='high'), Clip(low=0.0, high=0.05)]
f4 = apply_operators(f1, operators)
print(f'Pipeline result range: [{f4.min():.3f}, {f4.max():.3f}]')
