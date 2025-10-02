import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from dataclasses import dataclass

# Reuse data preparation from run_PPT.py
from run_PPT import prepare_data

@dataclass
class PPTConfigTorch:
    H_SET: Tuple[int, ...] = (1, 1)
    learning_rate: float = 0.005
    epochs: int = 1000
    lambda_p: float = 0.5      ## BCE loss weight
    lambda_L2: float = 1e-4    ## L2 regularization weight
    early_stop_threshold: float = 0.0001  ## Early stopping: loss change threshold
    early_stop_patience: int = 20        ## Early stopping: consecutive epochs
    score_threshold: float = 0.0          ## Signal generation: minimum score threshold
    p_threshold: float = 0.5              ## Signal generation: minimum probability threshold

class PPTModelTorch(nn.Module):
    def __init__(self, config: PPTConfigTorch):
        super().__init__()
        self.config = config
        
        # Parameters: Gate, Driver, Quality
        STD = 0.1
        init_param = lambda: nn.Parameter(torch.randn(1) * STD + 1)
        
        # Soft-binary version: 4 parameters (k1, k2, c1, c2)
        self.gate = nn.ParameterDict({
            'k1': init_param(),  # for p_run
            'k2': init_param(),  # for p_dd
            'c1': init_param(),  # for p_run
            'c2': init_param(),  # for p_dd
        })
        
        
        
        # Heads: Linear layers for each horizon
        self.heads = nn.ModuleDict({str(h): nn.ModuleDict({'reg': nn.Linear(1, 1), 'cls': nn.Linear(1, 1)}) for h in config.H_SET})
        
    def _sigmoid_rescale(self, x, lo, hi): return lo + (hi - lo) * F.sigmoid(x)
    def _tanh_rescale(self, x, scale): return scale * F.tanh(x)
    def _zscore_safe(self, x):
        finite = torch.isfinite(x)
        if not finite.any(): return torch.zeros_like(x)
        m, s = x[finite].mean(), x[finite].std(unbiased=False).clamp_min(1e-6)
        return (x - m) / s
        
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        
        runlen, dd_abs_z, dd_atr_z, LV_z, mom_z = features['runlen'], features['dd_abs_z'], features['dd_atr_z'], features['LV_z'], features['mom_z']
        
        # Soft-binary version: prob = σ(k1(r - c1)) · σ(k2(c2 - s0)) · 1_{r≥3}
        # where s0 can be ret4 or -ret4/ATR20 (z-scored)
        
        # Transform parameters
        k1 = F.softplus(self.gate["k1"]) + 1e-3  # k1 > 0
        k2 = F.softplus(self.gate["k2"]) + 1e-3  # k2 > 0
        c1 = self.gate["c1"]
        c2 = self.gate["c2"]
        
        # Compute s0 (using z-scored -ret4/ATR20 or just dd_abs_z)
        s0 = dd_abs_z  # Can also use: zscore(-ret4 / atr20)
        
        # Compute probability components
        p_run = F.sigmoid(k1 * (runlen - c1)) ##* F.sigmoid(k0 * (runlen - 3.5)) 
        p_dd = F.sigmoid(k2 * (s0 - c2))
        prob = p_run * p_dd
        
        # Use prob directly as input to heads (continuous signal)
        # Note: Decision threshold pi should be applied at signal generation stage, not in forward pass
        
        # Heads: predict μ and p for each horizon
        outputs = {}
        for h in self.config.H_SET:
            # Use prob (continuous) as feature for prediction heads
            # mu_hat = self.heads[str(h)]['reg'](prob.unsqueeze(-1)).squeeze(-1)
            # p_hat = F.sigmoid(self.heads[str(h)]['cls'](prob.unsqueeze(-1)).squeeze(-1))
            p_hat = prob
            mu_hat = prob
            outputs[h] = (mu_hat, p_hat)
        
        return outputs

def train_ppt_torch(model: PPTModelTorch, features: Dict, mu: Dict, p: Dict, mask: np.ndarray, config: PPTConfigTorch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Prepare tensors
    feat_tensors = {k: torch.FloatTensor(v[mask]).to(device) for k, v in features.items()}
    mu_tensors = {h: torch.FloatTensor(mu[h][mask]).to(device) for h in config.H_SET}
    p_tensors = {h: torch.FloatTensor(p[h][mask]).to(device) for h in config.H_SET}
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    huber_loss = nn.HuberLoss(delta=0.01)
    bce_loss = nn.BCELoss()
    
    # Track parameter history - automatically detect all parameters
    history = {'epochs': [], 'loss': []}
    if hasattr(model, 'gate'):
        for param_name in model.gate.keys():
            history[f'gate_{param_name}'] = []
    if hasattr(model, 'driver'):
        for param_name in model.driver.keys():
            history[f'driver_{param_name}'] = []
    if hasattr(model, 'quality'):
        for param_name in model.quality.keys():
            history[f'quality_{param_name}'] = []
    
    # Early stopping
    loss_history = []
    
    print(f'Training on {mask.sum()} samples, device: {device}')
    
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        outputs = model(feat_tensors)
        
        loss = 0.0
        for h in config.H_SET:
            mu_hat, p_hat = outputs[h]
            loss += (huber_loss(mu_hat, mu_tensors[h]) + config.lambda_p * bce_loss(p_hat, p_tensors[h]))
        
        loss += config.lambda_L2 * sum(p.pow(2).sum() for p in model.parameters())
        
        loss.backward()
        optimizer.step()
        
        # Early stopping check
        loss_history.append(loss.item())
        if len(loss_history) > config.early_stop_patience:
            recent_losses = loss_history[-config.early_stop_patience-1:]
            abs_changes = [abs(recent_losses[i] - recent_losses[i-1]) for i in range(1, len(recent_losses))]
            
            if all(abs_change < config.early_stop_threshold for abs_change in abs_changes):
                max_change = max(abs_changes)
                min_change = min(abs_changes)
                start_epoch = epoch - config.early_stop_patience
                print(f'Early stopping at epoch {epoch}:')
                print(f'  Loss range: {recent_losses[0]:.6f} -> {recent_losses[-1]:.6f}')
                print(f'  Change range: [{min_change:.6f}, {max_change:.6f}], all < {config.early_stop_threshold}')
                print(f'  Checked epochs: {start_epoch} to {epoch} ({config.early_stop_patience} consecutive)')
                break
        
        # Record raw parameters every 10 epochs
        if epoch % 10 == 0:
            with torch.no_grad():
                history['epochs'].append(epoch)
                history['loss'].append(loss.item())
                if hasattr(model, 'gate'):
                    for k in model.gate.keys():
                        history[f'gate_{k}'].append(model.gate[k].item())
                if hasattr(model, 'driver'):
                    for k in model.driver.keys():
                        history[f'driver_{k}'].append(model.driver[k].item())
                if hasattr(model, 'quality'):
                    for k in model.quality.keys():
                        history[f'quality_{k}'].append(model.quality[k].item())
        
        if epoch % 50 == 0: print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
    
    return model, history

def plot_parameter_analysis(history):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Parameter Evolution Analysis', fontsize=16)
    
    # Loss
    axes[0, 0].plot(history['epochs'], history['loss'], 'k-', linewidth=2)
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Automatically detect parameters
    gate_params = sorted([k for k in history.keys() if k.startswith('gate_')])
    driver_params = sorted([k for k in history.keys() if k.startswith('driver_')])
    quality_params = sorted([k for k in history.keys() if k.startswith('quality_')])
    
    # Gate parameters (first 2-3)
    colors_plot = ['b', 'r', 'g', 'm', 'c']
    for i, param in enumerate(gate_params[:min(3, len(gate_params))]):
        axes[0, 1].plot(history['epochs'], history[param], colors_plot[i], label=param.split('_')[1], linewidth=2)
    axes[0, 1].set_title('Gate Parameters')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gate parameters (remaining)
    for i, param in enumerate(gate_params[3:]):
        axes[0, 2].plot(history['epochs'], history[param], colors_plot[(i+3) % len(colors_plot)], label=param.split('_')[1], linewidth=2)
    axes[0, 2].set_title('Gate Parameters (cont.)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Driver parameters
    for i, param in enumerate(driver_params):
        axes[1, 0].plot(history['epochs'], history[param], colors_plot[i % len(colors_plot)], label=param.split('_')[1], linewidth=2)
    axes[1, 0].set_title('Driver Parameters')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Quality parameters (first 2-3)
    for i, param in enumerate(quality_params[:min(3, len(quality_params))]):
        axes[1, 1].plot(history['epochs'], history[param], colors_plot[i % len(colors_plot)], label=param.split('_')[1], linewidth=2)
    axes[1, 1].set_title('Quality Parameters')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Quality parameters (remaining)
    for i, param in enumerate(quality_params[3:]):
        axes[1, 2].plot(history['epochs'], history[param], colors_plot[(i+3) % len(colors_plot)], label=param.split('_')[1], linewidth=2)
    axes[1, 2].set_title('Quality Parameters (cont.)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Parameter changes (all parameters)
    all_params = gate_params + driver_params + quality_params
    changes = [history[p][-1] - history[p][0] for p in all_params]
    colors_bar = ['b' if c > 0 else 'r' for c in changes]
    param_labels = [p.split('_')[1] for p in all_params]
    
    axes[2, 0].barh(param_labels, changes, color=colors_bar)
    axes[2, 0].set_title('Parameter Changes (Final - Initial)')
    axes[2, 0].set_xlabel('Change')
    axes[2, 0].grid(True, alpha=0.3, axis='x')
    
    # Correlation: first gate vs driver params
    if gate_params and driver_params:
        first_gate = gate_params[0]
        for i, param in enumerate(driver_params[:3]):
            axes[2, 1].scatter(history[first_gate], history[param], alpha=0.6, s=20, label=param.split('_')[1])
        axes[2, 1].set_xlabel(first_gate.split('_')[1])
        axes[2, 1].set_ylabel('Driver weights')
        axes[2, 1].set_title(f'{first_gate.split("_")[1]} vs Driver')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
    
    # Final values
    gate_params = sorted([k for k in history.keys() if k.startswith('gate_')])
    driver_params = sorted([k for k in history.keys() if k.startswith('driver_')])
    quality_params = sorted([k for k in history.keys() if k.startswith('quality_')])
    display_params = (gate_params + driver_params + quality_params)[:8]
    final_vals = [f"{p.split('_')[1]}: {history[p][-1]:.3f}" for p in display_params]
    
    axes[2, 2].axis('off')
    axes[2, 2].text(0.1, 0.9, 'Final Values:', fontsize=12, fontweight='bold', transform=axes[2, 2].transAxes)
    for i, txt in enumerate(final_vals): axes[2, 2].text(0.1, 0.8 - i*0.10, txt, fontsize=9, transform=axes[2, 2].transAxes)
    axes[2, 2].text(0.1, 0.15, f"Loss: {history['loss'][0]:.4f} → {history['loss'][-1]:.4f}", fontsize=10, transform=axes[2, 2].transAxes)
    
    plt.tight_layout()
    plt.savefig('parameter_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary - automatically detect all parameters
    print(f"\n=== Parameter Analysis ===")
    print(f"Loss: {history['loss'][0]:.6f} → {history['loss'][-1]:.6f} (Δ: {history['loss'][-1] - history['loss'][0]:.6f})")
    print(f"\n=== All Final Parameters ===")
    
    for component in ['gate', 'driver', 'quality']:
        comp_params = sorted([k for k in history.keys() if k.startswith(f'{component}_')])
        if comp_params:
            param_str = ', '.join([f"{k.replace(component+'_', '')}={history[k][-1]:+.4f}" for k in comp_params])
            print(f"{component.capitalize():8s}: {param_str}")

if __name__ == "__main__":
    # np.random.seed(42)
    # n_days = 1000
    # returns = np.random.normal(0, 0.02, n_days)
    # close = 100 * np.exp(np.cumsum(returns))
    # high = close * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    # low = close * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    # df = pd.DataFrame({'close': close, 'high': high, 'low': low})
    
    df = pd.read_csv('spx.csv')
    
    config = PPTConfigTorch()
    features, mu, p, mask = prepare_data(df, config)
    model = PPTModelTorch(config)
    model, history = train_ppt_torch(model, features, mu, p, mask, config)
    
    plot_parameter_analysis(history)
    
    # Test
    test_features = {k: torch.FloatTensor(v[mask][:100]) for k, v in features.items()}
    with torch.no_grad():
        outputs = model(test_features)
        for h in config.H_SET:
            mu_hat, p_hat = outputs[h]
            print(f"H={h}: mu_hat[{mu_hat.min():.4f}, {mu_hat.max():.4f}], p_hat[{p_hat.min():.4f}, {p_hat.max():.4f}]")
    print("PyTorch PPT completed.")