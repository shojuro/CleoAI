"""Unit tests for MoE model module."""
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import pytest

from src.model.moe_model import (
    MoEConfig,
    ExpertGatingNetwork,
    ExpertFFN,
    MoELayer,
)


class TestMoEConfig:
    """Test cases for MoEConfig dataclass."""
    
    def test_default_config_values(self):
        """Test default values for MoEConfig."""
        config = MoEConfig()
        
        assert config.num_experts == 8
        assert config.num_experts_per_token == 2
        assert config.expert_dropout == 0.1
        assert config.gate_type == "top_k"
        assert config.routing_strategy == "tokens"
        assert config.jitter_noise == 0.1
        assert config.load_balancing_loss_weight == 0.01
    
    def test_custom_config_values(self):
        """Test custom values for MoEConfig."""
        config = MoEConfig(
            num_experts=16,
            num_experts_per_token=4,
            expert_dropout=0.2,
            gate_type="sampling",
            routing_strategy="sequences",
            jitter_noise=0.05,
            load_balancing_loss_weight=0.02
        )
        
        assert config.num_experts == 16
        assert config.num_experts_per_token == 4
        assert config.expert_dropout == 0.2
        assert config.gate_type == "sampling"
        assert config.routing_strategy == "sequences"
        assert config.jitter_noise == 0.05
        assert config.load_balancing_loss_weight == 0.02
    
    def test_config_validation(self):
        """Test that config values are valid."""
        config = MoEConfig()
        
        assert config.num_experts > 0
        assert config.num_experts_per_token > 0
        assert config.num_experts_per_token <= config.num_experts
        assert 0 <= config.expert_dropout <= 1
        assert config.gate_type in ["top_k", "sampling", "switch"]
        assert config.routing_strategy in ["tokens", "sequences"]
        assert config.jitter_noise >= 0
        assert config.load_balancing_loss_weight >= 0


class TestExpertGatingNetwork:
    """Test cases for ExpertGatingNetwork."""
    
    def test_initialization(self):
        """Test ExpertGatingNetwork initialization."""
        hidden_size = 768
        num_experts = 8
        
        gating = ExpertGatingNetwork(hidden_size, num_experts)
        
        assert gating.hidden_size == hidden_size
        assert gating.num_experts == num_experts
        assert gating.jitter_noise == 0.1
        assert isinstance(gating.router, nn.Linear)
        assert gating.router.in_features == hidden_size
        assert gating.router.out_features == num_experts
    
    def test_forward_pass_training(self):
        """Test forward pass in training mode."""
        batch_size = 2
        seq_len = 10
        hidden_size = 768
        num_experts = 8
        num_experts_per_token = 2
        
        gating = ExpertGatingNetwork(hidden_size, num_experts)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        router_logits, expert_weights, expert_indices = gating(
            hidden_states, 
            num_experts_per_token=num_experts_per_token,
            training=True
        )
        
        # Check shapes
        assert router_logits.shape == (batch_size, seq_len, num_experts)
        assert expert_weights.shape == (batch_size, seq_len, num_experts_per_token)
        assert expert_indices.shape == (batch_size, seq_len, num_experts_per_token)
        
        # Check that weights sum to 1
        weight_sums = expert_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)
        
        # Check that indices are valid
        assert (expert_indices >= 0).all()
        assert (expert_indices < num_experts).all()
    
    def test_forward_pass_inference(self):
        """Test forward pass in inference mode."""
        batch_size = 2
        seq_len = 10
        hidden_size = 768
        num_experts = 8
        
        gating = ExpertGatingNetwork(hidden_size, num_experts, jitter_noise=0.5)
        gating.eval()
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Run multiple times to check determinism in eval mode
        with torch.no_grad():
            results1 = gating(hidden_states, training=False)
            results2 = gating(hidden_states, training=False)
        
        # Results should be identical in eval mode
        assert torch.allclose(results1[0], results2[0])
        assert torch.allclose(results1[1], results2[1])
        assert torch.equal(results1[2], results2[2])
    
    def test_different_num_experts_per_token(self):
        """Test with different numbers of experts per token."""
        batch_size = 1
        seq_len = 5
        hidden_size = 768
        num_experts = 8
        
        gating = ExpertGatingNetwork(hidden_size, num_experts)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        for k in [1, 2, 4, 8]:
            _, expert_weights, expert_indices = gating(
                hidden_states, 
                num_experts_per_token=k
            )
            
            assert expert_weights.shape[-1] == k
            assert expert_indices.shape[-1] == k


class TestExpertFFN:
    """Test cases for ExpertFFN."""
    
    def test_initialization(self):
        """Test ExpertFFN initialization."""
        hidden_size = 768
        intermediate_size = 3072
        dropout_prob = 0.1
        
        expert = ExpertFFN(hidden_size, intermediate_size, dropout_prob)
        
        assert isinstance(expert.dense1, nn.Linear)
        assert expert.dense1.in_features == hidden_size
        assert expert.dense1.out_features == intermediate_size
        
        assert isinstance(expert.dense2, nn.Linear)
        assert expert.dense2.in_features == intermediate_size
        assert expert.dense2.out_features == hidden_size
        
        assert isinstance(expert.act_fn, nn.GELU)
        assert isinstance(expert.dropout, nn.Dropout)
        assert expert.dropout.p == dropout_prob
    
    def test_forward_pass(self):
        """Test ExpertFFN forward pass."""
        batch_size = 2
        seq_len = 10
        hidden_size = 768
        intermediate_size = 3072
        
        expert = ExpertFFN(hidden_size, intermediate_size)
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        output = expert(x)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, hidden_size)
        
        # Check that output is different from input (transformation occurred)
        assert not torch.allclose(output, x)
    
    def test_gradient_flow(self):
        """Test that gradients flow through ExpertFFN."""
        hidden_size = 128
        intermediate_size = 512
        
        expert = ExpertFFN(hidden_size, intermediate_size)
        x = torch.randn(1, 5, hidden_size, requires_grad=True)
        
        output = expert(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert expert.dense1.weight.grad is not None
        assert expert.dense2.weight.grad is not None


@pytest.fixture
def moe_layer_config():
    """Fixture for MoE layer configuration."""
    return {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_experts": 8,
        "num_experts_per_token": 2,
        "expert_dropout": 0.1,
        "jitter_noise": 0.1,
    }


class TestMoELayer:
    """Test cases for MoELayer."""
    
    def test_initialization(self, moe_layer_config):
        """Test MoELayer initialization."""
        layer = MoELayer(**moe_layer_config)
        
        assert layer.num_experts == moe_layer_config["num_experts"]
        assert layer.num_experts_per_token == moe_layer_config["num_experts_per_token"]
        assert isinstance(layer.gating_network, ExpertGatingNetwork)
        assert hasattr(layer, "experts")
        assert len(layer.experts) == moe_layer_config["num_experts"]
        
        # Check that all experts are ExpertFFN instances
        for expert in layer.experts:
            assert isinstance(expert, ExpertFFN)
    
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 10),
        (2, 20),
        (4, 5),
    ])
    def test_forward_pass_shapes(self, moe_layer_config, batch_size, seq_len):
        """Test MoELayer forward pass with different input shapes."""
        layer = MoELayer(**moe_layer_config)
        hidden_states = torch.randn(
            batch_size, seq_len, moe_layer_config["hidden_size"]
        )
        
        output = layer(hidden_states)
        
        # Check output shape matches input shape
        assert output.shape == hidden_states.shape
    
    def test_load_balancing_loss(self, moe_layer_config):
        """Test that load balancing loss is computed."""
        layer = MoELayer(**moe_layer_config)
        hidden_states = torch.randn(2, 10, moe_layer_config["hidden_size"])
        
        output = layer(hidden_states)
        
        # Check that auxiliary loss is stored
        assert hasattr(layer, "load_balancing_loss")
        assert layer.load_balancing_loss is not None
        assert layer.load_balancing_loss.item() >= 0
    
    def test_expert_dropout(self, moe_layer_config):
        """Test expert dropout functionality."""
        # Set high dropout for testing
        moe_layer_config["expert_dropout"] = 0.5
        layer = MoELayer(**moe_layer_config)
        
        hidden_states = torch.randn(1, 10, moe_layer_config["hidden_size"])
        
        # Run in training mode
        layer.train()
        outputs_train = []
        for _ in range(10):
            output = layer(hidden_states)
            outputs_train.append(output)
        
        # Outputs should vary due to dropout
        all_same = all(
            torch.allclose(outputs_train[0], out) 
            for out in outputs_train[1:]
        )
        assert not all_same
        
        # Run in eval mode
        layer.eval()
        with torch.no_grad():
            output1 = layer(hidden_states)
            output2 = layer(hidden_states)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2)
    
    def test_gradient_flow_through_layer(self, moe_layer_config):
        """Test that gradients flow through the entire MoE layer."""
        layer = MoELayer(**moe_layer_config)
        hidden_states = torch.randn(
            1, 5, moe_layer_config["hidden_size"], 
            requires_grad=True
        )
        
        output = layer(hidden_states)
        loss = output.sum() + layer.load_balancing_loss
        loss.backward()
        
        # Check gradients exist
        assert hidden_states.grad is not None
        assert layer.gating_network.router.weight.grad is not None
        
        # Check that at least some experts received gradients
        experts_with_grad = sum(
            1 for expert in layer.experts 
            if expert.dense1.weight.grad is not None
        )
        assert experts_with_grad > 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_device_compatibility(device, moe_layer_config):
    """Test that MoE components work on different devices."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    layer = MoELayer(**moe_layer_config).to(device)
    hidden_states = torch.randn(
        2, 10, moe_layer_config["hidden_size"]
    ).to(device)
    
    output = layer(hidden_states)
    
    assert output.device.type == device
    assert output.shape == hidden_states.shape


def test_moe_memory_efficiency():
    """Test that MoE uses memory efficiently by not computing all experts."""
    # This is a conceptual test - in practice would need memory profiling
    config = {
        "hidden_size": 128,
        "intermediate_size": 512,
        "num_experts": 32,  # Many experts
        "num_experts_per_token": 2,  # But only use 2
    }
    
    layer = MoELayer(**config)
    hidden_states = torch.randn(1, 100, config["hidden_size"])
    
    # Track which experts were used
    used_experts = set()
    
    def hook_fn(module, input, output, expert_idx):
        used_experts.add(expert_idx)
    
    # Add hooks to track expert usage
    for i, expert in enumerate(layer.experts):
        expert.register_forward_hook(
            lambda m, inp, out, idx=i: hook_fn(m, inp, out, idx)
        )
    
    output = layer(hidden_states)
    
    # Not all experts should be used
    assert len(used_experts) < config["num_experts"]
    assert len(used_experts) > 0