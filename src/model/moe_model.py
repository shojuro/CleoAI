"""
Mixture-of-Experts (MoE) model implementation for the AI Autonomous Agent.

This module defines the core MoE architecture with expert routing, providing
efficient scaling through conditional computation where only a subset of
experts are activated for each token.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
from typing import Dict, List, Optional, Tuple, Union, Any, Literal
import math
import logging
from dataclasses import dataclass

# Configure logging
logger: logging.Logger = logging.getLogger(__name__)

@dataclass
class MoEConfig:
    """
    Configuration class for Mixture of Experts model.
    
    Attributes:
        num_experts: Total number of expert networks
        num_experts_per_token: Number of experts to activate per token
        expert_dropout: Dropout rate for expert outputs
        gate_type: Gating mechanism type ('top_k', 'sampling', or 'switch')
        routing_strategy: How to route tokens to experts ('tokens' or 'sequences')
        jitter_noise: Noise level for load balancing during training
        load_balancing_loss_weight: Weight for auxiliary load balancing loss
    """
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_dropout: float = 0.1
    gate_type: Literal["top_k", "sampling", "switch"] = "top_k"
    routing_strategy: Literal["tokens", "sequences"] = "tokens"
    jitter_noise: float = 0.1
    load_balancing_loss_weight: float = 0.01

class ExpertGatingNetwork(nn.Module):
    """
    Expert gating network for routing tokens to experts.
    
    This network determines which experts should process each token
    based on the token's representation. It uses a learned routing
    function to produce expert weights.
    
    Args:
        hidden_size: Dimension of input hidden states
        num_experts: Total number of experts available
        jitter_noise: Noise level for training stability
    """
    
    def __init__(self, hidden_size: int, num_experts: int, jitter_noise: float = 0.1) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_experts: int = num_experts
        self.jitter_noise: float = jitter_noise
        
        # Router projection
        self.router: nn.Linear = nn.Linear(hidden_size, num_experts)
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        num_experts_per_token: int = 2, 
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating network.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            num_experts_per_token: Number of experts to select per token
            training: Whether the model is in training mode
            
        Returns:
            Tuple containing:
                - router_logits: Router scores [batch_size, seq_len, num_experts]
                - expert_weights: Normalized weights for selected experts
                - expert_indices: Indices of selected experts
        
        Raises:
            ValueError: If num_experts_per_token > num_experts
        """
        if num_experts_per_token > self.num_experts:
            raise ValueError(
                f"num_experts_per_token ({num_experts_per_token}) cannot be "
                f"greater than num_experts ({self.num_experts})"
            )
        batch_size, seq_len, _ = hidden_states.shape
        
        # Apply router to get logits
        router_logits = self.router(hidden_states)  # [batch_size, seq_len, num_experts]
        
        if training and self.jitter_noise > 0:
            # Add jitter noise for exploration and better load balancing during training
            router_logits += torch.randn_like(router_logits) * self.jitter_noise
        
        # Get top-k experts per token
        expert_weights, expert_indices = torch.topk(
            router_logits, num_experts_per_token, dim=-1
        )  # Shapes: [batch_size, seq_len, num_experts_per_token]
        
        # Apply softmax to get normalized weights for selected experts
        expert_weights = F.softmax(expert_weights, dim=-1)
        
        return router_logits, expert_weights, expert_indices

class ExpertFFN(nn.Module):
    """
    Expert Feed-Forward Network.
    
    Each expert is a standard transformer FFN with two linear layers
    and a GELU activation. This represents a single expert in the MoE layer.
    
    Args:
        hidden_size: Input and output dimension
        intermediate_size: Hidden dimension of the FFN
        dropout_prob: Dropout probability
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        dropout_prob: float = 0.1
    ) -> None:
        super().__init__()
        self.dense1: nn.Linear = nn.Linear(hidden_size, intermediate_size)
        self.act_fn: nn.GELU = nn.GELU()
        self.dense2: nn.Linear = nn.Linear(intermediate_size, hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single expert.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        x = self.dense1(x)
        x = self.act_fn(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x

class MoELayer(nn.Module):
    """
    Mixture of Experts layer with gating network and multiple FFN experts.
    """
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        num_experts: int = 8, 
        num_experts_per_token: int = 2,
        expert_dropout: float = 0.1,
        dropout_prob: float = 0.1,
        jitter_noise: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.expert_dropout = expert_dropout
        
        # Create gating network
        self.gate = ExpertGatingNetwork(hidden_size, num_experts, jitter_noise)
        
        # Create experts
        self.experts = nn.ModuleList([
            ExpertFFN(hidden_size, intermediate_size, dropout_prob)
            for _ in range(num_experts)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass for the MoE layer.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            training: Whether the model is in training mode
            
        Returns:
            output: Output tensor after applying MoE
            aux_loss: Auxiliary loss for load balancing
        """
        # Apply layer normalization
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        
        # Get router logits and expert selection
        batch_size, seq_len, _ = hidden_states.shape
        router_logits, expert_weights, expert_indices = self.gate(
            hidden_states, 
            num_experts_per_token=self.num_experts_per_token, 
            training=training
        )
        
        # Initialize output tensor
        final_output = torch.zeros_like(hidden_states)
        
        # Compute load balancing loss
        # Count how many tokens are routed to each expert
        router_probs = F.softmax(router_logits, dim=-1)
        # Mean probability of routing to each expert
        token_usage = router_probs.mean(dim=[0, 1])
        # Ideal usage would be uniform across experts
        balance_loss = self.num_experts * torch.sum(token_usage * token_usage) - 1.0
        
        # Process tokens through their respective experts
        for expert_idx in range(self.num_experts):
            # Find which positions use this expert
            expert_mask = (expert_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Extract relevant hidden states for this expert
            # Reshape to 2D for processing
            expert_input = hidden_states[expert_mask].reshape(-1, self.hidden_size)
            
            # Get output from this expert
            expert_output = self.experts[expert_idx](expert_input)
            
            # Get the corresponding weight for this expert at each position
            # For each position where this expert is used, find its weight
            weight_idx = (expert_indices == expert_idx).int().argmax(dim=-1)
            expert_weight = torch.zeros_like(expert_mask, dtype=torch.float32)
            for b in range(batch_size):
                for s in range(seq_len):
                    if expert_mask[b, s]:
                        idx = weight_idx[b, s].item()
                        expert_weight[b, s] = expert_weights[b, s, idx]
            
            # Apply expert dropout during training
            if training and self.expert_dropout > 0:
                dropout_mask = torch.rand_like(expert_weight) > self.expert_dropout
                expert_weight = expert_weight * dropout_mask.float()
            
            # Combine expert output with weights
            output_flat = torch.zeros(batch_size * seq_len, self.hidden_size, device=hidden_states.device)
            output_flat[expert_mask.reshape(-1)] = expert_output * expert_weight.reshape(-1, 1)[expert_mask.reshape(-1)]
            expert_output = output_flat.reshape(batch_size, seq_len, self.hidden_size)
            
            # Add to final output
            final_output += expert_output
        
        # Add residual connection
        output = residual + final_output
        
        # Return output and auxiliary loss
        aux_loss = {
            "load_balance_loss": balance_loss
        }
        
        return output, aux_loss

class MoETransformerBlock(nn.Module):
    """
    Transformer block with MoE FFN layer.
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        expert_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        jitter_noise: float = 0.1
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, 
            num_attention_heads, 
            dropout=attention_dropout,
            batch_first=True
        )
        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        
        # MoE layer
        self.moe = MoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            expert_dropout=expert_dropout,
            dropout_prob=hidden_dropout,
            jitter_noise=jitter_noise
        )
        
        self.dropout = nn.Dropout(hidden_dropout)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass for the transformer block.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask tensor
            
        Returns:
            hidden_states: Output tensor
            aux_loss: Auxiliary loss
        """
        residual = hidden_states
        
        # Self-attention
        hidden_states = self.attention_layer_norm(hidden_states)
        attention_output, _ = self.attention(
            hidden_states, 
            hidden_states, 
            hidden_states,
            key_padding_mask=attention_mask
        )
        hidden_states = residual + self.dropout(attention_output)
        
        # MoE FFN
        hidden_states, aux_loss = self.moe(hidden_states)
        
        return hidden_states, aux_loss

class MoEModel(nn.Module):
    """
    Full Mixture-of-Experts model for the AI Autonomous Agent.
    """
    def __init__(self, config):
        super().__init__()
        # Load base model configuration
        self.config = config
        
        # Create initial embedding layer
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Create transformer blocks with MoE
        self.layers = nn.ModuleList([
            MoETransformerBlock(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_token,
                expert_dropout=config.expert_dropout,
                attention_dropout=config.attention_dropout_prob,
                hidden_dropout=config.hidden_dropout_prob,
                jitter_noise=config.jitter_noise
            )
            for _ in range(config.num_hidden_layers)
        ])
        
        # Output layer normalization
        self.ln_f = nn.LayerNorm(config.hidden_size)
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights between embedding and LM head
        self.lm_head.weight = self.embeddings.weight
        
    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None):
        """
        Forward pass for the MoE model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            labels: Labels for LM loss calculation
            
        Returns:
            loss: Total loss if labels are provided
            logits: Output logits
            aux_loss: Auxiliary losses dict
        """
        batch_size, seq_length = input_ids.shape
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
        # Get embeddings
        inputs_embeds = self.embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        # Process through transformer layers
        total_aux_loss = 0.0
        for layer in self.layers:
            hidden_states, aux_loss = layer(hidden_states, attention_mask)
            total_aux_loss += aux_loss["load_balance_loss"]
            
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            # Combine with auxiliary loss
            loss = lm_loss + config.load_balancing_loss_weight * total_aux_loss
            
        return {
            "loss": loss,
            "logits": logits,
            "aux_loss": total_aux_loss
        }

def load_pretrained_model_with_moe(
    model_name: str,
    num_experts: int = 8,
    num_experts_per_token: int = 2,
    expert_dropout: float = 0.1,
    load_balancing_loss_weight: float = 0.01,
    jitter_noise: float = 0.1
):
    """
    Load a pretrained model and convert its FFN layers to MoE layers.
    
    Args:
        model_name: Name or path of the pretrained model
        num_experts: Number of experts
        num_experts_per_token: Number of experts to use per token
        expert_dropout: Dropout rate for expert selection
        load_balancing_loss_weight: Weight for load balancing loss
        jitter_noise: Noise for better load balancing
        
    Returns:
        model: MoE model
    """
    # Load pretrained model configuration
    config = AutoConfig.from_pretrained(model_name)
    
    # Add MoE-specific configuration
    config.num_experts = num_experts
    config.num_experts_per_token = num_experts_per_token
    config.expert_dropout = expert_dropout
    config.load_balancing_loss_weight = load_balancing_loss_weight
    config.jitter_noise = jitter_noise
    
    # Create MoE model with this configuration
    model = MoEModel(config)
    
    # Load pretrained weights
    pretrained_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Copy weights from pretrained model to MoE model
    # This is a simplified approach - in practice, you would need to carefully map weights
    # from the original architecture to the MoE architecture
    model.embeddings.weight.data.copy_(pretrained_model.get_input_embeddings().weight.data)
    if hasattr(pretrained_model, "position_embeddings"):
        model.position_embeddings.weight.data.copy_(pretrained_model.position_embeddings.weight.data)
    
    # Copy layer norm weights
    model.ln_f.weight.data.copy_(pretrained_model.ln_f.weight.data)
    model.ln_f.bias.data.copy_(pretrained_model.ln_f.bias.data)
    
    # Initialize experts using weights from the original FFN
    for i, layer in enumerate(model.layers):
        pretrained_layer = pretrained_model.h[i]
        
        # Copy attention weights
        layer.attention.in_proj_weight.data.copy_(
            torch.cat([
                pretrained_layer.attn.q_proj.weight,
                pretrained_layer.attn.k_proj.weight,
                pretrained_layer.attn.v_proj.weight
            ], dim=0)
        )
        if hasattr(layer.attention, 'in_proj_bias'):
            layer.attention.in_proj_bias.data.copy_(
                torch.cat([
                    pretrained_layer.attn.q_proj.bias,
                    pretrained_layer.attn.k_proj.bias,
                    pretrained_layer.attn.v_proj.bias
                ], dim=0)
            )
        
        # Copy layer norm weights
        layer.attention_layer_norm.weight.data.copy_(pretrained_layer.ln_1.weight.data)
        layer.attention_layer_norm.bias.data.copy_(pretrained_layer.ln_1.bias.data)
        layer.moe.layer_norm.weight.data.copy_(pretrained_layer.ln_2.weight.data)
        layer.moe.layer_norm.bias.data.copy_(pretrained_layer.ln_2.bias.data)
        
        # Initialize each expert with the same weights from the pretrained FFN
        for expert in layer.moe.experts:
            expert.dense1.weight.data.copy_(pretrained_layer.mlp.c_fc.weight.data)
            expert.dense1.bias.data.copy_(pretrained_layer.mlp.c_fc.bias.data)
            expert.dense2.weight.data.copy_(pretrained_layer.mlp.c_proj.weight.data)
            expert.dense2.bias.data.copy_(pretrained_layer.mlp.c_proj.bias.data)
    
    return model
