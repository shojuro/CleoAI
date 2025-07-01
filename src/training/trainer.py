"""
Training pipeline for the AI Autonomous Agent.
Handles multi-phase training with checkpointing and evaluation.
"""
import os
import json
import time
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm.auto import tqdm

from transformers import (
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
import evaluate
from datasets import load_dataset, Dataset as HFDataset
import wandb
import deepspeed
from accelerate import Accelerator

from src.model.moe_model import MoEModel, load_pretrained_model_with_moe
from src.memory.memory_manager import MemoryManager, ProceduralMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingPhase:
    """Class representing a training phase with specific datasets and objectives"""
    def __init__(
        self, 
        phase_name: str,
        datasets: List[str],
        steps: int,
        description: str,
        learning_rate: float = 2e-5,
        eval_datasets: Optional[List[str]] = None,
        custom_metrics: Optional[List[str]] = None
    ):
        self.phase_name = phase_name
        self.datasets = datasets
        self.steps = steps
        self.description = description
        self.learning_rate = learning_rate
        self.eval_datasets = eval_datasets or []
        self.custom_metrics = custom_metrics or []
        
    def __repr__(self):
        return f"TrainingPhase(name={self.phase_name}, steps={self.steps})"

class CustomDataset(Dataset):
    """Custom dataset for training with conversation/instruction data"""
    def __init__(
        self, 
        data: List[Dict], 
        tokenizer, 
        max_length: int = 2048,
        input_key: str = "input",
        output_key: str = "output"
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_key = input_key
        self.output_key = output_key
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format as instruction-response pair
        input_text = item.get(self.input_key, "")
        output_text = item.get(self.output_key, "")
        
        full_text = f"{input_text}\n{output_text}"
        
        # Tokenize
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Convert to proper format for training
        input_ids = encodings.input_ids[0]
        attention_mask = encodings.attention_mask[0]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # For language modeling
        }

class ModelTrainer:
    """
    Trainer for the AI Autonomous Agent model.
    Handles training, evaluation, and checkpointing.
    """
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        config: Any,
        use_moe: bool = True,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        device: str = "cuda",
        fp16: bool = False,
        bf16: bool = True,
        use_deepspeed: bool = True,
        deepspeed_config_path: str = "configs/deepspeed_config.json",
        log_to_wandb: bool = True,
        wandb_project: str = "ai-dating-assistant",
        wandb_entity: Optional[str] = None
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.config = config
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.device = device
        self.fp16 = fp16
        self.bf16 = bf16
        self.use_deepspeed = use_deepspeed
        self.deepspeed_config_path = deepspeed_config_path
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and tokenizer
        self._init_model_and_tokenizer()
        
        # Initialize memory manager
        self.memory_manager = MemoryManager()
        
        # Initialize wandb
        self.log_to_wandb = log_to_wandb
        if log_to_wandb:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config={
                    "model_name": model_name,
                    "use_moe": use_moe,
                    "num_experts": num_experts,
                    "num_experts_per_token": num_experts_per_token,
                    "device": device,
                    "fp16": fp16,
                    "bf16": bf16
                }
            )
        
    def _init_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        from transformers import AutoTokenizer
        
        logger.info(f"Loading model {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with MoE architecture if specified
        if self.use_moe:
            logger.info(f"Creating MoE model with {self.num_experts} experts")
            self.model = load_pretrained_model_with_moe(
                model_name=self.model_name,
                num_experts=self.num_experts,
                num_experts_per_token=self.num_experts_per_token,
                expert_dropout=0.1,
                load_balancing_loss_weight=0.01,
                jitter_noise=0.1
            )
        else:
            from transformers import AutoModelForCausalLM
            logger.info("Loading standard model")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
        # Set pad token ID for model
        if hasattr(self.model.config, "pad_token_id") and self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
        logger.info(f"Model and tokenizer loaded successfully")
    
    def _prepare_training_arguments(
        self, 
        phase: TrainingPhase, 
        eval_steps: int, 
        save_steps: int,
        gradient_accumulation_steps: int = 8
    ) -> TrainingArguments:
        """Prepare training arguments for the Trainer"""
        phase_output_dir = self.output_dir / phase.phase_name
        
        training_args = TrainingArguments(
            output_dir=str(phase_output_dir),
            overwrite_output_dir=True,
            
            # Training parameters
            num_train_epochs=1,  # We'll control training by steps, not epochs
            max_steps=phase.steps,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            
            # Optimization
            learning_rate=phase.learning_rate,
            weight_decay=self.config.weight_decay,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=self.config.max_grad_norm,
            
            # Scheduler
            lr_scheduler_type=self.config.scheduler,
            warmup_steps=self.config.warmup_steps,
            
            # Evaluation and checkpointing
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            
            # Logging
            logging_dir=str(self.output_dir / "logs"),
            logging_strategy="steps",
            logging_steps=self.config.logging_steps,
            report_to=["wandb"] if self.log_to_wandb else [],
            
            # Hardware
            fp16=self.fp16,
            bf16=self.bf16,
            dataloader_num_workers=4,
            
            # DeepSpeed
            deepspeed=self.deepspeed_config_path if self.use_deepspeed else None,
        )
        
        return training_args
    
    def _load_datasets(self, dataset_names: List[str]) -> HFDataset:
        """Load and prepare datasets for training"""
        datasets = []
        
        for dataset_name in dataset_names:
            # This is a placeholder for actual dataset loading
            # You would replace this with your actual dataset loading code
            
            # Mock example for placeholder
            try:
                # First try to load from Hugging Face
                dataset = load_dataset(dataset_name)
                datasets.append(dataset["train"])
                logger.info(f"Loaded dataset {dataset_name} from Hugging Face")
            except Exception as e:
                # If not found, check local filesystem
                local_path = Path("data/processed") / f"{dataset_name}.json"
                
                if local_path.exists():
                    with open(local_path, "r") as f:
                        data = json.load(f)
                    
                    dataset = HFDataset.from_dict({
                        "input": [item.get("input", "") for item in data],
                        "output": [item.get("output", "") for item in data]
                    })
                    datasets.append(dataset)
                    logger.info(f"Loaded dataset {dataset_name} from local file")
                else:
                    logger.warning(f"Dataset {dataset_name} not found, skipping")
        
        # Concatenate all datasets
        if not datasets:
            raise ValueError("No datasets could be loaded")
            
        if len(datasets) == 1:
            return datasets[0]
            
        # Concatenate datasets
        combined_dataset = datasets[0]
        for dataset in datasets[1:]:
            combined_dataset = HFDataset.concatenate_datasets([combined_dataset, dataset])
            
        return combined_dataset
    
    def _prepare_compute_metrics(self, custom_metrics: List[str]) -> Callable:
        """Prepare compute_metrics function for evaluation"""
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            
            # Basic metric: perplexity
            loss = torch.nn.functional.cross_entropy(
                torch.tensor(logits.reshape(-1, logits.shape[-1])), 
                torch.tensor(labels.reshape(-1)),
                ignore_index=-100
            ).item()
            
            perplexity = float(np.exp(loss))
            
            results = {
                "perplexity": perplexity,
                "loss": loss
            }
            
            # Add custom metrics based on specified metrics
            for metric_name in custom_metrics:
                if metric_name == "accuracy":
                    # Calculate accuracy
                    predictions = logits.argmax(axis=-1)
                    valid_mask = (labels != -100).flatten()
                    accuracy = np.mean(
                        (predictions.reshape(-1)[valid_mask] == labels.reshape(-1)[valid_mask]).astype(np.float32)
                    )
                    results["accuracy"] = float(accuracy)
                    
                # Add more custom metrics as needed
                
            return results
            
        return compute_metrics
    
    def train_phase(self, phase: TrainingPhase):
        """Train the model for a specific phase"""
        logger.info(f"Starting training phase: {phase.phase_name}")
        logger.info(f"Description: {phase.description}")
        
        # Load datasets
        train_dataset = self._load_datasets(phase.datasets)
        eval_dataset = self._load_datasets(phase.eval_datasets) if phase.eval_datasets else None
        
        # Prepare training arguments
        training_args = self._prepare_training_arguments(
            phase=phase,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )
        
        # Prepare compute_metrics function
        compute_metrics = self._prepare_compute_metrics(phase.custom_metrics)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # We're doing causal language modeling
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Save phase configuration
        phase_config = {
            "phase_name": phase.phase_name,
            "datasets": phase.datasets,
            "steps": phase.steps,
            "description": phase.description,
            "learning_rate": phase.learning_rate,
            "eval_datasets": phase.eval_datasets,
            "custom_metrics": phase.custom_metrics
        }
        phase_config_path = self.output_dir / phase.phase_name / "phase_config.json"
        with open(phase_config_path, "w") as f:
            json.dump(phase_config, f, indent=2)
        
        # Train the model
        logger.info(f"Training model for {phase.steps} steps")
        train_result = trainer.train()
        
        # Save model and tokenizer
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        # Log training results
        metrics = train_result.metrics
        logger.info(f"Training metrics: {metrics}")
        
        # Save training metrics
        metrics_path = self.output_dir / phase.phase_name / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
        # Evaluate
        if eval_dataset:
            logger.info("Evaluating model")
            eval_results = trainer.evaluate()
            logger.info(f"Evaluation results: {eval_results}")
            
            # Save evaluation results
            eval_metrics_path = self.output_dir / phase.phase_name / "eval_metrics.json"
            with open(eval_metrics_path, "w") as f:
                json.dump(eval_results, f, indent=2)
                
        return train_result
    
    def train_all_phases(self, phases: List[TrainingPhase]):
        """Train the model through all specified phases"""
        for i, phase in enumerate(phases):
            logger.info(f"=== Starting Phase {i+1}/{len(phases)}: {phase.phase_name} ===")
            
            # Train for this phase
            train_result = self.train_phase(phase)
            
            # Checkpoint all memory systems after each phase
            self.memory_manager.save_all(
                base_directory=str(self.output_dir / phase.phase_name / "memory_checkpoint")
            )
            
            logger.info(f"=== Completed Phase {i+1}/{len(phases)}: {phase.phase_name} ===")
            
        logger.info("All training phases completed")
        
    def get_or_create_protocol(self, protocol_id: str, name: str, steps: List[Dict], 
                              category: str = "training", description: str = "") -> ProceduralMemory:
        """
        Gets an existing protocol or creates a new one if it doesn't exist.
        
        Args:
            protocol_id: Unique identifier for the protocol
            name: Human-readable name for the protocol
            steps: List of steps in the protocol, each a dictionary
            category: Category of the protocol
            description: Detailed description of the protocol
            
        Returns:
            ProceduralMemory: The retrieved or newly created protocol
        """
        try:
            # Try to retrieve the existing protocol
            protocol = self.memory_manager.get_procedural_protocol(protocol_id)
            if protocol:
                logger.info(f"Retrieved existing protocol: {protocol_id}")
                return protocol
        except Exception as e:
            logger.warning(f"Error retrieving protocol {protocol_id}: {e}")
        
        # Create a new protocol if it doesn't exist
        try:
            # Validate steps format
            if not isinstance(steps, list):
                raise ValueError("Steps must be a list of dictionaries")
            
            for step in steps:
                if not isinstance(step, dict) or "id" not in step or "name" not in step:
                    raise ValueError("Each step must be a dictionary with at least 'id' and 'name' keys")
            
            protocol = ProceduralMemory(
                protocol_id=protocol_id,
                name=name,
                steps=steps,
                category=category,
                description=description
            )
            
            # Save the protocol to the database
            self.memory_manager.create_procedural_protocol(
                name=name,
                steps=steps,
                category=category,
                description=description
            )
            
            logger.info(f"Created new protocol: {protocol_id}")
            return protocol
        except Exception as e:
            logger.error(f"Error creating protocol {protocol_id}: {e}")
            raise
    
    def create_training_phases(self) -> List[TrainingPhase]:
        """Create training phases based on configuration"""
        phases = [
            TrainingPhase(
                phase_name="foundation",
                datasets=self.config.train_datasets["phase1"],
                steps=self.config.phase1_steps,
                description="Foundation phase - general capabilities and knowledge",
                learning_rate=2e-5,
                eval_datasets=["general_evaluation"],
                custom_metrics=["perplexity"]
            ),
            TrainingPhase(
                phase_name="emotional",
                datasets=self.config.train_datasets["phase2"],
                steps=self.config.phase2_steps,
                description="Emotional & Safety phase - empathy and safety capabilities",
                learning_rate=1e-5,
                eval_datasets=["emotional_evaluation", "safety_evaluation"],
                custom_metrics=["perplexity", "emotion_detection_f1", "safety_compliance"]
            ),
            TrainingPhase(
                phase_name="relationship",
                datasets=self.config.train_datasets["phase3"],
                steps=self.config.phase3_steps,
                description="Relationship & Dating phase - dating-specific capabilities",
                learning_rate=8e-6,
                eval_datasets=["relationship_evaluation"],
                custom_metrics=["perplexity", "conversation_quality"]
            ),
            TrainingPhase(
                phase_name="integration",
                datasets=self.config.train_datasets["phase4"],
                steps=self.config.phase4_steps,
                description="Integration & Refinement phase - combining all skills",
                learning_rate=5e-6,
                eval_datasets=["integrated_evaluation"],
                custom_metrics=["perplexity", "emotion_detection_f1", "safety_compliance", "conversation_quality"]
            )
        ]
        
        return phases
    
    def run_full_training(self):
        """Run the full training pipeline with all phases"""
        logger.info(f"Starting full training pipeline with model {self.model_name}")
        
        start_time = time.time()
        
        # Create training phases
        phases = self.create_training_phases()
        
        # Train through all phases
        self.train_all_phases(phases)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"Training completed in {training_time / 3600:.2f} hours")
        
        # Save final model
        final_model_path = self.output_dir / "final_model"
        self.model.save_pretrained(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        # Save memory systems
        self.memory_manager.save_all(
            base_directory=str(self.output_dir / "final_memory_checkpoint")
        )
        
        # Log completion to wandb
        if self.log_to_wandb:
            wandb.log({
                "training_complete": True,
                "training_time_hours": training_time / 3600,
                "final_model_path": str(final_model_path)
            })
            
            # Finish wandb run
            wandb.finish()
        
        return {
            "training_time": training_time,
            "final_model_path": str(final_model_path)
        }
    
    def evaluate_model(self, test_dataset_names: List[str], metrics: List[str] = None):
        """Evaluate the model on test datasets"""
        logger.info(f"Evaluating model on {len(test_dataset_names)} test datasets")
        
        # Default metrics if none provided
        if metrics is None:
            metrics = ["perplexity", "accuracy"]
            
        # Load test datasets
        all_results = {}
        
        for dataset_name in test_dataset_names:
            logger.info(f"Evaluating on dataset: {dataset_name}")
            
            try:
                # Load dataset
                test_dataset = self._load_datasets([dataset_name])
                
                # Prepare arguments for evaluation
                eval_args = TrainingArguments(
                    output_dir=str(self.output_dir / "evaluation" / dataset_name),
                    per_device_eval_batch_size=self.config.eval_batch_size,
                    fp16=self.fp16,
                    bf16=self.bf16
                )
                
                # Create compute_metrics function
                compute_metrics = self._prepare_compute_metrics(metrics)
                
                # Create data collator
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False
                )
                
                # Create trainer for evaluation
                trainer = Trainer(
                    model=self.model,
                    args=eval_args,
                    eval_dataset=test_dataset,
                    tokenizer=self.tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics
                )
                
                # Evaluate
                results = trainer.evaluate()
                logger.info(f"Evaluation results for {dataset_name}: {results}")
                
                # Store results
                all_results[dataset_name] = results
                
                # Save results
                results_path = self.output_dir / "evaluation" / f"{dataset_name}_results.json"
                results_path.parent.mkdir(parents=True, exist_ok=True)
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Error evaluating on dataset {dataset_name}: {e}")
                all_results[dataset_name] = {"error": str(e)}
                
        # Save combined results
        combined_results_path = self.output_dir / "evaluation" / "all_results.json"
        with open(combined_results_path, "w") as f:
            json.dump(all_results, f, indent=2)
            
        return all_results
                
    def human_evaluation_setup(self, num_samples: int = 100):
        """Create samples for human evaluation"""
        logger.info(f"Creating {num_samples} samples for human evaluation")
        
        # This is a placeholder for creating human evaluation samples
        # You would replace this with your actual human evaluation setup
        
        # Create directory for human evaluation
        human_eval_dir = self.output_dir / "human_evaluation"
        human_eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple JSON file with prompts for human evaluation
        # This would be replaced with a more sophisticated approach in production
        
        sample_prompts = [
            {
                "id": f"sample_{i}",
                "prompt": f"This is a sample prompt for human evaluation #{i}",
                "model_response": "",  # To be filled by the model
                "human_rating": None,  # To be filled by human evaluators
                "human_feedback": ""   # To be filled by human evaluators
            }
            for i in range(num_samples)
        ]
        
        # Save prompts
        prompts_path = human_eval_dir / "evaluation_prompts.json"
        with open(prompts_path, "w") as f:
            json.dump(sample_prompts, f, indent=2)
            
        logger.info(f"Human evaluation setup complete. Prompts saved to {prompts_path}")
        
        return prompts_path
