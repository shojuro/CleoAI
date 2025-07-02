"""
Main entry point for the AI Autonomous Agent.
"""
import os
import sys
import time
import argparse
import logging
from pathlib import Path

from src.model.moe_model import load_pretrained_model_with_moe
from src.memory.memory_manager import MemoryManager
from src.training.trainer import ModelTrainer
from src.inference.inference_engine import InferenceEngine
from src.utils.config_validator import ensure_valid_configuration
from config import model_config, memory_config, training_config, evaluation_config, project_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/main_{Path(__file__).stem}.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="AI Autonomous Agent CLI")
    
    # Main command arguments
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--model", type=str, default=model_config.model_name, help="Model name or path")
    train_parser.add_argument("--output-dir", type=str, default="models/output", help="Output directory")
    train_parser.add_argument("--use-moe", action="store_true", default=True, help="Use Mixture of Experts architecture")
    train_parser.add_argument("--num-experts", type=int, default=model_config.num_experts, help="Number of experts")
    train_parser.add_argument("--deepspeed", action="store_true", default=True, help="Use DeepSpeed for training")
    
    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference with the model")
    infer_parser.add_argument("--model", type=str, required=True, help="Model path")
    infer_parser.add_argument("--use-moe", action="store_true", default=True, help="Use Mixture of Experts architecture")
    infer_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    infer_parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum new tokens to generate")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--model", type=str, required=True, help="Model path")
    eval_parser.add_argument("--dataset", type=str, default=None, help="Dataset to evaluate on")
    eval_parser.add_argument("--output-dir", type=str, default="outputs/evaluation", help="Output directory")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.add_argument("--model", type=str, required=True, help="Model path")
    interactive_parser.add_argument("--use-moe", action="store_true", default=True, help="Use Mixture of Experts architecture")
    interactive_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export the model")
    export_parser.add_argument("--model", type=str, required=True, help="Model path")
    export_parser.add_argument("--output", type=str, required=True, help="Output path")
    export_parser.add_argument("--format", type=str, default="onnx", help="Export format (onnx, torchscript)")
    
    return parser.parse_args()

def train(args):
    """Train the model"""
    logger.info(f"Training model {args.model}")
    
    # Initialize trainer
    trainer = ModelTrainer(
        model_name=args.model,
        output_dir=args.output_dir,
        config=training_config,
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        use_deepspeed=args.deepspeed
    )
    
    # Run training pipeline
    results = trainer.run_full_training()
    
    logger.info(f"Training completed. Model saved to {results['final_model_path']}")
    return results

def infer(args):
    """Run inference with the model"""
    logger.info(f"Running inference with model {args.model}")
    
    # Initialize inference engine
    engine = InferenceEngine(
        model_path=args.model,
        use_moe=args.use_moe,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )
    
    # Example inference
    response = engine.generate_response(
        user_id="example_user",
        conversation_id="example_conversation",
        query="Hello, how are you today?"
    )
    
    logger.info(f"Generated response: {response}")
    return response

def evaluate(args):
    """Evaluate the model"""
    logger.info(f"Evaluating model {args.model}")
    
    # Initialize trainer for evaluation
    trainer = ModelTrainer(
        model_name=args.model,
        output_dir=args.output_dir,
        config=evaluation_config,
        use_moe=True
    )
    
    # Determine datasets to evaluate on
    datasets = [args.dataset] if args.dataset else evaluation_config.benchmark_datasets
    
    # Run evaluation
    results = trainer.evaluate_model(test_dataset_names=datasets)
    
    logger.info(f"Evaluation completed. Results saved to {args.output_dir}")
    return results

def interactive(args):
    """Run interactive mode"""
    logger.info(f"Starting interactive mode with model {args.model}")
    
    # Initialize inference engine
    engine = InferenceEngine(
        model_path=args.model,
        use_moe=args.use_moe,
        temperature=args.temperature
    )
    
    # Create conversation ID and user ID
    conversation_id = f"interactive_{os.getpid()}_{int(time.time())}"
    user_id = "interactive_user"
    
    print("\n===== AI Autonomous Agent Interactive Mode =====")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check if user wants to exit
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nExiting interactive mode. Goodbye!")
                break
                
            # Generate response
            print("\nAI: ", end="", flush=True)
            
            # Stream the response
            full_response = ""
            def stream_callback(token):
                print(token, end="", flush=True)
                
            response = engine.stream_response(
                user_id=user_id,
                conversation_id=conversation_id,
                query=user_input,
                callback=stream_callback
            )
            
            print()  # New line after response
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"\nAn error occurred: {e}")
    
    return {"status": "completed", "conversation_id": conversation_id}

def export(args):
    """Export the model to a specific format"""
    logger.info(f"Exporting model {args.model} to {args.format} format")
    
    if args.format.lower() == "onnx":
        # Export to ONNX format
        from transformers import AutoTokenizer
        import torch
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        if args.use_moe:
            model = load_pretrained_model_with_moe(model_name=args.model)
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(args.model)
            
        # Export model to ONNX
        dummy_input = tokenizer("Hello, how are you?", return_tensors="pt")
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Export
        torch.onnx.export(
            model,
            (dummy_input.input_ids, dummy_input.attention_mask),
            args.output,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=12
        )
        
        logger.info(f"Model exported to {args.output}")
        
    elif args.format.lower() == "torchscript":
        # Export to TorchScript format
        import torch
        from transformers import AutoTokenizer
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        if args.use_moe:
            model = load_pretrained_model_with_moe(model_name=args.model)
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(args.model)
            
        # Create a script module
        dummy_input = tokenizer("Hello, how are you?", return_tensors="pt")
        traced_model = torch.jit.trace(
            model,
            (dummy_input.input_ids, dummy_input.attention_mask)
        )
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save the traced model
        traced_model.save(args.output)
        
        logger.info(f"Model exported to {args.output}")
        
    else:
        logger.error(f"Unsupported export format: {args.format}")
        return {"error": f"Unsupported export format: {args.format}"}
        
    return {"status": "completed", "output_path": args.output}

def main():
    """Main entry point"""
    args = parse_args()
    
    try:
        # Execute the appropriate command
        if args.command == "train":
            train(args)
        elif args.command == "infer":
            infer(args)
        elif args.command == "evaluate":
            evaluate(args)
        elif args.command == "interactive":
            interactive(args)
        elif args.command == "export":
            export(args)
        else:
            # If no command is specified, print help
            print("Please specify a command. Use --help for more information.")
            return 1
            
    except Exception as e:
        logger.error(f"Error executing command {args.command}: {e}")
        print(f"Error: {e}")
        return 1
        
    return 0
"""
Main entry point for the AI Autonomous Agent.
Provides command-line interface for training, inference, and evaluation.
"""
import os
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch

# Import project modules
from config import (
    model_config, 
    memory_config,
    memory_backend_config,
    api_config, 
    training_config, 
    evaluation_config, 
    project_config,
    MODELS_DIR, 
    DATA_DIR
)
from src.model.moe_model import MoEModel, load_pretrained_model_with_moe
from src.memory.memory_manager import MemoryManager
from src.memory.adapters.memory_adapter import HybridMemoryAdapter
from src.training.trainer import ModelTrainer
from src.inference.inference_engine import (
    create_inference_engine, 
    create_interactive_session
)

# Import monitoring modules
from src.monitoring import (
    setup_logging,
    initialize_tracing,
    initialize_error_tracking,
    instrument_app,
    MetricsConfig,
    TracingConfig,
    LogConfig,
    ErrorTrackingConfig
)

# Configure logging with enhanced monitoring
log_config = LogConfig()
setup_logging(log_config)
logger = logging.getLogger(__name__)

# Initialize tracing
tracing_config = TracingConfig()
tracer = initialize_tracing(tracing_config)

# Initialize error tracking
error_config = ErrorTrackingConfig()
initialize_error_tracking(error_config)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AI Autonomous Agent CLI")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # API mode
    api_parser = subparsers.add_parser("api", help="Start GraphQL API server")
    api_parser.add_argument("--host", type=str, default=api_config.api_host,
                           help="API server host")
    api_parser.add_argument("--port", type=int, default=api_config.api_port,
                           help="API server port")
    api_parser.add_argument("--workers", type=int, default=api_config.api_workers,
                           help="Number of worker processes")
    api_parser.add_argument("--debug", action="store_true", default=api_config.graphql_debug,
                           help="Enable debug mode")
    
    # Training mode
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--model", type=str, default=model_config.model_name,
                             help="Base model to use for training")
    train_parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR / "output"),
                             help="Directory to save model checkpoints")
    train_parser.add_argument("--use-moe", action="store_true", default=model_config.use_moe,
                             help="Use Mixture-of-Experts architecture")
    train_parser.add_argument("--num-experts", type=int, default=model_config.num_experts,
                             help="Number of experts for MoE")
    train_parser.add_argument("--phase", type=str, default=None,
                             help="Specific training phase to run (e.g., 'foundation')")
    train_parser.add_argument("--deepspeed", action="store_true", default=training_config.use_deepspeed,
                             help="Use DeepSpeed for training")
    train_parser.add_argument("--no-wandb", action="store_true", default=not project_config.use_wandb,
                             help="Disable Weights & Biases logging")
    
    # Inference mode
    infer_parser = subparsers.add_parser("infer", help="Run inference with a trained model")
    infer_parser.add_argument("--model", type=str, required=True,
                             help="Path to trained model")
    infer_parser.add_argument("--use-moe", action="store_true", default=model_config.use_moe,
                             help="Use Mixture-of-Experts architecture")
    infer_parser.add_argument("--device", type=str, default=model_config.device,
                             help="Device to run inference on (cuda, cpu)")
    infer_parser.add_argument("--precision", type=str, default=model_config.precision,
                             choices=["fp16", "bf16", "fp32"], help="Precision for inference")
    infer_parser.add_argument("--temperature", type=float, default=model_config.temperature,
                             help="Temperature for sampling")
    infer_parser.add_argument("--input", type=str, default=None,
                             help="Input text for inference")
    infer_parser.add_argument("--output", type=str, default=None,
                             help="Output file to save results")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Interactive chat with the model")
    interactive_parser.add_argument("--model", type=str, required=True,
                                  help="Path to trained model")
    interactive_parser.add_argument("--use-moe", action="store_true", default=model_config.use_moe,
                                  help="Use Mixture-of-Experts architecture")
    interactive_parser.add_argument("--device", type=str, default=model_config.device,
                                  help="Device to run inference on (cuda, cpu)")
    interactive_parser.add_argument("--precision", type=str, default=model_config.precision,
                                  choices=["fp16", "bf16", "fp32"], help="Precision for inference")
    interactive_parser.add_argument("--temperature", type=float, default=model_config.temperature,
                                  help="Temperature for sampling")
    interactive_parser.add_argument("--user-id", type=str, default="default_user",
                                  help="User ID for memory tracking")
    interactive_parser.add_argument("--system-prompt", type=str, default=None,
                                  help="System prompt for the model")
    interactive_parser.add_argument("--memory-dir", type=str, default=None,
                                  help="Directory to load/save memory state")
    
    # Evaluation mode
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", type=str, required=True,
                            help="Path to trained model")
    eval_parser.add_argument("--use-moe", action="store_true", default=model_config.use_moe,
                            help="Use Mixture-of-Experts architecture")
    eval_parser.add_argument("--dataset", type=str, default=None,
                            help="Dataset to evaluate on (or 'all' for all datasets)")
    eval_parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR / "evaluation"),
                            help="Directory to save evaluation results")
    eval_parser.add_argument("--human-eval", action="store_true", default=False,
                            help="Create samples for human evaluation")
    eval_parser.add_argument("--num-samples", type=int, default=evaluation_config.human_eval_samples,
                            help="Number of samples for human evaluation")
    
    return parser.parse_args()

def train_model(args):
    """Train the model with the specified arguments."""
    logger.info(f"Starting training with model {args.model}")
    
    # Update config with CLI arguments
    if args.model:
        model_config.model_name = args.model
    if args.use_moe is not None:
        model_config.use_moe = args.use_moe
    if args.num_experts:
        model_config.num_experts = args.num_experts
    if args.deepspeed is not None:
        training_config.use_deepspeed = args.deepspeed
    
    # Initialize trainer
    trainer = ModelTrainer(
        model_name=model_config.model_name,
        output_dir=args.output_dir,
        config=training_config,
        use_moe=model_config.use_moe,
        num_experts=model_config.num_experts,
        num_experts_per_token=model_config.num_experts_per_token,
        device=model_config.device,
        fp16=(model_config.precision == "fp16"),
        bf16=(model_config.precision == "bf16"),
        use_deepspeed=training_config.use_deepspeed,
        deepspeed_config_path=training_config.deepspeed_config_path,
        log_to_wandb=not args.no_wandb,
        wandb_project=project_config.wandb_project,
        wandb_entity=project_config.wandb_entity
    )
    
    # Train specific phase or all phases
    if args.phase:
        logger.info(f"Training specific phase: {args.phase}")
        phases = trainer.create_training_phases()
        
        # Find the requested phase
        selected_phase = None
        for phase in phases:
            if phase.phase_name == args.phase:
                selected_phase = phase
                break
                
        if selected_phase:
            trainer.train_phase(selected_phase)
        else:
            logger.error(f"Phase {args.phase} not found")
    else:
        logger.info("Starting full training pipeline")
        trainer.run_full_training()
    
    logger.info("Training completed")

def run_inference(args):
    """Run inference with the specified arguments."""
    logger.info(f"Running inference with model {args.model}")
    
    # Create inference engine
    engine = create_inference_engine(
        model_path=args.model,
        use_moe=args.use_moe,
        device=args.device,
        precision=args.precision,
        config={
            "temperature": args.temperature,
            "max_length": model_config.max_length,
            "max_new_tokens": model_config.max_new_tokens,
            "top_p": model_config.top_p,
            "top_k": model_config.top_k,
            "repetition_penalty": model_config.repetition_penalty
        }
    )
    
    # Run inference
    if args.input:
        # Get input from file or argument
        if os.path.isfile(args.input):
            with open(args.input, "r") as f:
                input_text = f.read()
        else:
            input_text = args.input
            
        # Generate response
        response = engine.respond(
            user_message=input_text,
            user_id="default_user"
        )
        
        # Print or save output
        if args.output:
            with open(args.output, "w") as f:
                json.dump(response, f, indent=2)
            logger.info(f"Response saved to {args.output}")
        else:
            print("\nInput:")
            print(input_text)
            print("\nResponse:")
            print(response["response"])
            print(f"\nGenerated in {response['response_time']:.2f} seconds")
    else:
        logger.error("No input provided for inference")

def run_interactive(args):
    """Run interactive session with the specified arguments."""
    logger.info(f"Starting interactive session with model {args.model}")
    
    # Create inference engine
    engine = create_inference_engine(
        model_path=args.model,
        use_moe=args.use_moe,
        device=args.device,
        precision=args.precision,
        config={
            "temperature": args.temperature,
            "max_length": model_config.max_length,
            "max_new_tokens": model_config.max_new_tokens,
            "top_p": model_config.top_p,
            "top_k": model_config.top_k,
            "repetition_penalty": model_config.repetition_penalty
        }
    )
    
    # Load memory state if specified
    if args.memory_dir:
        memory_dir = Path(args.memory_dir)
        if memory_dir.exists():
            logger.info(f"Loading memory state from {args.memory_dir}")
            engine.load_memory_state(args.memory_dir)
    
    # Create interactive session
    session = create_interactive_session(
        engine=engine,
        user_id=args.user_id,
        system_prompt=args.system_prompt
    )
    
    # Run console session
    session.run_console_session()
    
    # Save memory state if specified
    if args.memory_dir:
        logger.info(f"Saving memory state to {args.memory_dir}")
        memory_dir = Path(args.memory_dir)
        memory_dir.mkdir(parents=True, exist_ok=True)
        engine.save_memory_state(args.memory_dir)

def evaluate_model(args):
    """Evaluate the model with the specified arguments."""
    logger.info(f"Evaluating model {args.model}")
    
    # Update config with CLI arguments
    if args.model:
        model_config.model_name = args.model
    if args.use_moe is not None:
        model_config.use_moe = args.use_moe
    
    # Initialize trainer for evaluation
    trainer = ModelTrainer(
        model_name=args.model,
        output_dir=args.output_dir,
        config=training_config,
        use_moe=args.use_moe,
        device=model_config.device,
        fp16=(model_config.precision == "fp16"),
        bf16=(model_config.precision == "bf16"),
        use_deepspeed=False,
        log_to_wandb=project_config.use_wandb,
        wandb_project=project_config.wandb_project,
        wandb_entity=project_config.wandb_entity
    )
    
    # Determine datasets to evaluate on
    if args.dataset == "all":
        datasets = evaluation_config.benchmark_datasets
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = evaluation_config.benchmark_datasets
    
    # Run evaluation
    eval_results = trainer.evaluate_model(
        test_dataset_names=datasets,
        metrics=evaluation_config.metrics
    )
    
    # Print results
    print("\nEvaluation Results:")
    for dataset, results in eval_results.items():
        print(f"\n{dataset}:")
        for metric, value in results.items():
            print(f"  {metric}: {value}")
    
    # Create human evaluation samples if requested
    if args.human_eval:
        prompts_path = trainer.human_evaluation_setup(num_samples=args.num_samples)
        print(f"\nHuman evaluation prompts created at: {prompts_path}")

def run_api_server(args):
    """Run the GraphQL API server."""
    logger.info(f"Starting API server on {args.host}:{args.port}")
    
    # Validate configuration before starting
    ensure_valid_configuration()
    
    # Initialize memory adapter with hybrid approach
    memory_adapter = HybridMemoryAdapter(
        legacy_manager=MemoryManager() if memory_backend_config.use_sqlite else None,
        redis_config={
            "host": memory_backend_config.redis_host,
            "port": memory_backend_config.redis_port,
            "password": memory_backend_config.redis_password
        } if memory_backend_config.use_redis else None,
        supabase_config={
            "url": memory_backend_config.supabase_url,
            "key": memory_backend_config.supabase_anon_key
        } if memory_backend_config.use_supabase else None,
        pinecone_config={
            "api_key": memory_backend_config.pinecone_api_key,
            "environment": memory_backend_config.pinecone_environment,
            "index_name": memory_backend_config.pinecone_index_name,
            "dimension": memory_backend_config.pinecone_dimension
        } if memory_backend_config.use_pinecone else None,
        mongodb_config={
            "connection_string": memory_backend_config.mongodb_connection_string,
            "database": memory_backend_config.mongodb_database
        } if memory_backend_config.use_mongodb else None,
        use_legacy=memory_backend_config.use_sqlite,
        use_distributed=True
    )
    
    # Check if we have a model to serve
    model_path = None
    if hasattr(args, 'model'):
        model_path = args.model
    else:
        # Try to find a model in the models directory
        import glob
        model_files = glob.glob(str(MODELS_DIR / "**/config.json"), recursive=True)
        if model_files:
            model_path = Path(model_files[0]).parent
            logger.info(f"Found model at: {model_path}")
    
    # Initialize inference engine if model available
    inference_engine = None
    if model_path:
        try:
            inference_engine = create_inference_engine(
                model_path=str(model_path),
                use_moe=model_config.use_moe,
                device=model_config.device,
                precision=model_config.precision
            )
            logger.info("Inference engine initialized")
        except Exception as e:
            logger.warning(f"Could not initialize inference engine: {e}")
            logger.warning("API will run without inference capabilities")
    
    # Create FastAPI app
    from src.api.api_router import create_api_app
    app = create_api_app(
        inference_engine=inference_engine,
        memory_adapter=memory_adapter,
        cors_origins=api_config.cors_origins,
        debug=args.debug
    )
    
    # Apply monitoring configuration
    metrics_config = MetricsConfig()
    metrics_config.apply_to_app(app)
    
    # Instrument app for tracing
    instrument_app(app, tracing_config)
    
    # Run with uvicorn
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers if not args.debug else 1,
        log_level="debug" if args.debug else "info"
    )


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Check CUDA availability for modes that need it
    if hasattr(args, 'device') and args.device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            args.device = "cpu"
    
    # Run appropriate function based on mode
    if args.mode == "train":
        train_model(args)
    elif args.mode == "infer":
        run_inference(args)
    elif args.mode == "interactive":
        run_interactive(args)
    elif args.mode == "evaluate":
        evaluate_model(args)
    elif args.mode == "api":
        run_api_server(args)
    else:
        # No mode specified, show help
        parse_args()

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    sys.exit(main())
