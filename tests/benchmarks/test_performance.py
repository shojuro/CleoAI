"""Performance benchmarking tests for CleoAI system."""
import time
import torch
import numpy as np
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional
from unittest.mock import MagicMock, patch
import pytest
import json

from src.model.moe_model import MoEModel, MoEConfig, ExpertGatingNetwork
from src.memory.memory_manager import MemoryManager, Conversation
from src.inference.inference_engine import InferenceEngine
from src.training.trainer import ModelTrainer, CustomDataset


class PerformanceMetrics:
    """Helper class to collect and analyze performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.start_gpu_memory: Optional[float] = None
    
    def start_measurement(self, name: str) -> None:
        """Start measuring performance for a specific operation."""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    
    def end_measurement(self, name: str) -> Dict[str, float]:
        """End measurement and record metrics."""
        if self.start_time is None:
            raise ValueError("Measurement not started")
        
        duration = time.time() - self.start_time
        memory_used = psutil.Process().memory_info().rss / 1024 / 1024 - self.start_memory
        
        result = {
            "duration": duration,
            "memory_used_mb": memory_used,
        }
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_used = torch.cuda.memory_allocated() / 1024 / 1024 - self.start_gpu_memory
            result["gpu_memory_used_mb"] = gpu_memory_used
        
        # Store metrics
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(duration)
        
        return result
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all measurements."""
        summary = {}
        for name, durations in self.metrics.items():
            summary[name] = {
                "mean": np.mean(durations),
                "std": np.std(durations),
                "min": np.min(durations),
                "max": np.max(durations),
                "median": np.median(durations),
            }
        return summary


@pytest.mark.benchmark
class TestModelPerformance:
    """Benchmark tests for model components."""
    
    @pytest.fixture
    def metrics(self):
        """Create performance metrics collector."""
        return PerformanceMetrics()
    
    @pytest.fixture
    def moe_configs(self):
        """Different MoE configurations to test."""
        return [
            MoEConfig(num_experts=4, num_experts_per_token=2),
            MoEConfig(num_experts=8, num_experts_per_token=2),
            MoEConfig(num_experts=16, num_experts_per_token=4),
            MoEConfig(num_experts=32, num_experts_per_token=4),
        ]
    
    def test_expert_routing_performance(self, metrics, moe_configs):
        """Benchmark expert routing performance with different configurations."""
        hidden_size = 768
        batch_sizes = [1, 8, 32]
        seq_lengths = [128, 512, 1024]
        
        results = {}
        
        for config in moe_configs:
            config_results = {}
            gating = ExpertGatingNetwork(hidden_size, config.num_experts)
            gating.eval()
            
            for batch_size in batch_sizes:
                for seq_len in seq_lengths:
                    test_name = f"experts_{config.num_experts}_batch_{batch_size}_seq_{seq_len}"
                    
                    # Create input
                    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
                    
                    # Warm up
                    with torch.no_grad():
                        _ = gating(hidden_states, config.num_experts_per_token, training=False)
                    
                    # Measure
                    metrics.start_measurement(test_name)
                    with torch.no_grad():
                        for _ in range(10):
                            _ = gating(hidden_states, config.num_experts_per_token, training=False)
                    
                    result = metrics.end_measurement(test_name)
                    config_results[test_name] = result
            
            results[f"config_experts_{config.num_experts}"] = config_results
        
        # Print results
        print("\nExpert Routing Performance Results:")
        for config_name, config_results in results.items():
            print(f"\n{config_name}:")
            for test_name, result in config_results.items():
                print(f"  {test_name}: {result['duration']:.4f}s")
    
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_precision_performance(self, metrics, dtype):
        """Benchmark performance with different precision levels."""
        if dtype == torch.bfloat16 and not torch.cuda.is_available():
            pytest.skip("BF16 requires CUDA")
        
        config = MoEConfig(num_experts=8)
        hidden_size = 768
        batch_size = 8
        seq_len = 512
        
        gating = ExpertGatingNetwork(hidden_size, config.num_experts)
        gating = gating.to(dtype=dtype)
        gating.eval()
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
        
        # Measure
        metrics.start_measurement(f"precision_{dtype}")
        with torch.no_grad():
            for _ in range(20):
                _ = gating(hidden_states, training=False)
        
        result = metrics.end_measurement(f"precision_{dtype}")
        print(f"\nPrecision {dtype} performance: {result['duration']:.4f}s")
    
    def test_memory_usage_scaling(self, metrics):
        """Test memory usage scaling with model size."""
        hidden_sizes = [256, 512, 768, 1024]
        results = {}
        
        for hidden_size in hidden_sizes:
            config = MoEConfig(num_experts=8)
            model = MoEModel(config, hidden_size=hidden_size, num_layers=12)
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            
            # Measure memory
            metrics.start_measurement(f"model_hidden_{hidden_size}")
            
            # Forward pass
            dummy_input = torch.randint(0, 1000, (1, 128))
            with torch.no_grad():
                _ = model(dummy_input)
            
            result = metrics.end_measurement(f"model_hidden_{hidden_size}")
            
            results[hidden_size] = {
                "num_params": num_params,
                "memory_mb": result["memory_used_mb"],
            }
        
        print("\nMemory Usage Scaling:")
        for hidden_size, info in results.items():
            print(f"  Hidden {hidden_size}: {info['num_params']:,} params, "
                  f"{info['memory_mb']:.2f} MB")


@pytest.mark.benchmark
class TestMemorySystemPerformance:
    """Benchmark tests for memory system components."""
    
    @pytest.fixture
    def memory_manager(self, temp_dir):
        """Create memory manager instance."""
        with patch('chromadb.PersistentClient'):
            return MemoryManager(base_path=temp_dir)
    
    def test_conversation_retrieval_speed(self, memory_manager, metrics):
        """Benchmark conversation retrieval with different sizes."""
        conversation_sizes = [10, 50, 100, 500]
        results = {}
        
        for size in conversation_sizes:
            # Create conversation with many messages
            conv = Conversation(f"conv_{size}", "user1")
            for i in range(size):
                conv.add_message("user", f"Message {i}" * 10)
                conv.add_message("assistant", f"Response {i}" * 10)
            
            memory_manager.short_term_memory.add_conversation(conv)
            
            # Measure retrieval
            metrics.start_measurement(f"retrieve_conv_size_{size}")
            for _ in range(100):
                _ = memory_manager.short_term_memory.get_recent_context(
                    f"conv_{size}", 
                    max_tokens=1000
                )
            result = metrics.end_measurement(f"retrieve_conv_size_{size}")
            
            results[size] = result
        
        print("\nConversation Retrieval Performance:")
        for size, result in results.items():
            print(f"  Size {size}: {result['duration']:.4f}s for 100 retrievals")
    
    def test_vector_search_performance(self, metrics):
        """Benchmark vector search performance."""
        with patch('chromadb.Collection') as mock_collection:
            # Mock vector search
            mock_collection.query.return_value = {
                'ids': [['id1', 'id2', 'id3']],
                'distances': [[0.1, 0.2, 0.3]],
                'metadatas': [[{}, {}, {}]],
                'documents': [['doc1', 'doc2', 'doc3']]
            }
            
            from src.memory.memory_manager import LongTermMemory
            memory = LongTermMemory(storage_type="vector")
            memory.vector_collection = mock_collection
            
            # Measure search performance
            metrics.start_measurement("vector_search")
            for _ in range(1000):
                _ = memory.semantic_search("test query", top_k=10)
            result = metrics.end_measurement("vector_search")
            
            print(f"\nVector Search Performance: {result['duration']:.4f}s for 1000 searches")
    
    def test_memory_persistence_performance(self, memory_manager, temp_dir, metrics):
        """Benchmark memory save/load performance."""
        # Add data
        for i in range(10):
            conv = Conversation(f"conv_{i}", f"user_{i}")
            for j in range(50):
                conv.add_message("user", f"Message {j}")
            memory_manager.short_term_memory.add_conversation(conv)
        
        # Measure save
        metrics.start_measurement("memory_save")
        memory_manager.save_state()
        save_result = metrics.end_measurement("memory_save")
        
        # Measure load
        new_manager = MemoryManager(base_path=temp_dir)
        metrics.start_measurement("memory_load")
        new_manager.load_state()
        load_result = metrics.end_measurement("memory_load")
        
        print(f"\nMemory Persistence Performance:")
        print(f"  Save: {save_result['duration']:.4f}s")
        print(f"  Load: {load_result['duration']:.4f}s")


@pytest.mark.benchmark
class TestInferencePerformance:
    """Benchmark tests for inference engine."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create mock inference engine."""
        with patch('src.inference.inference_engine.AutoTokenizer.from_pretrained'):
            with patch('src.inference.inference_engine.AutoModelForCausalLM.from_pretrained'):
                with patch('src.inference.inference_engine.MemoryManager'):
                    engine = InferenceEngine(
                        model_path="mock_model",
                        device="cpu",
                        use_moe=False
                    )
                    return engine
    
    def test_token_generation_speed(self, mock_engine, metrics):
        """Benchmark token generation speed."""
        sequence_lengths = [50, 100, 200, 500]
        results = {}
        
        with patch.object(mock_engine, 'model') as mock_model:
            # Mock model generation
            mock_model.generate.side_effect = lambda **kwargs: torch.randint(
                0, 1000, (1, kwargs.get('max_new_tokens', 100))
            )
            
            for seq_len in sequence_lengths:
                metrics.start_measurement(f"generate_{seq_len}_tokens")
                
                _ = mock_engine.generate_response(
                    user_id="test",
                    conversation_id="test",
                    query="Generate text",
                    max_new_tokens=seq_len
                )
                
                result = metrics.end_measurement(f"generate_{seq_len}_tokens")
                results[seq_len] = result
        
        print("\nToken Generation Performance:")
        for seq_len, result in results.items():
            tokens_per_sec = seq_len / result['duration']
            print(f"  {seq_len} tokens: {result['duration']:.4f}s "
                  f"({tokens_per_sec:.2f} tokens/sec)")
    
    def test_context_building_performance(self, mock_engine, metrics):
        """Benchmark context building performance."""
        context_sizes = [1, 5, 10, 20]
        results = {}
        
        for num_messages in context_sizes:
            # Mock memory context
            with patch.object(mock_engine.memory_manager, 'get_context') as mock_context:
                mock_context.return_value = {
                    "conversation_history": [
                        {"role": "user", "content": f"Message {i}"},
                        {"role": "assistant", "content": f"Response {i}"}
                    ] * num_messages,
                    "user_preferences": {},
                    "related_episodes": []
                }
                
                metrics.start_measurement(f"context_{num_messages}_messages")
                for _ in range(100):
                    _ = mock_engine.build_context("user", "conv", "query")
                result = metrics.end_measurement(f"context_{num_messages}_messages")
                
                results[num_messages] = result
        
        print("\nContext Building Performance:")
        for num_messages, result in results.items():
            print(f"  {num_messages} messages: {result['duration']:.4f}s for 100 builds")


@pytest.mark.benchmark
class TestTrainingPerformance:
    """Benchmark tests for training pipeline."""
    
    def test_data_loading_performance(self, metrics):
        """Benchmark data loading performance."""
        dataset_sizes = [100, 1000, 10000]
        batch_sizes = [4, 8, 16]
        results = {}
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (512,)),
            "attention_mask": torch.ones(512)
        }
        
        for size in dataset_sizes:
            size_results = {}
            
            # Create dataset
            data = [{"input": f"Input {i}", "output": f"Output {i}"} for i in range(size)]
            dataset = CustomDataset(data, mock_tokenizer)
            
            for batch_size in batch_sizes:
                dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0
                )
                
                # Measure loading
                metrics.start_measurement(f"load_size_{size}_batch_{batch_size}")
                for _ in dataloader:
                    pass  # Just iterate through
                result = metrics.end_measurement(f"load_size_{size}_batch_{batch_size}")
                
                size_results[batch_size] = result
            
            results[size] = size_results
        
        print("\nData Loading Performance:")
        for size, size_results in results.items():
            print(f"\nDataset size {size}:")
            for batch_size, result in size_results.items():
                samples_per_sec = size / result['duration']
                print(f"  Batch {batch_size}: {result['duration']:.4f}s "
                      f"({samples_per_sec:.2f} samples/sec)")
    
    def test_gradient_accumulation_performance(self, metrics):
        """Benchmark gradient accumulation impact."""
        accumulation_steps = [1, 4, 8, 16]
        results = {}
        
        # Mock model
        model = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        for acc_steps in accumulation_steps:
            metrics.start_measurement(f"grad_acc_{acc_steps}")
            
            for step in range(100):
                # Mock forward pass
                input_data = torch.randn(4, 768)
                output = model(input_data)
                loss = output.mean()
                
                # Accumulate gradients
                loss = loss / acc_steps
                loss.backward()
                
                if (step + 1) % acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            result = metrics.end_measurement(f"grad_acc_{acc_steps}")
            results[acc_steps] = result
        
        print("\nGradient Accumulation Performance:")
        for acc_steps, result in results.items():
            print(f"  Accumulation {acc_steps}: {result['duration']:.4f}s")


@pytest.mark.benchmark
class TestSystemIntegrationPerformance:
    """End-to-end system performance benchmarks."""
    
    def test_full_inference_pipeline(self, temp_dir, metrics):
        """Benchmark complete inference pipeline."""
        with patch('src.inference.inference_engine.AutoTokenizer.from_pretrained') as mock_tok:
            with patch('src.inference.inference_engine.AutoModelForCausalLM.from_pretrained') as mock_model:
                # Setup mocks
                tokenizer = MagicMock()
                tokenizer.encode.return_value = torch.tensor([1, 2, 3])
                tokenizer.decode.return_value = "Generated response"
                mock_tok.return_value = tokenizer
                
                model = MagicMock()
                model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
                mock_model.return_value = model
                
                # Create engine
                engine = InferenceEngine(
                    model_path=str(temp_dir),
                    use_moe=False,
                    device="cpu"
                )
                
                # Measure full pipeline
                queries = [
                    "Short query",
                    "This is a medium length query with more context",
                    "This is a very long query " * 20
                ]
                
                for query in queries:
                    metrics.start_measurement(f"pipeline_query_len_{len(query)}")
                    
                    for _ in range(10):
                        _ = engine.generate_response(
                            user_id="user1",
                            conversation_id="conv1",
                            query=query
                        )
                    
                    result = metrics.end_measurement(f"pipeline_query_len_{len(query)}")
                    print(f"\nPipeline performance for query length {len(query)}: "
                          f"{result['duration']:.4f}s")
    
    def test_concurrent_request_handling(self, metrics):
        """Benchmark concurrent request handling."""
        import threading
        import queue
        
        num_threads = [1, 2, 4, 8]
        num_requests = 100
        
        def process_request(request_queue, result_queue):
            while True:
                try:
                    request = request_queue.get(timeout=1)
                    # Simulate processing
                    time.sleep(0.01)
                    result_queue.put(f"Processed: {request}")
                    request_queue.task_done()
                except queue.Empty:
                    break
        
        for n_threads in num_threads:
            request_queue = queue.Queue()
            result_queue = queue.Queue()
            
            # Add requests
            for i in range(num_requests):
                request_queue.put(f"Request {i}")
            
            # Measure concurrent processing
            metrics.start_measurement(f"concurrent_{n_threads}_threads")
            
            threads = []
            for _ in range(n_threads):
                t = threading.Thread(target=process_request, args=(request_queue, result_queue))
                t.start()
                threads.append(t)
            
            # Wait for completion
            request_queue.join()
            for t in threads:
                t.join()
            
            result = metrics.end_measurement(f"concurrent_{n_threads}_threads")
            
            requests_per_sec = num_requests / result['duration']
            print(f"\nConcurrent processing with {n_threads} threads: "
                  f"{result['duration']:.4f}s ({requests_per_sec:.2f} req/sec)")


def generate_performance_report(metrics: PerformanceMetrics, output_path: Path) -> None:
    """Generate a comprehensive performance report."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "metrics_summary": metrics.get_summary(),
        "raw_metrics": metrics.metrics,
    }
    
    if torch.cuda.is_available():
        report["system_info"]["gpu_info"] = {
            "name": torch.cuda.get_device_name(0),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024,
        }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nPerformance report saved to: {output_path}")


if __name__ == "__main__":
    # Run benchmarks and generate report
    pytest.main([__file__, "-v", "-m", "benchmark", "--tb=short"])