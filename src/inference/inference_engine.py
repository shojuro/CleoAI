"""
Inference engine for the AI Autonomous Agent.
Handles request processing, context building, and response generation.
"""
import os
import json
import time
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList
)
"""
Inference engine for the AI Autonomous Agent.
Provides optimized text generation with context building and memory integration.
"""
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.model.moe_model import MoEModel
from src.memory.memory_manager import MemoryManager, Conversation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Inference engine for text generation with memory integration.
    """
    def __init__(
        self,
        model_path: str,
        use_moe: bool = True,
        device: str = "cuda",
        precision: str = "bf16",
        max_length: int = 2048,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        memory_config: Optional[Dict[str, Any]] = None
    ):
        self.model_path = model_path
        self.use_moe = use_moe
        self.device = device
        self.precision = precision
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        
        # Initialize model and tokenizer
        self._init_model_and_tokenizer()
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(**(memory_config or {}))
        
        # Initialize conversation tracking
        self.active_conversations = {}
        
        logger.info(f"Initialized InferenceEngine with model {model_path}")
    
    def _init_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Check if MoE config exists
        moe_config_path = os.path.join(self.model_path, "moe_config.json")
        use_moe_model = self.use_moe and os.path.exists(moe_config_path)
        
        # Load model based on type
        if use_moe_model:
            logger.info("Loading MoE model")
            self.model = MoEModel.from_pretrained(self.model_path)
        else:
            logger.info("Loading standard model")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            
        # Set model precision
        if self.precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.model = self.model.to(torch.bfloat16)
        elif self.precision == "fp16" and torch.cuda.is_available():
            self.model = self.model.to(torch.float16)
            
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def _build_prompt(
        self, 
        user_message: str, 
        conversation_id: str, 
        user_id: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Build a full prompt with context from memory and conversation history.
        """
        # Get conversation history
        conversation_history = self.memory_manager.get_conversation_history(
            conversation_id=conversation_id,
            limit=20  # Limit to recent messages
        )
        
        # Format conversation history
        formatted_history = ""
        for msg in conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['content']}\n\n"
        
        # Get memory context
        memory_context = self.memory_manager.build_context(
            user_id=user_id,
            conversation_id=conversation_id,
            query=user_message
        )
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = (
                "You are a helpful AI assistant. You provide accurate, detailed, and "
                "thoughtful responses to the user's questions and requests. You are "
                "knowledgeable, friendly, and communicative."
            )
        
        # Build full prompt
        full_prompt = f"""
### System
{system_prompt}

{memory_context}

### Chat History
{formatted_history}

### User
{user_message}

### Assistant
"""
        
        return full_prompt.strip()
    
    def generate(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from a prompt.
        """
        # Use provided parameters or defaults
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        top_k = top_k if top_k is not None else self.top_k
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length
        ).to(self.device)
        
        # Set pad token ID if needed
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id
        
        # Generate text
        with torch.no_grad():
            if hasattr(self.model, "generate"):
                # Use standard generate method
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id
                )
            else:
                # Use custom generate method for MoE model
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id
                )
        
        # Get generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the newly generated content
        prompt_length = len(prompt)
        new_text = generated_text[prompt_length:].strip()
        
        # Clean up the response
        if "### User" in new_text:
            new_text = new_text.split("### User")[0].strip()
            
        if "### Assistant" in new_text:
            new_text = new_text.split("### Assistant")[1].strip()
            
        return new_text
    
    def respond(
        self, 
        user_message: str, 
        conversation_id: Optional[str] = None, 
        user_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message and generate a response.
        
        Args:
            user_message: Message from the user
            conversation_id: ID for the conversation (will create a new one if None)
            user_id: ID for the user (will use default if None)
            system_prompt: Optional system prompt to override the default
            generation_params: Optional parameters for text generation
            
        Returns:
            Dictionary with response text and metadata
        """
        # Start timing
        start_time = time.time()
        
        # Set defaults if not provided
        if not conversation_id:
            conversation_id = f"conv_{int(time.time())}"
            
        if not user_id:
            user_id = "default_user"
            
        # Add user message to memory
        self.memory_manager.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="user",
            content=user_message
        )
        
        # Build prompt
        prompt = self._build_prompt(
            user_message=user_message,
            conversation_id=conversation_id,
            user_id=user_id,
            system_prompt=system_prompt
        )
        
        # Generate response
        generation_params = generation_params or {}
        response_text = self.generate(prompt, **generation_params)
        
        # Add assistant response to memory
        self.memory_manager.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="assistant",
            content=response_text
        )
        
        # Extract user preferences from the message
        self.memory_manager.long_term.infer_preference(
            user_id=user_id,
            text=user_message,
            confidence=0.6,
            source="message"
        )
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Prepare response
        response = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "input": user_message,
            "response": response_text,
            "response_time": response_time,
            "timestamp": time.time()
        }
        
        return response
    
    def continue_conversation(
        self, 
        conversation_id: str, 
        user_message: str, 
        user_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Continue an existing conversation.
        
        Args:
            conversation_id: ID for the existing conversation
            user_message: New message from the user
            user_id: ID for the user (will use from conversation if None)
            system_prompt: Optional system prompt to override the default
            generation_params: Optional parameters for text generation
            
        Returns:
            Dictionary with response text and metadata
        """
        # Get conversation
        conversation = self.memory_manager.short_term.get_conversation(conversation_id)
        
        # Use conversation's user_id if not provided
        if not user_id and conversation:
            user_id = conversation.user_id
        elif not user_id:
            user_id = "default_user"
            
        # Respond to the message
        return self.respond(
            user_message=user_message,
            conversation_id=conversation_id,
            user_id=user_id,
            system_prompt=system_prompt,
            generation_params=generation_params
        )
    
    def create_episodic_memory(
        self, 
        conversation_id: str, 
        user_id: Optional[str] = None,
        importance: float = 0.7,
        emotion: str = "neutral"
    ) -> Dict[str, Any]:
        """
        Create an episodic memory from a conversation.
        
        Args:
            conversation_id: ID of the conversation to memorize
            user_id: ID of the user (will use from conversation if None)
            importance: Importance score for the memory (0-1)
            emotion: Emotional tone of the memory
            
        Returns:
            Dictionary with memory information
        """
        # Get conversation
        conversation = self.memory_manager.short_term.get_conversation(conversation_id)
        
        if not conversation:
            logger.warning(f"Conversation {conversation_id} not found")
            return {"error": "Conversation not found"}
            
        # Use conversation's user_id if not provided
        if not user_id:
            user_id = conversation.user_id
            
        # Create episodic memory
        memory = self.memory_manager.episodic.create_memory_from_conversation(
            user_id=user_id,
            conversation=conversation,
            importance=importance,
            emotion=emotion
        )
        
        return {
            "memory_id": memory.memory_id,
            "title": memory.title,
            "user_id": memory.user_id,
            "created_at": memory.created_at,
            "importance": memory.importance,
            "emotion": memory.emotion
        }
    
    def retrieve_memories(
        self, 
        query: str, 
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on a query.
        
        Args:
            query: Search query
            user_id: ID of the user to retrieve memories for (optional)
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memory dictionaries
        """
        # Search episodic memories
        memories = self.memory_manager.episodic.search_memories(
            query=query,
            user_id=user_id,
            limit=limit
        )
        
        # Format memories
        memory_dicts = []
        for memory in memories:
            memory_dicts.append({
                "memory_id": memory.memory_id,
                "title": memory.title,
                "content": memory.content,
                "created_at": memory.created_at,
                "importance": memory.importance,
                "emotion": memory.emotion
            })
            
        return memory_dicts
    
    def save_memory_state(self, save_dir: str):
        """
        Save the current state of all memory systems.
        
        Args:
            save_dir: Directory to save memory state
        """
        # Create directory if it doesn't exist
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Save all memory systems
        self.memory_manager.save_all(save_dir)
        
        logger.info(f"Saved memory state to {save_dir}")
        
    def load_memory_state(self, load_dir: str):
        """
        Load memory state from disk.
        
        Args:
            load_dir: Directory to load memory state from
        """
        if not Path(load_dir).exists():
            logger.warning(f"Memory state directory {load_dir} does not exist")
            return
            
        # Load all memory systems
        self.memory_manager.load_all(load_dir)
        
        logger.info(f"Loaded memory state from {load_dir}")

class InteractiveSession:
    """
    Interactive session for chatting with the AI Autonomous Agent.
    """
    def __init__(
        self,
        engine: InferenceEngine,
        user_id: str = "default_user",
        system_prompt: Optional[str] = None
    ):
        self.engine = engine
        self.user_id = user_id
        self.system_prompt = system_prompt
        self.conversation_id = f"session_{int(time.time())}"
        
        logger.info(f"Started interactive session with conversation ID: {self.conversation_id}")
    
    def chat(self, user_input: str, generation_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a user input and return the model's response.
        
        Args:
            user_input: Input text from the user
            generation_params: Optional parameters for text generation
            
        Returns:
            Response text from the model
        """
        response = self.engine.respond(
            user_message=user_input,
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            system_prompt=self.system_prompt,
            generation_params=generation_params
        )
        
        return response["response"]
    
    def run_console_session(self):
        """
        Run an interactive console session.
        """
        print("\n==== Interactive AI Agent Session ====")
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'save <filename>' to save the conversation")
        print("======================================\n")
        
        try:
            while True:
                user_input = input("\nYou: ").strip()
                
                # Check for special commands
                if user_input.lower() in ["exit", "quit"]:
                    print("\nEnding session...")
                    break
                    
                elif user_input.lower().startswith("save "):
                    filename = user_input[5:].strip()
                    self.save_conversation(filename)
                    print(f"\nConversation saved to {filename}")
                    continue
                    
                # Normal chat interaction
                start_time = time.time()
                response = self.chat(user_input)
                end_time = time.time()
                
                print(f"\nAssistant: {response}")
                print(f"\n[Response time: {end_time - start_time:.2f}s]")
                
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Ending session...")
            
        except Exception as e:
            print(f"\n\nError occurred: {e}")
            
        finally:
            # Save the conversation as an episodic memory
            self.engine.create_episodic_memory(
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                importance=0.7
            )
            print("\nSession ended and saved to memory")
    
    def save_conversation(self, filename: str):
        """
        Save the current conversation to a file.
        
        Args:
            filename: Name of the file to save to
        """
        # Get conversation
        conversation = self.engine.memory_manager.short_term.get_conversation(self.conversation_id)
        
        if not conversation:
            print("No conversation to save")
            return
            
        # Format conversation
        formatted_conversation = {
            "conversation_id": conversation.conversation_id,
            "user_id": conversation.user_id,
            "created_at": conversation.created_at,
            "messages": conversation.messages
        }
        
        # Save to file
        with open(filename, "w") as f:
            json.dump(formatted_conversation, f, indent=2)

def create_inference_engine(
    model_path: str,
    use_moe: bool = True,
    device: str = "cuda",
    precision: str = "bf16",
    config: Optional[Dict[str, Any]] = None
) -> InferenceEngine:
    """
    Create an inference engine with the specified configuration.
    
    Args:
        model_path: Path to the model
        use_moe: Whether to use MoE architecture
        device: Device to run inference on
        precision: Precision to use (fp16, bf16, fp32)
        config: Additional configuration parameters
        
    Returns:
        Configured InferenceEngine
    """
    config = config or {}
    
    # Create engine
    engine = InferenceEngine(
        model_path=model_path,
        use_moe=use_moe,
        device=device,
        precision=precision,
        max_length=config.get("max_length", 2048),
        max_new_tokens=config.get("max_new_tokens", 512),
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.9),
        top_k=config.get("top_k", 50),
        repetition_penalty=config.get("repetition_penalty", 1.1),
        memory_config=config.get("memory_config", None)
    )
    
    return engine

def create_interactive_session(
    engine: InferenceEngine,
    user_id: str = "default_user",
    system_prompt: Optional[str] = None
) -> InteractiveSession:
    """
    Create an interactive session with the specified configuration.
    
    Args:
        engine: Inference engine to use
        user_id: ID of the user
        system_prompt: System prompt to use
        
    Returns:
        Configured InteractiveSession
    """
    return InteractiveSession(
        engine=engine,
        user_id=user_id,
        system_prompt=system_prompt
    )
from src.memory.memory_manager import MemoryManager, Conversation
from src.model.moe_model import MoEModel, load_pretrained_model_with_moe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for text generation"""
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class InferenceEngine:
    """
    Inference engine for the AI Autonomous Agent.
    Handles loading the model, building context from memory,
    and generating responses.
    """
    def __init__(
        self,
        model_path: str,
        use_moe: bool = True,
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        memory_manager: Optional[MemoryManager] = None
    ):
        self.model_path = model_path
        self.use_moe = use_moe
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        
        # Initialize model and tokenizer
        self._init_model_and_tokenizer()
        
        # Initialize memory manager if not provided
        self.memory_manager = memory_manager or MemoryManager()
        
        logger.info(f"Inference engine initialized with model from {model_path}")
    
    def _init_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        if self.use_moe:
            logger.info("Loading MoE model")
            # Check if this is a HuggingFace model or our custom MoE model
            if os.path.exists(os.path.join(self.model_path, "model.safetensors")):
                # Standard HuggingFace model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    use_flash_attention_2=True
                )
            else:
                # Our custom MoE model
                self.model = MoEModel.from_pretrained(self.model_path)
                self.model = self.model.to(self.device)
                if torch.cuda.is_available():
                    self.model = self.model.bfloat16()
        else:
            logger.info("Loading standard model")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
        # Set eval mode
        self.model.eval()
        
        logger.info("Model and tokenizer loaded successfully")
        
    def _build_prompt(
        self, 
        user_id: str, 
        conversation_id: str, 
        query: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Build a prompt for the model using memory context.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            query: User query
            system_prompt: Optional system prompt to override default
            
        Returns:
            str: Full prompt for the model
        """
        # Get context from memory manager
        context = self.memory_manager.build_context(
            user_id=user_id,
            conversation_id=conversation_id,
            query=query
        )
        
        # Default system prompt if not provided
        if system_prompt is None:
            system_prompt = """You are an AI assistant that helps users with their questions and tasks.
You are helpful, respectful, and insightful.
You carefully provide accurate information and acknowledge when you're unsure.
"""
        
        # Build the full prompt
        full_prompt = f"{system_prompt}\n\n{context}\n\nASSISTANT:"
        
        return full_prompt
    
    def generate_response(
        self, 
        user_id: str, 
        conversation_id: str, 
        query: str,
        system_prompt: Optional[str] = None,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response to the user query.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            query: User query
            system_prompt: Optional system prompt to override default
            generation_params: Optional parameters to override default generation params
            
        Returns:
            str: Generated response
        """
        # Add user message to memory
        self.memory_manager.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="user",
            content=query
        )
        
        # Build prompt
        prompt = self._build_prompt(
            user_id=user_id,
            conversation_id=conversation_id,
            query=query,
            system_prompt=system_prompt
        )
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Get generation parameters
        params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        
        # Override with provided params if any
        if generation_params:
            params.update(generation_params)
            
        # Add stopping criteria
        stop_token_ids = [self.tokenizer.eos_token_id]
        if self.tokenizer.pad_token_id is not None:
            stop_token_ids.append(self.tokenizer.pad_token_id)
            
        stopping_criteria = StoppingCriteriaList([
            StopOnTokens(stop_token_ids=stop_token_ids)
        ])
        params["stopping_criteria"] = stopping_criteria
        
        # Generate response
        start_time = time.time()
        logger.info(f"Generating response for user {user_id}, conversation {conversation_id}")
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **params
            )
            
        # Decode response (remove prompt)
        prompt_length = inputs.input_ids.shape[1]
        response_ids = output[0][prompt_length:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        generation_time = time.time() - start_time
        logger.info(f"Response generated in {generation_time:.2f} seconds")
        
        # Add assistant message to memory
        self.memory_manager.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="assistant",
            content=response
        )
        
        return response
    
    def stream_response(
        self, 
        user_id: str, 
        conversation_id: str, 
        query: str,
        system_prompt: Optional[str] = None,
        generation_params: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Stream a response to the user query, sending partial results through callback.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            query: User query
            system_prompt: Optional system prompt to override default
            generation_params: Optional parameters to override default generation params
            callback: Callback function to receive streamed tokens
            
        Returns:
            str: Complete generated response
        """
        # Add user message to memory
        self.memory_manager.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="user",
            content=query
        )
        
        # Build prompt
        prompt = self._build_prompt(
            user_id=user_id,
            conversation_id=conversation_id,
            query=query,
            system_prompt=system_prompt
        )
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]
        
        # Get generation parameters
        params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        
        # Override with provided params if any
        if generation_params:
            params.update(generation_params)
        
        # Stream tokens
        logger.info(f"Streaming response for user {user_id}, conversation {conversation_id}")
        generated_text = ""
        
        with torch.no_grad():
            # Initialize past_key_values for more efficient generation
            past = None
            
            # Initialize input_ids
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            
            for _ in range(params["max_new_tokens"]):
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True
                )
                
                # Get logits for next token prediction
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply repetition penalty
                if params["repetition_penalty"] > 1.0:
                    for seq_idx in range(input_ids.shape[0]):
                        for token_idx in set(input_ids[seq_idx].tolist()):
                            next_token_logits[seq_idx, token_idx] /= params["repetition_penalty"]
                
                # Apply temperature
                next_token_logits = next_token_logits / params["temperature"]
                
                # Apply top-k filtering
                if params["top_k"] > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, params["top_k"])[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")
                
                # Apply top-p (nucleus) filtering
                if params["top_p"] < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > params["top_p"]
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter back to the original indices
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1, 
                        index=sorted_indices, 
                        src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float("-inf")
                
                # Sample next token
                if params["do_sample"]:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Decode token
                token_text = self.tokenizer.decode([next_token[0].item()], skip_special_tokens=True)
                generated_text += token_text
                
                # Call callback with new token if provided
                if callback:
                    callback(token_text)
                
                # Check if we've hit the EOS token
                if next_token[0].item() in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                    break
                
                # Update input_ids, attention_mask, and past for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([
                    attention_mask, 
                    attention_mask.new_ones((attention_mask.shape[0], 1))
                ], dim=-1)
                past = outputs.past_key_values
        
        # Add assistant message to memory
        self.memory_manager.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="assistant",
            content=generated_text
        )
        
        return generated_text
    
    def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history from memory.
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages to return
            
        Returns:
            List of conversation messages
        """
        return self.memory_manager.get_conversation_history(
            conversation_id=conversation_id,
            limit=limit
        )
    
    def analyze_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Analyze a conversation to extract insights and metadata.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Dict with analysis results
        """
        # Get conversation history
        conversation = self.memory_manager.short_term.get_conversation(conversation_id)
        if not conversation:
            return {"error": "Conversation not found"}
            
        # Get messages
        messages = conversation.get_messages()
        if not messages:
            return {"error": "No messages in conversation"}
            
        # Basic metrics
        user_messages = [m for m in messages if m["role"] == "user"]
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        
        # Calculate response times
        response_times = []
        for i in range(1, len(messages)):
            if messages[i-1]["role"] == "user" and messages[i]["role"] == "assistant":
                response_time = messages[i]["timestamp"] - messages[i-1]["timestamp"]
                response_times.append(response_time)
                
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Calculate message lengths
        user_message_lengths = [len(m["content"]) for m in user_messages]
        assistant_message_lengths = [len(m["content"]) for m in assistant_messages]
        
        avg_user_message_length = sum(user_message_lengths) / len(user_message_lengths) if user_message_lengths else 0
        avg_assistant_message_length = sum(assistant_message_lengths) / len(assistant_message_lengths) if assistant_message_lengths else 0
        
        # Time span of conversation
        time_span = messages[-1]["timestamp"] - messages[0]["timestamp"] if len(messages) > 1 else 0
        
        # Return analysis
        return {
            "conversation_id": conversation_id,
            "user_id": conversation.user_id,
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "avg_response_time_seconds": avg_response_time,
            "avg_user_message_length": avg_user_message_length,
            "avg_assistant_message_length": avg_assistant_message_length,
            "conversation_duration_seconds": time_span,
            "first_message_timestamp": messages[0]["timestamp"] if messages else None,
            "last_message_timestamp": messages[-1]["timestamp"] if messages else None
        }
    
    def save_state(self, path: str):
        """
        Save the state of the inference engine.
        
        Args:
            path: Path to save the state
        """
        state_dir = Path(path)
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # Save memory manager state
        self.memory_manager.save_all(base_directory=str(state_dir / "memory"))
        
        # Save engine configuration
        config = {
            "model_path": self.model_path,
            "use_moe": self.use_moe,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty
        }
        
        with open(state_dir / "engine_config.json", "w") as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Inference engine state saved to {path}")
    
    def load_state(self, path: str):
        """
        Load the state of the inference engine.
        
        Args:
            path: Path to load the state from
        """
        state_dir = Path(path)
        
        if not state_dir.exists():
            logger.error(f"State directory {path} does not exist")
            return
            
        # Load engine configuration
        config_path = state_dir / "engine_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
                
            # Update configuration
            self.max_new_tokens = config.get("max_new_tokens", self.max_new_tokens)
            self.temperature = config.get("temperature", self.temperature)
            self.top_p = config.get("top_p", self.top_p)
            self.top_k = config.get("top_k", self.top_k)
            self.repetition_penalty = config.get("repetition_penalty", self.repetition_penalty)
            
        # Load memory manager state
        memory_dir = state_dir / "memory"
        if memory_dir.exists():
            self.memory_manager.load_all(base_directory=str(memory_dir))
            
        logger.info(f"Inference engine state loaded from {path}")
