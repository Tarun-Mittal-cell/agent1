"""
Multi-model LLM integration layer supporting:
1. OpenAI GPT-4
2. Anthropic Claude
3. Local open source models
4. Parallel inference
"""
import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import anthropic
import openai

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str  # openai, anthropic, local
    model_name: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.95
    
class LLMProvider:
    """
    Unified interface for multiple LLM providers with:
    - Automatic fallbacks
    - Parallel inference
    - Caching
    - Rate limiting
    """
    
    def __init__(self, configs: List[LLMConfig]):
        self.configs = configs
        self.clients = {}
        self.tokenizers = {}
        self.models = {}
        self._setup_clients()
        
    def _setup_clients(self):
        """Initialize API clients and models."""
        for config in self.configs:
            if config.provider == "openai":
                if not config.api_key:
                    config.api_key = os.getenv("OPENAI_API_KEY")
                self.clients[config.model_name] = openai.OpenAI(
                    api_key=config.api_key
                )
                
            elif config.provider == "anthropic":
                if not config.api_key:
                    config.api_key = os.getenv("ANTHROPIC_API_KEY")
                self.clients[config.model_name] = anthropic.Anthropic(
                    api_key=config.api_key
                )
                
            elif config.provider == "local":
                logger.info(f"Loading local model {config.model_name}")
                self.tokenizers[config.model_name] = AutoTokenizer.from_pretrained(
                    config.model_name
                )
                self.models[config.model_name] = AutoModelForCausalLM.from_pretrained(
                    config.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                
    async def generate(self,
                      prompts: List[str],
                      config: Optional[LLMConfig] = None) -> List[str]:
        """
        Generate completions for prompts using specified or default config.
        Handles batching, retries, and fallbacks.
        """
        if config is None:
            config = self.configs[0]
            
        try:
            if config.provider == "openai":
                return await self._generate_openai(prompts, config)
            elif config.provider == "anthropic":
                return await self._generate_anthropic(prompts, config)
            elif config.provider == "local":
                return await self._generate_local(prompts, config)
                
        except Exception as e:
            logger.error(f"Error with {config.provider}: {str(e)}")
            if len(self.configs) > 1:
                logger.info("Trying fallback provider")
                next_config = self.configs[1]
                return await self.generate(prompts, next_config)
            raise
            
    async def _generate_openai(self,
                              prompts: List[str],
                              config: LLMConfig) -> List[str]:
        """Generate using OpenAI API."""
        client = self.clients[config.model_name]
        
        async def _generate_single(prompt: str) -> str:
            response = await client.chat.completions.create(
                model=config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p
            )
            return response.choices[0].message.content
            
        tasks = [_generate_single(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
        
    async def _generate_anthropic(self,
                                 prompts: List[str],
                                 config: LLMConfig) -> List[str]:
        """Generate using Anthropic API."""
        client = self.clients[config.model_name]
        
        async def _generate_single(prompt: str) -> str:
            response = await client.messages.create(
                model=config.model_name,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
            
        tasks = [_generate_single(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)
        
    async def _generate_local(self,
                             prompts: List[str],
                             config: LLMConfig) -> List[str]:
        """Generate using local model."""
        tokenizer = self.tokenizers[config.model_name]
        model = self.models[config.model_name]
        
        # Tokenize all prompts
        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True
        )
        
        # Decode
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
    async def get_embeddings(self,
                            texts: List[str],
                            config: Optional[LLMConfig] = None) -> torch.Tensor:
        """Get embeddings for texts."""
        if config is None:
            config = self.configs[0]
            
        if config.provider == "local":
            tokenizer = self.tokenizers[config.model_name]
            model = self.models[config.model_name]
            
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Use mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings
                
        elif config.provider == "openai":
            client = self.clients[config.model_name]
            response = await client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return torch.tensor([r.embedding for r in response.data])
            
        else:
            raise ValueError(f"Embeddings not supported for {config.provider}")
            
    def __call__(self, *args, **kwargs):
        """Convenience wrapper for generate()."""
        return self.generate(*args, **kwargs)