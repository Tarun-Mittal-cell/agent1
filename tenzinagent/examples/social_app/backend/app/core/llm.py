"""
Multi-model LLM integration with fallbacks
"""
import os
from typing import List, Optional
import asyncio
import logging
from dataclasses import dataclass

import openai
import anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str  # openai, anthropic, huggingface
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
    
    def __init__(self):
        # Initialize API keys
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.hf_key = os.getenv("HUGGINGFACE_API_KEY")
        
        # Initialize clients
        self.clients = {}
        self._setup_clients()
        
    def _setup_clients(self):
        """Initialize API clients."""
        # OpenAI
        if self.openai_key:
            self.clients["openai"] = openai.OpenAI(
                api_key=self.openai_key
            )
            
        # Anthropic
        if self.anthropic_key:
            self.clients["anthropic"] = anthropic.Anthropic(
                api_key=self.anthropic_key
            )
            
        # Hugging Face
        if self.hf_key:
            # Load local model
            self.clients["huggingface"] = {
                "tokenizer": AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1"),
                "model": AutoModelForCausalLM.from_pretrained(
                    "mistralai/Mistral-7B-v0.1",
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            }
            
    async def generate(self,
                      prompt: str,
                      provider: str = "openai",
                      **kwargs) -> str:
        """Generate text using specified provider with fallbacks."""
        try:
            if provider == "openai":
                return await self._generate_openai(prompt, **kwargs)
            elif provider == "anthropic":
                return await self._generate_anthropic(prompt, **kwargs)
            elif provider == "huggingface":
                return await self._generate_huggingface(prompt, **kwargs)
                
        except Exception as e:
            logger.error(f"Error with {provider}: {str(e)}")
            # Try fallback providers
            for fallback in ["anthropic", "openai", "huggingface"]:
                if fallback != provider and fallback in self.clients:
                    try:
                        return await self.generate(prompt, provider=fallback, **kwargs)
                    except:
                        continue
                        
            raise Exception("All providers failed")
            
    async def _generate_openai(self, prompt: str, **kwargs) -> str:
        """Generate using OpenAI."""
        client = self.clients["openai"]
        
        response = await client.chat.completions.create(
            model=kwargs.get("model", "gpt-4"),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2000),
            top_p=kwargs.get("top_p", 0.95)
        )
        
        return response.choices[0].message.content
        
    async def _generate_anthropic(self, prompt: str, **kwargs) -> str:
        """Generate using Anthropic."""
        client = self.clients["anthropic"]
        
        response = await client.messages.create(
            model=kwargs.get("model", "claude-2"),
            max_tokens=kwargs.get("max_tokens", 2000),
            temperature=kwargs.get("temperature", 0.7),
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
        
    async def _generate_huggingface(self, prompt: str, **kwargs) -> str:
        """Generate using local Hugging Face model."""
        tokenizer = self.clients["huggingface"]["tokenizer"]
        model = self.clients["huggingface"]["model"]
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=kwargs.get("max_tokens", 2000)
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", 2000),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
            do_sample=True
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    async def get_embeddings(self,
                           texts: List[str],
                           provider: str = "openai") -> torch.Tensor:
        """Get embeddings for texts."""
        try:
            if provider == "openai":
                client = self.clients["openai"]
                response = await client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=texts
                )
                return torch.tensor([r.embedding for r in response.data])
                
            elif provider == "huggingface":
                tokenizer = self.clients["huggingface"]["tokenizer"]
                model = self.clients["huggingface"]["model"]
                
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
                    
        except Exception as e:
            logger.error(f"Error getting embeddings with {provider}: {str(e)}")
            # Try fallback
            if provider == "openai":
                return await self.get_embeddings(texts, provider="huggingface")
            else:
                return await self.get_embeddings(texts, provider="openai")