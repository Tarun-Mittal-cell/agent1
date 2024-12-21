"""
Security agent for code scanning and vulnerability detection.
"""
import logging
from typing import Dict, List, Any

from ..llm_integration import LLMProvider

logger = logging.getLogger(__name__)

class SecurityAgent:
    """Scans code for security vulnerabilities."""
    
    def __init__(self, llm: LLMProvider):
        self.llm = llm
        
    def audit_and_fix(self, code: str) -> str:
        """Scan code for vulnerabilities and fix them."""
        logger.info("Running security audit...")
        
        # Get security analysis from LLM
        analysis = self.llm.generate(
            prompt=f"Analyze this code for security vulnerabilities:\n\n{code}",
            provider="openai"
        )
        
        # If vulnerabilities found, fix them
        if "vulnerability" in analysis.lower():
            logger.info("Fixing security vulnerabilities...")
            fixed_code = self.llm.generate(
                prompt=f"Fix these security issues:\n\nCode:\n{code}\n\nIssues:\n{analysis}",
                provider="openai"
            )
            return fixed_code
            
        return code