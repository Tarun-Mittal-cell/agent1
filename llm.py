"""
LLM integration for code generation and analysis.
"""
import os
from typing import Dict, List, Optional
import anthropic
import openai
from dotenv import load_dotenv

load_dotenv()

class LLMProvider:
    """Manages interactions with language models."""
    
    def __init__(self):
        self.anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.openai = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def generate_code(self, prompt: str, context: Dict, 
                     language: str, max_tokens: int = 4000) -> str:
        """Generate code using Claude for maximum accuracy."""
        system_prompt = self._build_system_prompt(language, context)
        
        response = self.anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=max_tokens,
            temperature=0.1,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
        
    def analyze_code(self, code: str, language: str) -> Dict:
        """Analyze code for potential issues and improvements."""
        prompt = f"""Analyze this {language} code for:
1. Potential bugs
2. Security issues
3. Performance optimizations
4. Best practice violations

Code:
{code}

Provide analysis in JSON format with these categories."""
        
        response = self.openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
        
    def fix_issues(self, code: str, issues: List[str], 
                  language: str) -> str:
        """Fix identified issues in the code."""
        prompt = f"""Fix these issues in the {language} code:
Issues:
{chr(10).join(f'- {issue}' for issue in issues)}

Code:
{code}

Return only the fixed code without explanations."""
        
        response = self.anthropic.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=len(code) * 2,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
        
    def _build_system_prompt(self, language: str, context: Dict) -> str:
        """Build system prompt with language-specific context."""
        return f"""You are an expert {language} developer. Generate production-quality code following these principles:
1. Use modern best practices and idioms
2. Include comprehensive error handling
3. Follow security best practices
4. Write maintainable, well-documented code
5. Include appropriate tests
6. Consider performance implications

Additional context:
{context}"""