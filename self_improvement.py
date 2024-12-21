"""
Self-improvement system that allows the agent to analyze and optimize its own code.
"""
from typing import Dict, List, Optional, Type
from pathlib import Path
import inspect
import ast
import importlib
import logging
from dataclasses import dataclass

from .llm import LLMProvider
from .core import CodeArtifact
from .agents.base import BaseAgent

@dataclass
class ImprovementSuggestion:
    """Represents a suggested improvement to the agent's code."""
    file_path: Path
    description: str
    severity: str  # high/medium/low
    code_changes: Dict[str, str]  # old -> new code
    impact: str
    confidence: float

class SelfImprovementAgent:
    """Enables the agent to analyze and improve its own codebase."""
    
    def __init__(self):
        self.llm = LLMProvider()
        self.logger = logging.getLogger(__name__)
        
    def analyze_self(self) -> List[ImprovementSuggestion]:
        """Analyze agent's codebase for potential improvements."""
        suggestions = []
        
        # Get all Python files in the agent's codebase
        agent_files = self._get_agent_files()
        
        for file_path in agent_files:
            # Analyze file for improvements
            file_suggestions = self._analyze_file(file_path)
            suggestions.extend(file_suggestions)
            
        return suggestions
        
    def fix_self(self, suggestions: List[ImprovementSuggestion]) -> bool:
        """Apply suggested improvements to agent's codebase."""
        try:
            for suggestion in suggestions:
                if suggestion.confidence >= 0.8:  # Only apply high-confidence changes
                    self._apply_improvement(suggestion)
                    
            # Reload modified modules
            self._reload_modules()
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying improvements: {str(e)}")
            return False
            
    def generate_agent(self, task_type: str) -> Type[BaseAgent]:
        """Generate a new specialized agent for a specific task."""
        # Get requirements for the new agent
        agent_spec = self._generate_agent_spec(task_type)
        
        # Generate agent code
        agent_code = self._generate_agent_code(agent_spec)
        
        # Create new agent module
        module_path = self._create_agent_module(task_type, agent_code)
        
        # Load and return new agent class
        return self._load_agent_class(module_path)
        
    def _get_agent_files(self) -> List[Path]:
        """Get all Python files in the agent's codebase."""
        base_path = Path(__file__).parent
        return list(base_path.rglob("*.py"))
        
    def _analyze_file(self, file_path: Path) -> List[ImprovementSuggestion]:
        """Analyze a single file for potential improvements."""
        with open(file_path) as f:
            content = f.read()
            
        # Use LLM to analyze code
        analysis = self.llm.analyze_code(
            content,
            "python",
            context={"purpose": "self_improvement"}
        )
        
        suggestions = []
        if "improvements" in analysis:
            for imp in analysis["improvements"]:
                suggestions.append(ImprovementSuggestion(
                    file_path=file_path,
                    description=imp["description"],
                    severity=imp["severity"],
                    code_changes=imp["changes"],
                    impact=imp["impact"],
                    confidence=imp["confidence"]
                ))
                
        return suggestions
        
    def _apply_improvement(self, suggestion: ImprovementSuggestion):
        """Apply a single improvement to the codebase."""
        with open(suggestion.file_path) as f:
            content = f.read()
            
        # Apply each code change
        for old_code, new_code in suggestion.code_changes.items():
            content = content.replace(old_code, new_code)
            
        # Validate modified code
        try:
            ast.parse(content)  # Ensure valid Python syntax
        except SyntaxError:
            raise ValueError("Invalid syntax in modified code")
            
        # Write changes
        with open(suggestion.file_path, "w") as f:
            f.write(content)
            
    def _reload_modules(self):
        """Reload all modified modules."""
        for module in list(sys.modules.values()):
            if hasattr(module, "__file__") and module.__file__:
                if Path(module.__file__).is_relative_to(Path(__file__).parent):
                    importlib.reload(module)
                    
    def _generate_agent_spec(self, task_type: str) -> Dict:
        """Generate specifications for a new agent."""
        prompt = f"""Generate specifications for a new agent that handles {task_type} tasks.
Include:
1. Required capabilities
2. Core methods
3. Dependencies
4. Integration points
Return as JSON object."""
        
        return self.llm.generate_code(
            prompt=prompt,
            context={"type": "agent_spec"},
            language="json"
        )
        
    def _generate_agent_code(self, spec: Dict) -> str:
        """Generate code for a new agent based on specifications."""
        prompt = f"""Generate a new agent class with these specifications:
{spec}

Include:
1. Class definition
2. All required methods
3. Documentation
4. Type hints
5. Error handling

Return only the Python code."""
        
        return self.llm.generate_code(
            prompt=prompt,
            context=spec,
            language="python"
        )
        
    def _create_agent_module(self, task_type: str, code: str) -> Path:
        """Create a new module file for the agent."""
        module_name = f"{task_type.lower()}_agent.py"
        module_path = Path(__file__).parent / "agents" / module_name
        
        with open(module_path, "w") as f:
            f.write(code)
            
        return module_path
        
    def _load_agent_class(self, module_path: Path) -> Type[BaseAgent]:
        """Load and return the agent class from a module."""
        spec = importlib.util.spec_from_file_location(
            module_path.stem,
            module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find and return the agent class
        for item in dir(module):
            obj = getattr(module, item)
            if (isinstance(obj, type) and 
                issubclass(obj, BaseAgent) and 
                obj != BaseAgent):
                return obj
                
        raise ValueError("No agent class found in module")