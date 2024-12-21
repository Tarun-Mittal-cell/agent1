"""
Interactive code refinement system.
"""
from typing import Dict, List, Optional
from pathlib import Path
import difflib

from .core import CodeArtifact
from .llm import LLMProvider

class InteractiveRefiner:
    """Handles interactive code refinement with users."""
    
    def __init__(self):
        self.llm = LLMProvider()
        
    def refine_code(self, artifact: CodeArtifact, feedback: str) -> CodeArtifact:
        """Refine code based on user feedback."""
        # Parse feedback to understand requested changes
        changes = self._analyze_feedback(feedback)
        
        # Apply changes to relevant files
        updated_files = {}
        for file_path, content in artifact.code.items():
            if self._should_modify_file(file_path, changes):
                updated_content = self._apply_changes(content, changes)
                updated_files[file_path] = updated_content
            else:
                updated_files[file_path] = content
                
        # Update artifact with new code
        artifact.code = updated_files
        return artifact
        
    def show_diff(self, original: str, modified: str) -> str:
        """Show diff between original and modified code."""
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True)
        )
        return "".join(diff)
        
    def _analyze_feedback(self, feedback: str) -> Dict:
        """Analyze user feedback to determine required changes."""
        prompt = f"""Analyze this feedback and break it down into specific code changes:
{feedback}

Return a JSON object with:
1. Files to modify
2. Type of changes (add/modify/delete)
3. Specific changes to make
4. Priority of changes"""
        
        return self.llm.analyze_code(prompt, "json")
        
    def _should_modify_file(self, file_path: Path, changes: Dict) -> bool:
        """Determine if file should be modified based on changes."""
        if "files" not in changes:
            return False
            
        # Check if file matches any patterns in changes
        return any(
            pattern in str(file_path)
            for pattern in changes["files"]
        )
        
    def _apply_changes(self, content: str, changes: Dict) -> str:
        """Apply specified changes to code content."""
        if "changes" not in changes:
            return content
            
        # Generate prompt for modifications
        prompt = f"""Modify this code according to these changes:
{changes['changes']}

Original code:
{content}

Return only the modified code without explanations."""
        
        return self.llm.generate_code(
            prompt=prompt,
            context=changes,
            language=self._detect_language(content)
        )
        
    def _detect_language(self, code: str) -> str:
        """Detect programming language from code content."""
        # Use LLM to detect language
        prompt = f"""Detect the programming language of this code:
{code[:1000]}  # First 1000 chars

Return only the language name in lowercase."""
        
        return self.llm.generate_code(
            prompt=prompt,
            context={},
            language="text"
        ).strip().lower()