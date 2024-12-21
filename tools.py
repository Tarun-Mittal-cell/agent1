"""
Advanced tooling support with:
1. Language-specific tool integration
2. Cross-platform compatibility
3. Parallel execution
4. Error handling and reporting
5. Auto-configuration
"""
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import json
import tempfile
import shutil

logger = logging.getLogger(__name__)

@dataclass
class ToolResult:
    """Result from running a tool."""
    success: bool
    output: str
    errors: List[str]
    fixes: Optional[Dict[str, str]] = None
    metrics: Optional[Dict[str, float]] = None

class ToolManager:
    """
    Manages external tools for:
    - Linting
    - Formatting
    - Testing
    - Building
    - Security scanning
    - Profiling
    """
    
    def __init__(self, knowledge_base: Any):
        self.kb = knowledge_base
        self.tool_cache = {}
        
    async def run_tool(self,
                      tool_type: str,
                      language: str,
                      files: Dict[Path, str],
                      config: Optional[Dict] = None) -> ToolResult:
        """
        Run a specific type of tool on given files.
        Handles tool selection, execution, and result parsing.
        """
        # Get available tools
        tools = self.kb.get_tool_commands(language, tool_type)
        if not tools:
            return ToolResult(
                success=False,
                output="",
                errors=[f"No {tool_type} tools found for {language}"]
            )
            
        # Select best tool
        tool = tools[0]  # Could implement more sophisticated selection
        
        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Write files
            for file_path, content in files.items():
                full_path = temp_dir / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(content)
                    
            # Run tool
            try:
                if tool_type == "lint":
                    return await self._run_linter(tool, temp_dir, language)
                elif tool_type == "format":
                    return await self._run_formatter(tool, temp_dir, language)
                elif tool_type == "test":
                    return await self._run_tests(tool, temp_dir, language)
                elif tool_type == "build":
                    return await self._run_build(tool, temp_dir, language)
                elif tool_type == "security":
                    return await self._run_security_scan(tool, temp_dir, language)
                elif tool_type == "profile":
                    return await self._run_profiler(tool, temp_dir, language)
                else:
                    return ToolResult(
                        success=False,
                        output="",
                        errors=[f"Unknown tool type: {tool_type}"]
                    )
                    
            except Exception as e:
                logger.error(f"Error running {tool}: {str(e)}")
                return ToolResult(
                    success=False,
                    output="",
                    errors=[str(e)]
                )
                
    async def _run_linter(self,
                         tool: str,
                         directory: Path,
                         language: str) -> ToolResult:
        """Run linter and parse results."""
        if language == "python":
            if tool == "pylint":
                cmd = ["pylint", "--output-format=json", str(directory)]
            elif tool == "flake8":
                cmd = ["flake8", "--format=json", str(directory)]
            else:
                cmd = [tool, str(directory)]
                
        elif language == "javascript":
            if tool == "eslint":
                cmd = ["eslint", "-f", "json", str(directory)]
            else:
                cmd = [tool, str(directory)]
                
        # Run command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        # Parse results
        success = process.returncode == 0
        try:
            output = stdout.decode()
            if output.startswith("{"):
                results = json.loads(output)
            else:
                results = {"raw": output}
        except:
            results = {"raw": output}
            
        errors = []
        if stderr:
            errors.append(stderr.decode())
            
        return ToolResult(
            success=success,
            output=json.dumps(results, indent=2),
            errors=errors
        )
        
    async def _run_formatter(self,
                           tool: str,
                           directory: Path,
                           language: str) -> ToolResult:
        """Run code formatter."""
        if language == "python":
            if tool == "black":
                cmd = ["black", str(directory)]
            elif tool == "yapf":
                cmd = ["yapf", "-r", "-i", str(directory)]
                
        elif language == "javascript":
            if tool == "prettier":
                cmd = ["prettier", "--write", str(directory)]
                
        # Run command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        # Get formatted files
        fixes = {}
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                with open(file_path) as f:
                    fixes[file_path.relative_to(directory)] = f.read()
                    
        return ToolResult(
            success=process.returncode == 0,
            output=stdout.decode(),
            errors=[stderr.decode()] if stderr else [],
            fixes=fixes
        )
        
    async def _run_tests(self,
                        tool: str,
                        directory: Path,
                        language: str) -> ToolResult:
        """Run tests and collect results."""
        if language == "python":
            if tool == "pytest":
                cmd = ["pytest", "--json-report", str(directory)]
            else:
                cmd = [tool, str(directory)]
                
        elif language == "javascript":
            if tool == "jest":
                cmd = ["jest", "--json", str(directory)]
                
        # Run command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        # Parse results
        try:
            results = json.loads(stdout)
        except:
            results = {"raw": stdout.decode()}
            
        # Calculate metrics
        metrics = {
            "total_tests": results.get("total", 0),
            "passed_tests": results.get("passed", 0),
            "failed_tests": results.get("failed", 0),
            "coverage": results.get("coverage", 0.0)
        }
        
        return ToolResult(
            success=process.returncode == 0,
            output=json.dumps(results, indent=2),
            errors=[stderr.decode()] if stderr else [],
            metrics=metrics
        )
        
    async def _run_build(self,
                        tool: str,
                        directory: Path,
                        language: str) -> ToolResult:
        """Run build system."""
        if language == "python":
            if tool == "setuptools":
                cmd = ["python", "setup.py", "build"]
            elif tool == "poetry":
                cmd = ["poetry", "build"]
                
        elif language == "javascript":
            if tool == "npm":
                cmd = ["npm", "run", "build"]
                
        # Run command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(directory),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return ToolResult(
            success=process.returncode == 0,
            output=stdout.decode(),
            errors=[stderr.decode()] if stderr else []
        )
        
    async def _run_security_scan(self,
                               tool: str,
                               directory: Path,
                               language: str) -> ToolResult:
        """Run security scanner."""
        if language == "python":
            if tool == "bandit":
                cmd = ["bandit", "-r", "-f", "json", str(directory)]
            elif tool == "safety":
                cmd = ["safety", "check", "--json"]
                
        elif language == "javascript":
            if tool == "npm-audit":
                cmd = ["npm", "audit", "--json"]
                
        # Run command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        # Parse results
        try:
            results = json.loads(stdout)
        except:
            results = {"raw": stdout.decode()}
            
        return ToolResult(
            success=process.returncode == 0,
            output=json.dumps(results, indent=2),
            errors=[stderr.decode()] if stderr else []
        )
        
    async def _run_profiler(self,
                          tool: str,
                          directory: Path,
                          language: str) -> ToolResult:
        """Run code profiler."""
        if language == "python":
            if tool == "cProfile":
                cmd = ["python", "-m", "cProfile", "-o", "profile.stats", "main.py"]
            elif tool == "yappi":
                cmd = ["yappi", "main.py"]
                
        # Run command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(directory),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        # Parse profile results
        metrics = {}
        if tool == "cProfile":
            import pstats
            stats = pstats.Stats("profile.stats")
            metrics = {
                "total_calls": stats.total_calls,
                "total_time": stats.total_tt
            }
            
        return ToolResult(
            success=process.returncode == 0,
            output=stdout.decode(),
            errors=[stderr.decode()] if stderr else [],
            metrics=metrics
        )