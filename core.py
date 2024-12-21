"""
Core data structures and utilities for TenzinAgent.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import shutil

@dataclass
class ProjectSpec:
    """Project specification."""
    name: str
    type: str
    stack: Dict[str, Any]
    pages: List[Dict[str, Any]]
    design: Dict[str, Any]
    language: str = "python"
    framework: Optional[str] = None
    database: Optional[str] = None
    requirements: List[str] = field(default_factory=list)
    cross_platform: bool = True
    additional_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Set defaults if not provided
        if "frontend" not in self.stack:
            self.stack["frontend"] = {
                "framework": "next.js",
                "styling": "tailwindcss"
            }
        if "backend" not in self.stack:
            self.stack["backend"] = {
                "framework": "fastapi",
                "database": "postgresql"
            }

@dataclass
class TestResult:
    """Results from running tests on generated code."""
    passed: bool
    total_tests: int
    failed_tests: List[str]
    coverage: float
    execution_time: float

@dataclass
class DeploymentConfig:
    """Configuration for deploying the generated application."""
    platform: str
    dockerfile: Optional[str]
    requirements: str
    env_vars: Dict[str, str]
    deploy_scripts: Dict[str, str]

@dataclass
class CodeArtifact:
    """Container for generated code and associated artifacts."""
    code_files: Dict[Path, str] = field(default_factory=dict)
    tests: Dict[Path, str] = field(default_factory=dict)
    deployment: DeploymentConfig = None
    spec: ProjectSpec = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_file(self, path: Path, content: str, file_type: str = "code"):
        """Add a file to the artifact."""
        if file_type == "code":
            self.code_files[path] = content
        elif file_type == "test":
            self.tests[path] = content
        else:
            raise ValueError(f"Unknown file type: {file_type}")

    def save(self, output_dir: Path):
        """Save all artifact files to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save code files
        for path, content in self.code_files.items():
            file_path = output_dir / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)

        # Save test files
        test_dir = output_dir / "tests"
        test_dir.mkdir(parents=True, exist_ok=True)
        for path, content in self.tests.items():
            file_path = test_dir / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)

        # Save deployment files
        if self.deployment:
            deploy_dir = output_dir / "deployment"
            deploy_dir.mkdir(parents=True, exist_ok=True)
            
            if self.deployment.dockerfile:
                with open(deploy_dir / "Dockerfile", 'w') as f:
                    f.write(self.deployment.dockerfile)
            
            with open(deploy_dir / "requirements.txt", 'w') as f:
                f.write(self.deployment.requirements)
            
            for name, script in self.deployment.deploy_scripts.items():
                with open(deploy_dir / name, 'w') as f:
                    f.write(script)

        # Save metadata
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(self.metadata, f, indent=2)

class CodeGenerationError(Exception):
    """Custom exception for code generation errors."""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass