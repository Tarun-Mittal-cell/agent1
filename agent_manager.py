"""
Agent manager for coordinating multiple specialized agents.
"""
import logging
from pathlib import Path
from typing import Optional

from llm_integration import LLMProvider
from knowledge_base import KnowledgeBase
from core import ProjectSpec, CodeArtifact
from agents.code_generation import CodeGenerationAgent
from agents.debug import DebugAgent
from agents.security import SecurityAgent

logger = logging.getLogger(__name__)

class AgentManager:
    """Orchestrates multiple specialized agents."""
    
    def __init__(self,
                llm_provider: LLMProvider,
                knowledge_base: KnowledgeBase,
                enable_self_improvement: bool = False,
                output_dir: Optional[Path] = None):
        self.llm = llm_provider
        self.kb = knowledge_base
        self.enable_self_improvement = enable_self_improvement
        self.output_dir = output_dir or Path("output")
        
        # Initialize agents
        self.code_gen = CodeGenerationAgent(self.llm, self.kb)
        self.debugger = DebugAgent(self.llm)
        self.security = SecurityAgent(self.llm)
        
    def generate_application(self, spec: ProjectSpec) -> CodeArtifact:
        """Generate a complete application from specification."""
        logger.info(f"Generating {spec.type} application: {spec.name}")
        
        # Initialize artifact
        artifact = CodeArtifact(spec=spec)
        
        try:
            # Generate core application code
            self._generate_core(spec, artifact)
            
            # Add tests
            self._generate_tests(spec, artifact)
            
            # Add deployment configuration
            self._generate_deployment(spec, artifact)
            
            # Run security checks
            self._security_scan(artifact)
            
            # Optimize code
            if self.enable_self_improvement:
                self._optimize_code(artifact)
            
            return artifact
            
        except Exception as e:
            logger.error(f"Error generating application: {str(e)}")
            raise
            
    def _generate_core(self, spec: ProjectSpec, artifact: CodeArtifact):
        """Generate core application code."""
        logger.info("Generating core application code...")
        
        # Frontend
        if "frontend" in spec.stack:
            self._generate_frontend(spec, artifact)
            
        # Backend
        if "backend" in spec.stack:
            self._generate_backend(spec, artifact)
            
    def _generate_frontend(self, spec: ProjectSpec, artifact: CodeArtifact):
        """Generate frontend code."""
        logger.info("Generating frontend code...")
        
        # Get frontend stack
        frontend = spec.stack["frontend"]
        framework = frontend["framework"]
        
        if framework == "next.js":
            # Generate Next.js files
            for page in spec.pages:
                self._generate_next_page(page, artifact)
                
            # Generate components
            self._generate_components(spec, artifact)
            
            # Add configuration files
            self._add_next_config(spec, artifact)
            
    def _generate_backend(self, spec: ProjectSpec, artifact: CodeArtifact):
        """Generate backend code."""
        logger.info("Generating backend code...")
        
        # Get backend stack
        backend = spec.stack["backend"]
        framework = backend["framework"]
        
        if framework == "fastapi":
            # Generate FastAPI files
            self._generate_fastapi_app(spec, artifact)
            
            # Generate database models
            self._generate_models(spec, artifact)
            
            # Add API routes
            self._generate_routes(spec, artifact)
            
    def _generate_next_page(self, page: dict, artifact: CodeArtifact):
        """Generate a Next.js page."""
        name = page["name"].lower()
        components = page["components"]
        
        # Generate page file
        page_path = Path(f"frontend/pages/{name}.tsx")
        page_content = self.code_gen.generate_next_page(name, components)
        artifact.add_file(page_path, page_content)
        
    def _generate_components(self, spec: ProjectSpec, artifact: CodeArtifact):
        """Generate React components."""
        components = []
        for page in spec.pages:
            components.extend(page["components"])
            
        for component in components:
            comp_path = Path(f"frontend/components/{component}.tsx")
            comp_content = self.code_gen.generate_component(component, spec.design)
            artifact.add_file(comp_path, comp_content)
            
    def _add_next_config(self, spec: ProjectSpec, artifact: CodeArtifact):
        """Add Next.js configuration files."""
        # next.config.js
        config_path = Path("frontend/next.config.js")
        config_content = self.code_gen.generate_next_config()
        artifact.add_file(config_path, config_content)
        
        # tailwind.config.js
        tailwind_path = Path("frontend/tailwind.config.js")
        tailwind_content = self.code_gen.generate_tailwind_config(spec.design)
        artifact.add_file(tailwind_path, tailwind_content)
        
    def _generate_fastapi_app(self, spec: ProjectSpec, artifact: CodeArtifact):
        """Generate FastAPI application."""
        # Main app file
        main_path = Path("backend/app/main.py")
        main_content = self.code_gen.generate_fastapi_main()
        artifact.add_file(main_path, main_content)
        
        # Database configuration
        db_path = Path("backend/app/db/database.py")
        db_content = self.code_gen.generate_db_config(spec.stack["backend"]["database"])
        artifact.add_file(db_path, db_content)
        
    def _generate_models(self, spec: ProjectSpec, artifact: CodeArtifact):
        """Generate database models."""
        models_path = Path("backend/app/models")
        models = self.code_gen.generate_models(spec)
        
        for name, content in models.items():
            model_path = models_path / f"{name}.py"
            artifact.add_file(model_path, content)
            
    def _generate_routes(self, spec: ProjectSpec, artifact: CodeArtifact):
        """Generate API routes."""
        routes_path = Path("backend/app/api/v1")
        routes = self.code_gen.generate_routes(spec)
        
        for name, content in routes.items():
            route_path = routes_path / f"{name}.py"
            artifact.add_file(route_path, content)
            
    def _generate_tests(self, spec: ProjectSpec, artifact: CodeArtifact):
        """Generate tests."""
        logger.info("Generating tests...")
        
        # Frontend tests
        if "frontend" in spec.stack:
            self._generate_frontend_tests(spec, artifact)
            
        # Backend tests
        if "backend" in spec.stack:
            self._generate_backend_tests(spec, artifact)
            
    def _generate_deployment(self, spec: ProjectSpec, artifact: CodeArtifact):
        """Generate deployment configuration."""
        logger.info("Generating deployment configuration...")
        
        # Docker files
        docker_path = Path("deployment/Dockerfile")
        docker_content = self.code_gen.generate_dockerfile(spec)
        artifact.add_file(docker_path, docker_content)
        
        # Docker Compose
        compose_path = Path("deployment/docker-compose.yml")
        compose_content = self.code_gen.generate_docker_compose(spec)
        artifact.add_file(compose_path, compose_content)
        
        # Kubernetes manifests
        k8s_path = Path("deployment/k8s")
        k8s_files = self.code_gen.generate_kubernetes(spec)
        
        for name, content in k8s_files.items():
            manifest_path = k8s_path / f"{name}.yaml"
            artifact.add_file(manifest_path, content)
            
    def _security_scan(self, artifact: CodeArtifact):
        """Run security scans on generated code."""
        logger.info("Running security scans...")
        
        for path, content in artifact.code_files.items():
            secure_content = self.security.audit_and_fix(content)
            artifact.code_files[path] = secure_content
            
    def _optimize_code(self, artifact: CodeArtifact):
        """Optimize generated code."""
        logger.info("Optimizing code...")
        
        for path, content in artifact.code_files.items():
            optimized = self.debugger.optimize_code(content)
            artifact.code_files[path] = optimized