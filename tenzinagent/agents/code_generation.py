"""
Advanced code generation agent that can create full applications across:
- Frontend (React, Next.js, Vue)
- Backend (Node, Python, Go)
- Mobile (React Native, Flutter)
- Games (Unity, Unreal)
- DevOps (Docker, K8s)

Uses:
1. Q* reasoning for architecture decisions
2. MCTS for optimization
3. Neural guidance for code generation
4. Multi-model LLM integration
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..core import ProjectSpec, CodeArtifact
from ..llm_integration import LLMProvider
from ..knowledge_base import KnowledgeBase
from ..q_star import QStarReasoner
from ..mcts import MCTS

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for code generation."""
    architecture: str  # monolith, microservices, serverless
    testing: bool = True
    documentation: bool = True
    deployment: bool = True
    optimization: bool = True
    ui_generation: bool = True

class CodeGenerationAgent:
    """Generates complete, production-ready applications."""
    
    def __init__(self,
                llm_provider: LLMProvider,
                knowledge_base: KnowledgeBase,
                q_star: QStarReasoner,
                mcts: MCTS):
        self.llm = llm_provider
        self.kb = knowledge_base
        self.q_star = q_star
        self.mcts = mcts
        
    async def generate_application(self,
                                spec: ProjectSpec,
                                config: Optional[GenerationConfig] = None) -> CodeArtifact:
        """Generate a complete application from specification."""
        logger.info(f"Generating {spec.type} application: {spec.name}")
        
        if config is None:
            config = GenerationConfig(architecture="monolith")
            
        # Use Q* to break down the problem
        subtasks = self.q_star.decompose_problem(spec)
        
        # Initialize artifact
        artifact = CodeArtifact()
        
        # Generate core application code
        await self._generate_core(spec, config, artifact)
        
        # Generate UI/UX if needed
        if config.ui_generation:
            await self._generate_ui(spec, config, artifact)
            
        # Add tests if enabled
        if config.testing:
            await self._generate_tests(spec, artifact)
            
        # Add documentation if enabled
        if config.documentation:
            await self._generate_docs(spec, artifact)
            
        # Add deployment if enabled
        if config.deployment:
            await self._generate_deployment(spec, artifact)
            
        # Optimize if enabled
        if config.optimization:
            await self._optimize_code(spec, artifact)
            
        return artifact
        
    async def _generate_core(self,
                          spec: ProjectSpec,
                          config: GenerationConfig,
                          artifact: CodeArtifact):
        """Generate core application code."""
        # Get relevant templates and patterns
        templates = self.kb.get_templates(spec.type, spec.language)
        patterns = self.kb.get_patterns(spec.type)
        
        # Generate main application files
        if spec.type == "frontend":
            await self._generate_frontend(spec, templates, patterns, artifact)
        elif spec.type == "backend":
            await self._generate_backend(spec, templates, patterns, artifact)
        elif spec.type == "mobile":
            await self._generate_mobile(spec, templates, patterns, artifact)
        elif spec.type == "game":
            await self._generate_game(spec, templates, patterns, artifact)
            
    async def _generate_frontend(self,
                             spec: ProjectSpec,
                             templates: Dict,
                             patterns: Dict,
                             artifact: CodeArtifact):
        """Generate frontend application code."""
        # Framework-specific generation
        if spec.framework == "react":
            await self._generate_react(spec, templates, patterns, artifact)
        elif spec.framework == "next":
            await self._generate_next(spec, templates, patterns, artifact)
            
    async def _generate_react(self,
                          spec: ProjectSpec,
                          templates: Dict,
                          patterns: Dict,
                          artifact: CodeArtifact):
        """Generate React application."""
        # Add package.json
        package_json = {
            "name": spec.name,
            "version": "0.1.0",
            "private": True,
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "react-router-dom": "^6.8.0"
            }
        }
        artifact.add_file(Path("package.json"), package_json)
        
        # Add main App component
        app_code = templates["component"].replace(
            "Component",
            "App"
        )
        artifact.add_file(Path("src/App.tsx"), app_code)
        
        # Add routing
        router_code = templates["router"]
        artifact.add_file(Path("src/router.tsx"), router_code)
        
        # Add components
        components_path = Path("src/components")
        for component in spec.components:
            code = templates["component"].replace(
                "Component",
                component["name"]
            )
            artifact.add_file(components_path / f"{component['name']}.tsx", code)
            
    async def _generate_ui(self,
                        spec: ProjectSpec,
                        config: GenerationConfig,
                        artifact: CodeArtifact):
        """Generate UI/UX code and assets."""
        # Get design system
        design_system = self.kb.get_design_system()
        
        # Generate theme
        theme_code = self._generate_theme(design_system)
        artifact.add_file(Path("src/theme.ts"), theme_code)
        
        # Generate styled components
        components_path = Path("src/components")
        for component in spec.components:
            styled_code = self._generate_styled_component(
                component["name"],
                design_system
            )
            artifact.add_file(
                components_path / f"{component['name']}.styles.ts",
                styled_code
            )
            
    async def _generate_tests(self,
                          spec: ProjectSpec,
                          artifact: CodeArtifact):
        """Generate comprehensive tests."""
        # Add test setup
        jest_config = {
            "preset": "ts-jest",
            "testEnvironment": "jsdom",
            "setupFilesAfterEnv": ["@testing-library/jest-dom"]
        }
        artifact.add_file(Path("jest.config.js"), jest_config)
        
        # Generate component tests
        test_path = Path("src/__tests__")
        for component in spec.components:
            test_code = self._generate_component_test(component["name"])
            artifact.add_file(
                test_path / f"{component['name']}.test.tsx",
                test_code
            )
            
        # Generate E2E tests
        cypress_config = {
            "e2e": {
                "baseUrl": "http://localhost:3000"
            }
        }
        artifact.add_file(Path("cypress.config.ts"), cypress_config)
        
        e2e_path = Path("cypress/e2e")
        for flow in spec.flows:
            test_code = self._generate_e2e_test(flow)
            artifact.add_file(
                e2e_path / f"{flow['name']}.cy.ts",
                test_code
            )
            
    async def _optimize_code(self,
                         spec: ProjectSpec,
                         artifact: CodeArtifact):
        """Optimize generated code using MCTS."""
        for path, content in artifact.files.items():
            # Use MCTS to explore optimization options
            best_version = self.mcts.optimize_code(content)
            artifact.files[path] = best_version
            
    def _generate_theme(self, design_system: Dict) -> str:
        """Generate theme configuration."""
        return f"""
        export const theme = {{
            colors: {design_system['colors']},
            typography: {design_system['typography']},
            spacing: {design_system['spacing']},
            components: {design_system['components']}
        }}
        """
        
    def _generate_styled_component(self,
                               name: str,
                               design_system: Dict) -> str:
        """Generate styled component."""
        return f"""
        import styled from 'styled-components';
        
        export const {name}Wrapper = styled.div`
            padding: {design_system['spacing']['4']};
            background: {design_system['colors']['background']};
            border-radius: {design_system['components']['card']['borderRadius']};
        `;
        """
        
    def _generate_component_test(self, name: str) -> str:
        """Generate component test."""
        return f"""
        import React from 'react';
        import {{ render, screen }} from '@testing-library/react';
        import {{ {name} }} from '../components/{name}';
        
        describe('{name}', () => {{
            it('renders correctly', () => {{
                render(<{name} />);
                // Add assertions
            }});
        }});
        """
        
    def _generate_e2e_test(self, flow: Dict) -> str:
        """Generate E2E test."""
        return f"""
        describe('{flow["name"]}', () => {{
            it('completes successfully', () => {{
                cy.visit('/');
                // Add flow steps
            }});
        }});
        """