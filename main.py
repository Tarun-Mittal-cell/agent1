"""
TenzinAgent: Advanced Intelligent Coding Agent
Main orchestrator for multi-agent code generation system.
"""
import argparse
import json
import logging
from pathlib import Path

from core import ProjectSpec
from agent_manager import AgentManager
from llm_integration import LLMProvider
from knowledge_base import KnowledgeBase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for TenzinAgent."""
    parser = argparse.ArgumentParser(description="TenzinAgent CLI")
    parser.add_argument(
        "--spec", 
        type=str, 
        help="Path to project specification JSON file",
        required=True
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Output directory for generated code",
        default="./output"
    )
    parser.add_argument(
        "--enable-self-improvement",
        action="store_true",
        help="Enable agent self-improvement"
    )
    args = parser.parse_args()

    try:
        # Load project specification
        with open(args.spec) as f:
            spec_data = json.load(f)
        spec = ProjectSpec(**spec_data)
        
        # Initialize components
        llm = LLMProvider()
        kb = KnowledgeBase(Path("knowledge"))
        
        # Initialize agent manager
        manager = AgentManager(
            llm_provider=llm,
            knowledge_base=kb,
            enable_self_improvement=args.enable_self_improvement,
            output_dir=Path(args.output)
        )
        
        # Generate application
        logger.info(f"Generating application: {spec.name}")
        artifact = manager.generate_application(spec)
        
        # Save generated code
        artifact.save(Path(args.output))
        logger.info(f"Application generated successfully at: {args.output}")
        
        # Print next steps
        print("\nNext steps:")
        print("1. cd", args.output)
        print("2. npm install  # Install frontend dependencies")
        print("3. pip install -r requirements.txt  # Install backend dependencies")
        print("4. npm run dev  # Start frontend - http://localhost:3000")
        print("5. python -m uvicorn app.main:app --reload  # Start backend - http://localhost:8000")
        
    except Exception as e:
        logger.error(f"Error generating application: {str(e)}")
        raise

if __name__ == "__main__":
    main()