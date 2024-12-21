#!/bin/bash

# Check for required API keys
if [ -z "$OPENAI_API_KEY" ] || [ -z "$ANTHROPIC_API_KEY" ] || [ -z "$HUGGINGFACE_API_KEY" ]; then
    echo "Error: Missing required API keys"
    echo "Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, and HUGGINGFACE_API_KEY"
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Start services based on mode
MODE=${1:-dev}

case $MODE in
    "dev")
        echo "Starting development environment..."
        docker-compose -f docker-compose.dev.yml up -d
        
        # Start frontend
        echo "Starting frontend..."
        cd frontend && npm install && npm run dev &
        
        # Start backend
        echo "Starting backend..."
        cd backend && poetry install && poetry run uvicorn app.main:app --reload &
        ;;
        
    "prod")
        echo "Starting production environment..."
        
        # Build images
        docker-compose build
        
        # Start core services
        docker-compose up -d postgres redis elasticsearch
        
        # Wait for databases
        echo "Waiting for databases to be ready..."
        ./scripts/wait-for-it.sh postgres:5432
        ./scripts/wait-for-it.sh redis:6379
        ./scripts/wait-for-it.sh elasticsearch:9200
        
        # Run migrations
        docker-compose run --rm backend alembic upgrade head
        
        # Start all services
        docker-compose up -d
        
        # Start monitoring
        docker-compose -f docker-compose.monitoring.yml up -d
        ;;
        
    "k8s")
        echo "Starting Kubernetes deployment..."
        
        # Check kubectl access
        if ! kubectl get nodes > /dev/null; then
            echo "Error: Cannot access Kubernetes cluster"
            exit 1
        fi
        
        # Create namespace
        kubectl create namespace social-app
        
        # Create secrets
        kubectl create secret generic api-keys \
            --from-literal=openai-key=$OPENAI_API_KEY \
            --from-literal=anthropic-key=$ANTHROPIC_API_KEY \
            --from-literal=huggingface-key=$HUGGINGFACE_API_KEY \
            -n social-app
            
        # Deploy application
        kubectl apply -f deployment/k8s/ -n social-app
        
        # Wait for deployment
        kubectl wait --for=condition=available --timeout=300s deployment/frontend -n social-app
        kubectl wait --for=condition=available --timeout=300s deployment/backend -n social-app
        
        echo "Application deployed to Kubernetes!"
        ;;
        
    *)
        echo "Invalid mode. Use: dev, prod, or k8s"
        exit 1
        ;;
esac

# Print access URLs
echo "
Application started! Access at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/api/docs
- Grafana: http://localhost:3001
- Kibana: http://localhost:5601
"