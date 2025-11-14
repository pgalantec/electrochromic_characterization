.PHONY: init test format lint install-uv create_env_uv

# Check if uv is installed, if not install it
install-uv:
	@echo "🔍 Checking if uv is installed..."
	@if ! command -v uv >/dev/null 2>&1; then \
	echo "📦 uv not found. Installing uv..."; \
	curl -LsSf https://astral.sh/uv/install.sh | sh; \
	echo "✅ uv installed successfully!"; \
	echo "🔄 Please run 'source ~/.bashrc' or restart your terminal, then run 'make init' again"; \
	exit 1; \
	else \
	echo "✅ uv is already installed: $$(uv --version)"; \
	fi

# Create environment with uv (recommended - much faster)
init: install-uv
	@echo "🚀 Creating virtual environment with uv..."
	uv venv --python 3.10
	@echo "📦 Installing project in development mode..."
	uv pip install -e .
	@echo "✅ Environment created successfully!"
	@echo ""
	@echo "📋 To activate the environment:"
	@echo "   source .venv/bin/activate"
	@echo ""
	@echo "🎯 To run the configuration app:"
	@echo "   make run_config_app_uv"
	@echo ""
	@echo "🔬 To run experiments:"
	@echo "   make run_experiment_uv"

# Alternative: Create environment with standard venv (slower but no dependencies)
create_env:
	@echo "🐍 Creating virtual environment with standard venv..."
	python3 -m venv env
	./env/bin/python3 -m pip install -e .
	@echo "✅ Environment created successfully!"
	@echo ""
	@echo "📋 To activate the environment:"
	@echo "   source env/bin/activate"
	@echo ""
	@echo "🎯 To run the configuration app:"
	@echo "   make run_config_app"

# Run apps with uv environment
run_config_app_uv:
	@if [ ! -d ".venv" ]; then echo "❌ Virtual environment not found. Run 'make init' first."; exit 1; fi
	.venv/bin/python -m src.lego_app_v4

run_experiment_uv:
	@if [ ! -d ".venv" ]; then echo "❌ Virtual environment not found. Run 'make init' first."; exit 1; fi
	.venv/bin/python -m src.experiment

# Run apps with standard venv
run_config_app:
	@if [ ! -d "env" ]; then echo "❌ Virtual environment not found. Run 'make create_env' first."; exit 1; fi
	./env/bin/python3 -m src.lego_app_v4

run_experiment:
	@if [ ! -d "env" ]; then echo "❌ Virtual environment not found. Run 'make create_env' first."; exit 1; fi
	./env/bin/python3 -m src.experiment

# Development tools
format:
	@if command -v uv >/dev/null 2>&1 && [ -d ".venv" ]; then \
	.venv/bin/ruff format src; \
	elif [ -d "env" ]; then \
	./env/bin/ruff format src; \
	else \
	echo "❌ No virtual environment found. Run 'make init' or 'make create_env' first."; \
	exit 1; \
	fi

lint:
	@if command -v uv >/dev/null 2>&1 && [ -d ".venv" ]; then \
	.venv/bin/ruff check src; \
	elif [ -d "env" ]; then \
	./env/bin/ruff check src; \
	else \
	echo "❌ No virtual environment found. Run 'make init' or 'make create_env' first."; \
	exit 1; \
	fi

# Clean environments
clean:
	@echo "🧹 Cleaning up virtual environments..."
	@if [ -d ".venv" ]; then rm -rf .venv; echo "✅ Removed .venv"; fi
	@if [ -d "env" ]; then rm -rf env; echo "✅ Removed env"; fi

# Help
help:
	@echo "🛠️  Electrochromic Characterization - Available Commands"
	@echo ""
	@echo "📦 Environment Setup:"
	@echo "   make init          - Create environment with uv (recommended, fast)"
	@echo "   make create_env    - Create environment with standard venv (slower)"
	@echo "   make install-uv    - Install uv package manager"
	@echo ""
	@echo "🚀 Running Applications:"
	@echo "   make run_config_app_uv  - Run LEGO configuration app (uv env)"
	@echo "   make run_experiment_uv  - Run experiment processing (uv env)"
	@echo "   make run_config_app     - Run LEGO configuration app (standard env)"
	@echo "   make run_experiment     - Run experiment processing (standard env)"
	@echo ""
	@echo "🔧 Development Tools:"
	@echo "   make format        - Format code with ruff"
	@echo "   make lint          - Lint code with ruff"
	@echo "   make clean         - Remove all virtual environments"
	@echo "   make help          - Show this help message"
	@echo ""
	@echo "💡 Quick Start:"
	@echo "   1. make init                    # Setup environment"
	@echo "   2. source .venv/bin/activate    # Activate environment"
	@echo "   3. make run_config_app_uv       # Start configuration app"

.DEFAULT_GOAL := help
