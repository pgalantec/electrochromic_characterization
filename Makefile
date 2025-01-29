.PHONY: init test format lint

create_env: 
	@echo "Creating virtual environment..."
	python3 -m venv env
	./env/bin/python3 -m pip install -e .
	@echo activate env: source env/bin/activate

format:
	ruff src 

lint:
	ruff check src
