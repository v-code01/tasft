.PHONY: install lint typecheck test bench chaos clean

install:
	pip install -e ".[dev,train,eval]"
	pre-commit install

lint:
	ruff check tasft/ tests/
	ruff format --check tasft/ tests/

typecheck:
	mypy tasft/

test:
	pytest tests/unit/ tests/integration/ -v --tb=short -m "not slow"

bench:
	pytest tests/benchmarks/ --benchmark-only -v --benchmark-json=bench_results.json

chaos:
	pytest tests/chaos/ -v --timeout=120

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -f bench_results.json
