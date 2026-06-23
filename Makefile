# RoleRAG — common tasks. Run `make` or `make help` for the list.
# Thin wrappers over the documented commands; no logic lives here.
.DEFAULT_GOAL := help
.PHONY: help up down dev dev-down install check lint type test smoke

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

up: ## Run app + Qdrant in Docker (model server must run on the host, port 8080)
	docker compose up --build

down: ## Stop and remove the Docker stack
	docker compose down

dev: ## Start the full stack on the host (Qdrant + llama-server + API) via dev-up.sh
	bash scripts/dev-up.sh

dev-down: ## Stop everything dev-up started
	bash scripts/dev-down.sh

install: ## Create .venv and install the package with dev extras
	python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"

check: lint type test ## Run all quality gates (ruff + mypy + pytest)

lint: ## Ruff lint
	.venv/bin/ruff check .

type: ## Mypy strict type-check
	.venv/bin/mypy .

test: ## Run the Python test suite
	.venv/bin/pytest

smoke: ## Deterministic offline end-to-end smoke
	.venv/bin/python -m app.cli smoke-run
