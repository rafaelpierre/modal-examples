.PHONY: deploy

deploy:
	uv run modal deploy src/modal_examples/server/deployment.py
