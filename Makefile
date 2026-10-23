# tts-service: repository tasks.

.PHONY: help install-hooks check-md

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install-hooks: ## Use the versioned git hooks in .githooks/ (em-dash check)
	git config core.hooksPath .githooks

check-md: ## Markdown has no em-dashes (self-test, then every tracked .md)
	bash scripts/check-no-em-dash.sh --self-test
	bash scripts/check-no-em-dash.sh
