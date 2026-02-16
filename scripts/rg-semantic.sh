#!/bin/bash
# Semantic search wrapper for ResearchGravity

cd ~/researchgravity
source .venv/bin/activate
export COHERE_API_KEY=$(jq -r .cohere.api_key ~/.agent-core/config.json)
python3 test_semantic_search.py "$@"
