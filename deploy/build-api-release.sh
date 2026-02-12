#!/bin/sh
# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
# Personal Research Project · Author: S.Chang (常帅)
# Paper: 常帅. 企业级高性能离网知识库可溯源检索与生成架构[J].
#        China Information World, 2026(2).
#        DOI: 10.3969/j.issn.1671-3370.2026.02.032
#        https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

# Run from the repository root. Build provenance must match clean source.
set -eu
test -z "$(git status --porcelain)" || { echo "Refusing build from dirty source" >&2; exit 1; }
KB_GIT_COMMIT=$(git rev-parse HEAD)
KB_GIT_TAG=$(git tag --points-at HEAD | head -n 1)
export KB_GIT_COMMIT KB_GIT_TAG
docker-compose -f deploy/compose.yaml build api
