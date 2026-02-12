#!/usr/bin/env sh
# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
#         DOI: 10.3969/j.issn.1671-3370.2026.02.032 (S.Chang) https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

set -eu

# KB_BUILD_REGION 可显式设置为 cn 或 global；未设置时尝试根据公网出口国家判断。
if [ -z "${KB_BUILD_REGION:-}" ]; then
  country="$(curl -fsS --max-time 3 https://ipapi.co/country/ 2>/dev/null || true)"
  if [ "$country" = "CN" ]; then
    KB_BUILD_REGION=cn
  else
    KB_BUILD_REGION=global
  fi
fi

if [ "$KB_BUILD_REGION" = "cn" ]; then
  : "${PIP_INDEX_URL:=https://mirrors.aliyun.com/pypi/simple}"
else
  : "${PIP_INDEX_URL:=https://pypi.org/simple}"
fi
export KB_BUILD_REGION PIP_INDEX_URL
printf 'KB_BUILD_REGION=%s\nPIP_INDEX_URL=%s\n' "$KB_BUILD_REGION" "$PIP_INDEX_URL"
exec "$@"
