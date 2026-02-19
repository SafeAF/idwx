# Security

## Non-negotiable
- No API keys in git.
- No raw station data in git.
- No derived caches in git.

## .gitignore
Must include:
- data_cache/
- models/
- reports/
- *.parquet
- *.pkl
- .env

## If you ever accidentally commit secrets
- rotate immediately
- purge git history (git filter-repo)
