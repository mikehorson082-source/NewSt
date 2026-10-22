npx repomix \
  --include "src/**,strategies/st01/**,app/api/**,*.md,pyproject.toml" \
  --ignore "data/**,**/*.parquet,**/*.png,**/*.csv,**/*.lock" \
  -o full_context.txt && open -R full_context.txt
