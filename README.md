find src strategies -type f \( -name "*.py" -o -name "*.md" \) -not -path "*/__pycache__/*" -exec sh -c 'for f; do printf "\n\n=== FILE: %s ===\n\n" "$f"; cat "$f"; done' _ {} + > context.txt
