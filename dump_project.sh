#!/usr/bin/env bash
set -e

OUTPUT_FILE="project_dump.txt"

# Empty or create the output file
: > "$OUTPUT_FILE"

# Find all regular files, excluding:
# 1. .git directory
# 2. core/output directory (too big)
# 3. __pycache__ directories (unnecessary binaries)
# 4. The output file itself
find . \
  -path './.git' -prune -o \
  -path './core/output' -prune -o \
  -name '__pycache__' -prune -o \
  -type f ! -name "$OUTPUT_FILE" -print0 |
while IFS= read -r -d '' file; do
  echo "==================================================" >> "$OUTPUT_FILE"
  echo "FILE: $file" >> "$OUTPUT_FILE"
  echo "==================================================" >> "$OUTPUT_FILE"
  
  # Check for binary extensions (PNG, PYC, etc)
  if [[ "$file" == *.png || "$file" == *.pyc || "$file" == *.jpg ]]; then
    echo "(Binary file excluded)" >> "$OUTPUT_FILE"

  # Check for dataset folder
  elif [[ "$file" == ./dataset/* ]]; then
    echo "(Showing first 10 lines only – dataset truncated)" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    head -n 10 "$file" >> "$OUTPUT_FILE"

  # Standard text files
  else
    cat "$file" >> "$OUTPUT_FILE"
  fi

  echo -e "\n\n" >> "$OUTPUT_FILE"
done

echo "Done. Written to $OUTPUT_FILE"