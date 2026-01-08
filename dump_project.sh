#!/usr/bin/env bash
set -e

OUTPUT_FILE="project_dump.txt"

# Empty or create the output file
: > "$OUTPUT_FILE"

# Find all regular files, excluding .git and the output file itself
find . \
  -path './.git' -prune -o \
  -type f ! -name "$OUTPUT_FILE" -print0 |
while IFS= read -r -d '' file; do
  echo "==================================================" >> "$OUTPUT_FILE"
  echo "FILE: $file" >> "$OUTPUT_FILE"
  echo "==================================================" >> "$OUTPUT_FILE"
  
  if [[ "$file" == ./dataset/* ]]; then
    echo "(Showing first 10 lines only – dataset truncated)" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    head -n 10 "$file" >> "$OUTPUT_FILE"
  else
    cat "$file" >> "$OUTPUT_FILE"
  fi

  echo -e "\n\n" >> "$OUTPUT_FILE"
done

echo "Done. Written to \$OUTPUT_FILE"
