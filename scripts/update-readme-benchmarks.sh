#!/usr/bin/env bash
# update-readme-benchmarks.sh
#
# Runs Criterion benchmarks and updates the Performance section in README.md
# with the latest numbers.
#
# Usage: bash scripts/update-readme-benchmarks.sh
#
# Requires: cargo, grep, sed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== Running Criterion benchmarks (short sample) ==="
cargo bench -- --sample-size 10 --measurement-time 5 2>&1 | tee /tmp/bench_output.txt

echo ""
echo "=== Parsing benchmark results ==="

# Extract benchmark group names and their times from the Criterion output
# Format: "bench_name        time:   [X.XXX ms/µs/s]"

declare -A results

while IFS= read -r line; do
    # Match lines like "  bench_name        time:   [1.234 ms]"
    if [[ $line =~ ^[[:space:]]*([a-zA-Z_/]+)[[:space:]]+time:[[:space:]]+\[([0-9.]+)\ ([a-z]+) ]]; then
        name="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        unit="${BASH_REMATCH[3]}"
        results["$name"]="$value $unit"
    fi
done < /tmp/bench_output.txt

echo "Found ${#results[@]} benchmark results"

# Now update README.md
README="$PROJECT_DIR/README.md"

# Construct the benchmark table
BENCH_TABLE="| Operation | Time |\n|:----------|:-----|\n"

# Template matches - extract key benchmarks
for key in "${!results[@]}"; do
    # Try to map benchmark names to user-friendly labels
    label=""
    case "$key" in
        *"hidden_only"*) label="Prefill, no vocabulary projection" ;;
        *"prefill_dtype/f16"*) label="Prefill (FP16)" ;;
        *"prefill_dtype/f32"*) label="Prefill (FP32)" ;;
        *"prefill"*) label="Prefill (32 tokens)" ;;
        *"generator"*) label="Generate (32 prompt + 16 new)" ;;
        *"generate"*) label="Forward passes (32 prompt + 16 new)" ;;
        *"weight_load"*) label="Weight loading" ;;
        *) label="$key" ;;
    esac
    BENCH_TABLE+="| $label | ${results[$key]} |\n"
done

# If no results parsed, use placeholder
if [ ${#results[@]} -eq 0 ]; then
    BENCH_TABLE+="| _(run benchmarks to populate)_ | |\n"
fi

echo ""
echo "=== Updating README.md Performance Section ==="

# Check if the benchmark section exists
if grep -q "## ⚡ Performance" "$README"; then
    # Replace the content between "## ⚡ Performance" and the next "---"
    awk -v table="$BENCH_TABLE" '
    BEGIN { in_perf = 0; replaced = 0; }
    /^## ⚡ Performance/ { 
        in_perf = 1; 
        print; 
        print "";
        print "*Benchmarked on i5-6600 (4C/4T, 7.5GB RAM) — release mode*";
        print "";
        # Print the table
        split(table, lines, "\\n");
        for (i in lines) {
            if (lines[i] != "") print lines[i];
        }
        print "";
        next;
    }
    in_perf == 1 && /^---/ { 
        in_perf = 0; 
        replaced = 1; 
        print;
        next; 
    }
    in_perf == 1 { next; }
    { print; }
    ' "$README" > "${README}.tmp" && mv "${README}.tmp" "$README"
    echo "✓ Updated Performance section in README.md"
else
    echo "⚠ Could not find '## ⚡ Performance' section in README.md"
    exit 1
fi

echo ""
echo "=== Done ==="
echo "Benchmark results have been written to README.md"
echo ""
echo "To commit:"
echo "  git add README.md"
echo "  git commit -m \"docs: update benchmark results\""
