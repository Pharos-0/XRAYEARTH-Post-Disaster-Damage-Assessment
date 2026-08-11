#!/usr/bin/env bash
# run_map_eval.sh
# XRayEarth -- Run mAP evaluation for all versions sequentially
#
# Usage:
#   bash scripts/run_map_eval.sh
#   bash scripts/run_map_eval.sh --versions v1,v7,v10
#   bash scripts/run_map_eval.sh --map50-only
#   bash scripts/run_map_eval.sh --max-tiles 100   # quick debug

set -e
cd "$(dirname "$0")/.."

VERSIONS="v1 v2 v3 v4 v5 v6 v7 v8 v9 v10"
MAP50_ONLY=""
MAX_TILES=""
OUTPUT_DIR=""

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --versions)   VERSIONS="${2//,/ }";  shift 2 ;;
        --map50-only) MAP50_ONLY="--map50-only"; shift ;;
        --max-tiles)  MAX_TILES="--max-tiles $2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="--output-dir $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "================================================"
echo "  XRayEarth -- mAP Evaluation"
echo "  Versions: $VERSIONS"
echo "================================================"
echo ""

FAILED=()
PASSED=()

for v in $VERSIONS; do
    config="configs/${v}.yaml"

    if [ ! -f "$config" ]; then
        echo "[SKIP] $v -- config not found: $config"
        continue
    fi

    echo ""
    echo "---- $v ----------------------------------------"

    START=$SECONDS
    if python src/evaluate_map.py \
        --config "$config" \
        $MAP50_ONLY \
        $MAX_TILES \
        $OUTPUT_DIR; then
        ELAPSED=$(( SECONDS - START ))
        echo "[OK]   $v -- completed in ${ELAPSED}s"
        PASSED+=("$v")
    else
        echo "[FAIL] $v"
        FAILED+=("$v")
    fi
done

echo ""
echo "================================================"
echo "  mAP Evaluation Complete"
echo "  Passed: ${PASSED[*]}"
[ ${#FAILED[@]} -gt 0 ] && echo "  Failed: ${FAILED[*]}"
echo "================================================"
echo ""

# Regenerate final summary
echo "Generating summary..."
python src/evaluate_map.py --config configs/v1.yaml --summary $OUTPUT_DIR

echo ""
echo "Results saved to: outputs/map_results/"
