#!/bin/zsh
# Re-run of the arms that failed on a transient API outage (C3, D, Cbest).
set -u
cd "$(dirname "$0")/.."
BASE=(--config enterprise_rag --enterprise_subset_size 800 --enterprise_split test)
DEFAULT_W="{'alpha': 1.0, 'beta': 0.5, 'gamma': 0.5, 'delta': 0.3, 'lambda': 0.3}"
BEST_W="{'alpha': 1.0, 'beta': 0.5, 'gamma': 0.25, 'delta': 0.0, 'lambda': 0.3}"
run() {
  local tag=$1; shift
  echo "=== $tag  $(date +%H:%M:%S)"
  .venv/bin/python qa.py   "${BASE[@]}" --run_tag "$tag" "$@" 2>&1 | grep -v "it/s\]" | tail -2
  .venv/bin/python eval.py "${BASE[@]}" --run_tag "$tag" "$@" 2>&1 | grep "Evaluation results" | cut -c1-300
}
run C3    --retrieve_mode hybrid_score --query_understanding llm --score_weights "$DEFAULT_W"
run D     --retrieve_mode hybrid_score --query_understanding llm --score_weights "$DEFAULT_W" --metadata_filter true
run Cbest --retrieve_mode hybrid_score --query_understanding llm --score_weights "$BEST_W"
echo "=== done $(date +%H:%M:%S)"
