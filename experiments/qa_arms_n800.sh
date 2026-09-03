#!/bin/zsh
# QA + LLM-judge runs of the headline arms on the n800 test split (350 questions).
set -u
cd "$(dirname "$0")/.."
BASE=(--config enterprise_rag --enterprise_subset_size 800 --enterprise_split test)
DEFAULT_W="{'alpha': 1.0, 'beta': 0.5, 'gamma': 0.5, 'delta': 0.3, 'lambda': 0.3}"
BEST_W="{'alpha': 1.0, 'beta': 0.5, 'gamma': 0.25, 'delta': 0.0, 'lambda': 0.3}"   # dev grid best (output/experiments/n800_dev/grid.json)
run() {  # run <tag> <overrides...>
  local tag=$1; shift
  echo "=== $tag  $(date +%H:%M:%S)"
  .venv/bin/python qa.py   "${BASE[@]}" --run_tag "$tag" "$@" 2>&1 | grep -v "it/s\]" | tail -2
  .venv/bin/python eval.py "${BASE[@]}" --run_tag "$tag" "$@" 2>&1 | grep "Evaluation results" | cut -c1-2000
}
run A0    --retrieve_mode legacy --query_understanding none
run C0    --retrieve_mode hybrid_score --query_understanding none --score_weights "{'alpha': 1.0, 'beta': 0.5, 'gamma': 0.0, 'delta': 0.0, 'lambda': 0.0}"
run C3    --retrieve_mode hybrid_score --query_understanding llm  --score_weights "$DEFAULT_W"
run D     --retrieve_mode hybrid_score --query_understanding llm  --score_weights "$DEFAULT_W" --metadata_filter true
run Cbest --retrieve_mode hybrid_score --query_understanding llm  --score_weights "$BEST_W"
run B     --retrieve_mode legacy --query_understanding none --enterprise_chunk_metadata_prefix true
run BC3   --retrieve_mode hybrid_score --query_understanding llm  --score_weights "$DEFAULT_W" --enterprise_chunk_metadata_prefix true
echo "=== done $(date +%H:%M:%S)"
