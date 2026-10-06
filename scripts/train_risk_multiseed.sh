#!/usr/bin/env bash
# Independent training seeds on separate physical GPUs; one process per GPU.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
gpu_csv="${1:-0,2,3}"
data_path="${2:-output/dataset-261006-risk/manifest.json}"
config_path="${3:-configs/learning/p.json}"
if [[ "$gpu_csv" == --help || "$gpu_csv" == -h ]]; then
  echo 'Usage: bash scripts/train_risk_multiseed.sh [GPU_IDS=0,2,3] [MANIFEST] [CONFIG] [FRESH_RUN_DIRECTORY]'
  exit 0
fi
if [[ ! "$gpu_csv" =~ ^[0-9]+(,[0-9]+)*$ || ! -f "$config_path" ]]; then
  echo 'Supply distinct comma-separated GPU IDs and an existing config file.' >&2
  exit 1
fi
IFS=',' read -r -a gpu_ids <<< "$gpu_csv"
seen_gpus=","
for gpu in "${gpu_ids[@]}"; do
  if [[ "$seen_gpus" == *",$gpu,"* ]]; then
    echo "Duplicate GPU: $gpu" >&2
    exit 1
  fi
  seen_gpus="${seen_gpus}${gpu},"
done
mkdir -p output
exec 9>output/risk-multiseed.lock
if ! flock -n 9; then
  echo 'Another Risk multiseed launcher is running. See its logs before starting again.' >&2
  exit 1
fi
if [[ "$data_path" == output/dataset-261006-risk/manifest.json ]]; then
  # Wait for the already-started deterministic conversion instead of duplicating it.
  while pgrep -f '^python -m experiments\.build_dataset .*--out output/dataset-261006-risk$' >/dev/null; do
    echo 'Waiting for the existing real-window converter; models have not started.'
    sleep 30
  done
  if [[ ! -f "$data_path" ]]; then
    if [[ -d output/dataset-261006-risk ]]; then
      incomplete="output/dataset-261006-risk.incomplete-$(TZ=Asia/Shanghai date +%y%m%d-%H%M%S)"
      mv output/dataset-261006-risk "$incomplete"
      echo "Preserved incomplete conversion at $incomplete"
    fi
    echo 'Building real windows and checking Risk supervision availability.'
    MTHREADS_VISIBLE_DEVICES="${gpu_ids[0]}" MUSA_IMAGE="${MUSA_IMAGE:-dl-musa-render:latest}" \
      ./docker/run-musa.sh python -m experiments.build_dataset \
      --sources output/data-collection-261005/index/sources.json \
      --tracking-thresholds .06 .06 .1 .1 --skip-ineligible --require risk \
      --out output/dataset-261006-risk >output/dataset-261006-risk-build.log 2>&1
  fi
  python3 - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('output/dataset-261006-risk/report.json').read_text())
if not report.get('readiness', {}).get('risk', {}).get('ready'):
    raise SystemExit('Risk supervision is not ready. Inspect the dataset report before training.')
print('Risk readiness passed:', report['windows'], flush=True)
PY
fi
if [[ ! -f "$data_path" ]]; then
  echo "Dataset manifest does not exist: $data_path" >&2
  exit 1
fi
run_root="${4:-output/risk-multiseed-$(TZ=Asia/Shanghai date +%y%m%d-%H%M%S)}"
mkdir "$run_root"
python3 - "$run_root" "$data_path" "$config_path" "$gpu_csv" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, data, config = map(Path, sys.argv[1:4])
gpus = [int(x) for x in sys.argv[4].split(',')]
record = {'scope': 'Parallel independent Risk training seeds; one model per physical GPU',
          'data': str(data.resolve()), 'config': str(config.resolve()),
          'data_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
          'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
          'jobs': [{'seed': seed, 'physical_gpu': gpu} for seed, gpu in enumerate(gpus)]}
(out / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
PY
echo "Output: $repo_root/$run_root"
echo "Independent Risk training seeds on physical GPUs: $gpu_csv"
pids=()
for seed in "${!gpu_ids[@]}"; do
  gpu="${gpu_ids[$seed]}"
  # The selected physical device is logical MUSA device 0 inside its container.
  MTHREADS_VISIBLE_DEVICES="$gpu" MUSA_IMAGE="${MUSA_IMAGE:-dl-musa-render:latest}" \
    ./docker/run-musa.sh env MUSA_VISIBLE_DEVICES=0 python -m experiments.train \
    --stage risk --data "$data_path" --config "$config_path" --seed "$seed" \
    --device musa --threads 4 --out "$run_root/seed-$seed" \
    >"$run_root/seed-$seed.log" 2>&1 &
  pids+=("$!")
  echo "Started seed $seed on GPU $gpu; log: $run_root/seed-$seed.log"
done
failed=0
for seed in "${!pids[@]}"; do
  if wait "${pids[$seed]}"; then
    echo "Seed $seed completed successfully."
  else
    echo "Seed $seed failed; inspect $run_root/seed-$seed.log" >&2
    failed=1
  fi
done
python3 - "$run_root" "$failed" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
jobs = []
for job in json.loads((out / 'launch.json').read_text())['jobs']:
    path = out / f"seed-{job['seed']}" / 'report.json'
    report = json.loads(path.read_text()) if path.is_file() else {'status': 'failed', 'error': 'No report written'}
    jobs.append({**job, 'report': str(path), 'status': report.get('status'),
                 'best_val_loss': report.get('best_val_loss'), 'error': report.get('error')})
summary = {'status': 'failed' if int(sys.argv[2]) else 'passed', 'jobs': jobs}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary), flush=True)
PY
exit "$failed"
