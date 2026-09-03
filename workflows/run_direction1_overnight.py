"""Run resumable independent test2 screening batches."""

import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import evaluate  # noqa: E402


DATA = Path('/home/aiattacks/pengyang/Test/group1/test2(whole)/AADD_2026_Test')
EXCLUDE = Path('/home/aiattacks/pengyang/Test/group1/test1')
OUT = ROOT / 'experiments/direction1/overnight'


def make_cfg(name, attack, offset, count, vit_weight=0.5, dct_weight=0.5,
             step_size=0.5 / 255, iterations=40):
    cfg = {
        'original_root': str(DATA),
        'exclude_root': str(EXCLUDE),
        'sample_offset': offset,
        'max_images': count,
        'attack': attack,
        'save_attacked_dir': str(OUT / f'attacked_{name}'),
        'models_dir': str(ROOT / 'weights'),
        'save_json': str(OUT / f'results_{name}.json'),
        'classifiers': ['vit_b_16', 'densenet121_dct'],
        'dct_log_scale': True,
        'dct_resize_mode': 'bicubic',
        'weights': {'vit_b_16': 1.0, 'densenet121_dct': 1.0},
        'aggregate': 'sum',
        'device': 'cuda',
        'alpha': 0.5,
        'seed': 0,
    }
    if attack != 'identity':
        cfg.update({
            'epsilon': 8 / 255,
            'step_size': step_size,
            'iterations': iterations,
            'target': 0,
            'vit_weight': vit_weight,
            'dct_weight': dct_weight,
            'normalize_gradients': True,
        })
    return cfg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Offsets are applied after removing the 46 test1 filenames.
    batches = [('head', 0), ('middle', 727), ('tail', 1454)]
    runs = []
    for label, offset in batches:
        runs.append((f'{label}_identity', make_cfg(
            f'{label}_identity', 'identity', offset, 100)))
        runs.append((f'{label}_best', make_cfg(
            f'{label}_best', 'dual_pgd', offset, 100)))
        runs.append((f'{label}_dct65', make_cfg(
            f'{label}_dct65', 'dual_pgd', offset, 100, 0.35, 0.65,
            2 / 255, 10)))

    summary = []
    for name, cfg in runs:
        result_path = OUT / f'results_{name}.json'
        if result_path.exists():
            summary.append({'name': name, 'status': 'already_completed',
                            'result': json.loads(result_path.read_text())})
            continue
        (OUT / f'config_{name}.yaml').write_text(
            yaml.safe_dump(cfg, sort_keys=False))
        started = time.time()
        print(f'[WORKFLOW] START {name}', flush=True)
        try:
            evaluate(cfg)
            status = 'ok'
        except Exception as exc:
            status = f'error: {type(exc).__name__}: {exc}'
            print(f'[WORKFLOW] {name}: {status}', flush=True)
        item = {'name': name, 'status': status,
                'runtime_sec': time.time() - started}
        if result_path.exists():
            item['result'] = json.loads(result_path.read_text())
        summary.append(item)
        (OUT / f'status_{name}.json').write_text(json.dumps(item, indent=2))

    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    (OUT / 'COMPLETE').write_text(
        time.strftime('%Y-%m-%d %H:%M:%S UTC\n', time.gmtime()))
    print(f'[WORKFLOW] COMPLETE: {OUT / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
