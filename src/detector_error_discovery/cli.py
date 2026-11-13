import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .core import compose, verify_models


def main():
    parser = argparse.ArgumentParser(description='Compact detector-error ranking')
    parser.add_argument('mode', choices=['replay', 'predict', 'verify', 'download'])
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Repository/model bundle directory')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--split', default='test', choices=['test', 'train'])
    parser.add_argument('--day', help='For fold evaluation: predict only this held-out TRAIN day')
    parser.add_argument('--output', type=Path, default=Path('outputs/submission.csv'))
    parser.add_argument('--cache-dir', type=Path, default=Path('.cache/predictions'))
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    root = args.root.resolve()
    config = verify_models(root)
    if args.mode == 'verify':
        print('All bundled model checksums are valid.')
        return
    if args.mode == 'download':
        from huggingface_hub import snapshot_download
        path = snapshot_download(config['grounding_model'], revision=config['grounding_revision'], allow_patterns=['*.json', '*.safetensors', '*.txt'])
        print(f'Public Grounding DINO snapshot: {path}')
        return
    if args.mode == 'replay':
        features = pd.read_csv(root/'reference/features.csv', dtype={'id': str})
        signals = pd.read_csv(root/'reference/signals.csv', dtype={'id': str})
    else:
        if args.data_dir is None:
            parser.error('predict requires --data-dir')
        from .features import build
        from .vision import infer
        features = build(args.data_dir, args.split)
        if args.day:
            if args.split != 'train' or config.get('training_holdout') != args.day:
                parser.error('--day requires --split train and a bundle trained with this --holdout')
            features = features.loc[features.day.astype(str).eq(args.day)].reset_index(drop=True)
            if features.empty:
                parser.error('No detections found for requested day')
        features, signals = infer(features, args.data_dir, root, config, args.cache_dir, args.device)
    prediction = compose(features, signals, root, config)
    template_path = root/'reference/submission.csv' if args.mode == 'replay' else args.data_dir/'sample_submission.csv'
    if args.split == 'test' or args.mode == 'replay':
        template = pd.read_csv(template_path, dtype={'id': str})[['id']]
        if set(template.id) != set(prediction.id):
            raise ValueError('Submission IDs do not match the sample submission.')
        prediction = template.merge(prediction, on='id', validate='one_to_one', sort=False)
    if args.mode == 'replay':
        expected = pd.read_csv(root/'reference/submission.csv', dtype={'id': str})
        np.testing.assert_array_equal(prediction.id, expected.id)
        np.testing.assert_allclose(prediction.score, expected.score, atol=1e-12, rtol=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(args.output, index=False)
    print(f'Wrote {len(prediction)} detections to {args.output}')


if __name__ == '__main__':
    main()
