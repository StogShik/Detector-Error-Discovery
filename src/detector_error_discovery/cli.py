import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .core import apply_person_geometry, compose, verify_models


def main():
    parser = argparse.ArgumentParser(description='Compact detector-error ranking')
    parser.add_argument('mode', choices=['replay', 'predict', 'verify', 'download', 'correct-people'])
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Repository/model bundle directory')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--base-submission', type=Path, help='Existing id,score CSV for correct-people; other classes are preserved')
    parser.add_argument('--split', default='test', choices=['test', 'train'])
    parser.add_argument('--day', help='For fold evaluation: predict only this held-out TRAIN day')
    parser.add_argument('--output', type=Path, default=Path('outputs/submission.csv'))
    parser.add_argument('--cache-dir', type=Path, default=Path('.cache/predictions'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--recipe', choices=['standard', 'hand-consensus', 'person-truncation'], default='standard', help='Optional experimental recipe; default is unchanged')
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
        if args.recipe == 'person-truncation':
            from .person_geometry import download
            print(f'Public full-body model: {download(root)}')
        return
    if args.mode == 'correct-people':
        if args.data_dir is None or args.base_submission is None or args.split != 'test':
            parser.error('correct-people requires --data-dir, --base-submission and --split test')
        from .person_geometry import infer as infer_people
        features = pd.read_csv(args.data_dir/'test.csv', dtype={'id': str})
        features = features.loc[features.class_name.ne('table')].reset_index(drop=True)
        existing = pd.read_csv(args.base_submission, dtype={'id': str})
        if existing.id.duplicated().any() or set(existing.id) != set(features.id):
            raise ValueError('Base submission must contain every unique detection ID.')
        baseline = features.id.map(existing.set_index('id').score)
        if not np.isfinite(baseline).all() or not baseline.between(0, 1).all():
            raise ValueError('Base submission scores must be finite and in [0, 1].')
        signals = infer_people(features, args.data_dir, root, args.cache_dir, args.device)
        prediction = features[['id']].assign(score=apply_person_geometry(baseline, features, signals))
    elif args.mode == 'replay':
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
    if args.mode != 'correct-people' and args.recipe == 'hand-consensus':
        config = dict(config, hand_geometry='sharp_rotation_confidence')
    if args.mode != 'correct-people' and args.recipe == 'person-truncation':
        if args.mode == 'replay':
            extra = pd.read_csv(root/'reference/person_truncation_signals.csv', dtype={'id': str})
        else:
            from .person_geometry import infer as infer_people
            extra = infer_people(features, args.data_dir, root, args.cache_dir, args.device)
        signals = signals.merge(extra, on='id', validate='one_to_one', sort=False)
        config = dict(config, person_geometry='truncation')
    if args.mode != 'correct-people':
        prediction = compose(features, signals, root, config)
    template_path = root/'reference/submission.csv' if args.mode == 'replay' else args.data_dir/'sample_submission.csv'
    if args.split == 'test' or args.mode == 'replay':
        template = pd.read_csv(template_path, dtype={'id': str})[['id']]
        if set(template.id) != set(prediction.id):
            raise ValueError('Submission IDs do not match the sample submission.')
        prediction = template.merge(prediction, on='id', validate='one_to_one', sort=False)
    if args.mode == 'replay':
        filename = {'standard': 'submission.csv', 'hand-consensus': 'submission_hand_consensus.csv', 'person-truncation': 'submission_person_truncation.csv'}[args.recipe]
        expected = pd.read_csv(root/'reference'/filename, dtype={'id': str})
        np.testing.assert_array_equal(prediction.id, expected.id)
        np.testing.assert_allclose(prediction.score, expected.score, atol=1e-12, rtol=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(args.output, index=False)
    print(f'Wrote {len(prediction)} detections to {args.output}')


if __name__ == '__main__':
    main()
