"""CSV rows and metrics shared by training, validation and test."""
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from modal import model_mil

ATTENTION = ('attn_main', 'attn_lower', 'attn_upper', 'attn_texture')
METRICS = ('accuracy', 'macro_precision', 'macro_recall', 'macro_f1')
CASE_FIELDS = (
    'patient_id', 'true_class', 'pred_class', 'correct',
    'prob_0', 'prob_1', 'prob_2', 'pred_confidence', 'true_class_probability',
    'loss_smoothed', 'loss_ce', *ATTENTION, 'attn_max', 'attn_entropy',
)


@torch.no_grad()
def case_rows(logits, target, attention, patient_ids):
    if attention.shape[1] != len(ATTENTION):
        raise ValueError('Expected four attention weights per case')
    scores = logits.detach().cpu()
    labels = target.cpu()
    probs = scores.softmax(1).tolist()
    pred = scores.argmax(1).tolist()
    smooth = F.cross_entropy(scores, labels, label_smoothing=.1, reduction='none').tolist()
    raw = F.cross_entropy(scores, labels, reduction='none').tolist()
    weights = attention.detach().cpu()
    entropy = (-(weights * (weights + 1e-8).log()).sum(dim=1)).tolist()
    rows = []
    for i, (patient, label) in enumerate(zip(patient_ids, labels.tolist())):
        guess = pred[i]
        row = dict(patient_id=patient, true_class=label, pred_class=guess,
                   correct=int(guess == label), pred_confidence=probs[i][guess],
                   true_class_probability=probs[i][label],
                   loss_smoothed=smooth[i], loss_ce=raw[i],
                   attn_max=float(weights[i].max()), attn_entropy=entropy[i])
        row.update(zip(ATTENTION, weights[i].tolist()))
        row.update({f'prob_{j}': value for j, value in enumerate(probs[i])})
        rows.append(row)
    return rows


def summarize_cases(rows):
    if not rows:
        raise ValueError('Evaluation split is empty')
    actual = [row['true_class'] for row in rows]
    predicted = [row['pred_class'] for row in rows]
    p, r, f, support = precision_recall_fscore_support(actual, predicted, labels=[0, 1, 2], zero_division=0)
    report = {
        'num_cases': len(rows),
        'accuracy': float(np.mean([row['correct'] for row in rows])),
        'macro_precision': float(p.mean()), 'macro_recall': float(r.mean()), 'macro_f1': float(f.mean()),
        'confusion_matrix': confusion_matrix(actual, predicted, labels=[0, 1, 2]).tolist(),
        'mean_attention': {key: float(np.mean([row[key] for row in rows])) for key in ATTENTION},
        'per_class': {str(i): dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]),
                                   support=int(support[i])) for i in range(3)},
    }
    for key in ('loss_smoothed', 'loss_ce', 'attn_max', 'attn_entropy'):
        report[key] = float(np.mean([row[key] for row in rows]))
    return report


def write_csv(path, fields, rows=(), append=False):
    with open(path, 'a' if append else 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not append:
            writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_batches(model, loader):
    model.eval()
    device = next(model.parameters()).device
    for inputs, target, patient_ids in loader:
        inputs = model_mil.move_to_device(inputs, device)
        logits, attention = model(inputs)
        yield case_rows(logits, target, attention, patient_ids)


def evaluate(model, loader, csv_name, metrics_path=None, metadata=None):
    logger = logging.getLogger('evaluation')
    write_csv(csv_name, CASE_FIELDS)
    rows = []
    for batch in evaluate_batches(model, loader):
        write_csv(csv_name, CASE_FIELDS, batch, append=True)
        rows.extend(batch)
        logger.info('Evaluated %d cases; latest=%s', len(rows), batch[-1]['patient_id'])
    report = summarize_cases(rows)
    if metadata:
        report['evaluation'] = metadata
    message = ', '.join(f'{key}={report[key]:.4f}' for key in (*METRICS, 'loss_smoothed'))
    print(f'{message}\nConfusion matrix:\n{np.asarray(report["confusion_matrix"])}', flush=True)
    logger.info('%s; confusion matrix=%s', message, report['confusion_matrix'])
    if metrics_path:
        path = Path(metrics_path)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        keys = ('num_cases', *METRICS, 'loss_smoothed', 'loss_ce')
        overall = {'class': 'overall', **{key: report.get(key) for key in keys}, **report['mean_attention']}
        class_rows = []
        for label, metrics in report['per_class'].items():
            confusion = report['confusion_matrix'][int(label)]
            class_rows.append({'class': label, **metrics, **{f'pred_{i}': count for i, count in enumerate(confusion)}})
        write_csv(path.with_suffix('.csv'),
                  ['class', *keys, *ATTENTION, 'support', 'precision', 'recall', 'f1', 'pred_0', 'pred_1', 'pred_2'],
                  [overall, *class_rows])
    return tuple(report[key] for key in METRICS)

