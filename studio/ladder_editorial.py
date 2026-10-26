"""Verify a same-shot editorial crop without concealing its refused parent."""
import json

from PIL import Image


def verify(folder, row, job, model, size, key):
    from .ladder_replay import digest

    def bound(spec):
        path = (folder / spec['file']).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or path.stat().st_size > 20_000_000 \
                or digest(path) != spec.get('sha256'):
            raise ValueError('corrupt or unsafe editorial crop evidence')
        return path

    binding = row['editorial_crop']
    proof_path = bound(binding)
    if proof_path.stat().st_size > 1_000_000:
        raise ValueError('oversized editorial crop receipt')
    proof = json.loads(proof_path.read_text())
    if proof.get('schema') != 1 or proof.get('operation') != 'crop_and_resize' \
            or proof.get('resampling') != 'Pillow LANCZOS' or proof.get('output_size') != list(size) \
            or proof.get('source_image') != job['id'] or row.get('source_image', job['id']) != job['id'] \
            or row.get('generation'):
        raise ValueError('editorial crop must retain its exact original shot and operation')
    origin = row.get('source', {})
    if proof.get('source') != origin or origin.get('key') != key:
        raise ValueError('editorial crop differs from its original cloud source')
    plan = json.loads(bound(proof['plan']).read_text())
    report = json.loads(bound(proof['report']).read_text())
    reviews = json.loads(bound(proof['parent_reviews']).read_text())
    plan_artifact = json.loads(bound(proof['plan_artifact']).read_text())
    image_artifact = json.loads(bound(proof['image_artifact']).read_text())
    if any(m.get('run_id') != origin.get('run_id') or m.get('source_sha') != origin.get('sha')
           for m in (plan_artifact, image_artifact)) or image_artifact.get('id') != row.get('artifact_id'):
        raise ValueError('editorial crop artifact differs from the original cloud source')
    parent = bound(proof['parent'])
    from . import ladder
    source_job = next((j for j in ladder.image_jobs(plan['premise']) if j['id'] == job['id']), None)
    entries = [i for i in report.get('images', []) if i.get('id') == job['id']]
    if plan.get('key') != key or plan['premise']['id'] != report.get('premise') \
            or report.get('model') != model or len(entries) != 1 or not entries[0].get('ok') \
            or source_job != job or proof.get('source_prompt') != job['prompt']:
        raise ValueError('editorial crop differs from its original generated scene')
    kept = [k for k in entries[0].get('kept', []) if k.get('take') == row['take']]
    if len(kept) != 1 or proof['parent'].get('source_file') != kept[0]['plate'].replace('.jpg', '.png') \
            or entries[0]['takes'][row['take']]['exposure'] != row['exposure']:
        raise ValueError('editorial crop lacks its original take and exposure')
    holds = [r for r in reviews.get('images', []) if r.get('image_sha256') == proof['parent']['sha256']
             and r.get('decision') == 'hold']
    if not holds or proof.get('parent_refusal_reason') != holds[-1]['reason'] \
            or not isinstance(proof.get('excluded_defects'), list) or not proof['excluded_defects'] \
            or not all(isinstance(s, str) and s.strip() for s in proof['excluded_defects']):
        raise ValueError('editorial crop must preserve its parent refusal and excluded defects')
    box = proof.get('crop_ltrb')
    with Image.open(parent) as src:
        if src.size != size or not isinstance(box, list) or len(box) != 4 \
                or any(type(n) is not int for n in box) \
                or not 0 <= box[0] < box[2] <= src.width or not 0 <= box[1] < box[3] <= src.height \
                or box == [0, 0, src.width, src.height]:
            raise ValueError('editorial crop bounds must exclude part of the original image')
        expected = src.convert('RGB').crop(box).resize(size, Image.Resampling.LANCZOS)
    with Image.open(bound(row)) as actual:
        if actual.size != size or actual.convert('RGB').tobytes() != expected.tobytes():
            raise ValueError('editorial crop pixels differ from the exact parent transformation')
    crop_reviews = json.loads(bound(proof['crop_reviews']).read_text())
    exact = [r for r in crop_reviews.get('images', []) if r.get('image_sha256') == row['sha256']]
    if not any(r.get('decision') == 'pass' for r in exact) or any(r.get('decision') == 'hold' for r in exact):
        raise ValueError('editorial crop lacks a direct review of the exact derived pixels')
    return {'receipt_sha256': binding['sha256'], 'operation': proof['operation'],
            'parent_sha256': proof['parent']['sha256'], 'crop_ltrb': box,
            'parent_refusal_reason': proof['parent_refusal_reason'],
            'excluded_defects': proof['excluded_defects']}
