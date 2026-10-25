import io
import json
from argparse import Namespace
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from studio import config, ladder_cli, ladder_images as li, ladder_replay as replay, qa


@pytest.fixture
def cached(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    folder = tmp_path / ".github/replay"
    folder.mkdir(parents=True)
    job = {"id": "o3", "scene": "the original scene", "prompt": "exact painter prompt", "luma": (62, 198)}
    exposure = {"raw_p99": 200, "spread": 90}
    rows = []
    for take in range(2):
        path = folder / f"o3-{take}.png"
        Image.new("RGB", (8, 8), (180 + take, 0, 0)).save(path)
        rows.append({"file": path.name, "sha256": replay.digest(path), "prompt": job["prompt"],
                     "model": "test", "artifact_id": 99, "take": take, "exposure": exposure})
    source = {"run_id": 123, "sha": "a" * 40, "key": "original-key"}
    path = folder / "repair.json"
    path.write_text(json.dumps({"schema": 1, "premise": "how_deep-01", "source": source, "images": {"o3": rows}}))
    monkeypatch.setitem(li.sdcpp.CANDIDATES, "test", SimpleNamespace(width=8, height=8))
    monkeypatch.setattr(li.ladder, "image_jobs", lambda p: [job])
    monkeypatch.setattr(qa, "figure", lambda img: 0.0)
    monkeypatch.setattr(qa, "embed", lambda img: np.array([1.0, 0.0]))
    monkeypatch.setattr(li.plates, "save_plate", lambda img, stem, **kw: {
        "plate": stem.name + ".jpg", "depth": stem.name + ".depth.png"})
    return path, job, replay.load(".github/replay/repair.json"), exposure


def verdict(reasons=None):
    return qa.Verdict(not reasons, reasons or [], 0.0, 0.0, 0.0, 0.4, 6.0, 120.0, 50.0, 12.0)


@pytest.mark.parametrize('slots', [[], ['o4', 'o4'], ['unknown'], [True], 'o4'])
def test_art_only_slots_are_typed_and_bounded(cached, slots):
    path, _, _, _ = cached
    data = json.loads(path.read_text())
    data['art_only'] = slots
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='distinct known'):
        replay.load('.github/replay/repair.json')


def test_art_only_paints_only_requested_slots(cached, tmp_path, monkeypatch):
    _, job, data, _ = cached
    data['art_only'] = ['o3']
    monkeypatch.setattr(li.ladder, 'image_jobs', lambda p: [job, {**job, 'id': 'o4'}])
    monkeypatch.setattr(qa, 'review', lambda *a: verdict())
    monkeypatch.setattr(li.sdcpp, 'Server', lambda *a: pytest.fail('unrequested beat loaded a painter'))
    report = li.generate({'id': data['premise']}, 'original-key', tmp_path/'art', 0, 1,
                         model='test', replay=data)
    assert [entry['id'] for entry in report['images']] == ['o3']


def test_cloud_repair_keeps_actual_source_instead_of_original_run(cached):
    _, job, data, _ = cached
    origin = {'run_id': 456, 'sha': 'b' * 40, 'key': 'original-key'}
    data['images']['o3'][0]['source'] = origin
    assert replay.candidates(data, job, 'test', (8, 8))[0][3]['source'] == origin
    for broken in ({**origin, 'run_id': True}, {**origin, 'sha': 'wrong'}, {**origin, 'key': 'other'}):
        data['images']['o3'][0]['source'] = broken
        with pytest.raises(ValueError, match='exact cloud source'):
            replay.candidates(data, job, 'test', (8, 8))


def test_art_only_collects_no_video_and_cannot_publish(cached, tmp_path, monkeypatch):
    from studio import batch_release, shorts_ladder, youtube

    path, _, data, _ = cached
    raw = json.loads(path.read_text()); raw['art_only'] = ['o3']; path.write_text(json.dumps(raw))
    monkeypatch.setattr(config, 'WORK', tmp_path/'work')
    plan = {'mode': 'dry', 'premise': {'id': data['premise']},
            'image_replay': {'path': '.github/replay/repair.json', 'sha256': replay.digest(path)}}
    ladder_cli._save('A', 'plan.json', plan)
    ladder_cli._save('A', 'images/shard-00.json', {'images': [{'id': 'o3', 'ok': True,
                                                            'kept': [{'plate': 'o3.jpg'}]}]})
    image = ladder_cli._dir('A')/'images/o3.png'
    Image.new('RGB', (8, 8), 'red').save(image)
    monkeypatch.setattr(shorts_ladder, 'render', lambda *a: pytest.fail('art-only rendered a video'))
    monkeypatch.setattr(youtube, 'client', lambda *a: pytest.fail('art-only accessed YouTube'))
    ladder_cli.render(Namespace(slot='A'))
    gates = ladder_cli._load('A', 'out/gates.json')
    assert gates['ok'] is False and gates['art_only'] == ['o3']
    assert not list(config.WORK.rglob('*.mp4'))
    ladder_cli.publish(Namespace(slot='A', mode='dry'))
    assert ladder_cli._load('A', 'out/publish.json')['published'] is False
    for mode in ('publish', 'test'):
        with pytest.raises(SystemExit, match='cannot publish'):
            ladder_cli.publish(Namespace(slot='A', mode=mode))
    with pytest.raises(ValueError, match='failed original'):
        batch_release.validate({'premise': data['premise']}, config.WORK)
    image.unlink()
    with pytest.raises(ValueError, match='lacks passing'):
        ladder_cli.render(Namespace(slot='A'))
    plan['mode'] = 'publish'; ladder_cli._save('A', 'plan.json', plan)
    with pytest.raises(ValueError, match='dry-run only'):
        ladder_cli.render(Namespace(slot='A'))


def test_cached_alternates_are_reviewed_again_without_loading_a_painter(cached, tmp_path, monkeypatch):
    _, job, data, _ = cached
    seen = []
    def inspect(img, scene):
        seen.append(scene)
        return verdict()
    monkeypatch.setattr(qa, "review", inspect)
    monkeypatch.setattr(li.sdcpp, "Server", lambda *a: pytest.fail("unchanged pictures were repainted"))
    report = li.generate({"id": data["premise"]}, "original-key", tmp_path / "out", 0, 1, model="test", replay=data)
    assert seen == [job["scene"], job["scene"]]
    kept = report["images"][0]["kept"]
    assert [r["cached_from"]["take"] for r in kept] == [0, 1]
    assert all(r["cached_from"]["source"]["run_id"] == 123 for r in kept)
    assert report["load_s"] == 0


def test_cache_hashes_and_prompt_binding_cannot_be_ignored(cached):
    path, job, data, _ = cached
    (path.parent / "o3-0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="corrupt"):
        replay.candidates(data, job, "test", (8, 8))
    assert replay.candidates(data, {**job, "prompt": "repaired prompt"}, "test", (8, 8)) == []
    assert replay.candidates(data, job, "different model", (8, 8)) == []


def test_explicit_cached_only_repair_never_paints_unused_alternates(cached, tmp_path, monkeypatch):
    _, job, data, _ = cached
    data['cached_only'] = True
    data['images'][job['id']] = data['images'][job['id']][:1]
    monkeypatch.setattr(qa, 'review', lambda *a: verdict())
    monkeypatch.setattr(li.sdcpp, 'Server', lambda *a: pytest.fail('cache-only repair loaded a painter'))
    report = li.generate({'id': data['premise']}, 'original-key', tmp_path/'cache-only', 0, 1,
                         model='test', replay=data)
    assert report['cached_only'] and report['images'][0]['ok']
    assert len(report['images'][0]['kept']) == 1
    assert report['images'][0]['kept'][0]['cached_from']['take'] == 0


def test_cache_only_refusal_still_fails_and_changed_prompts_cannot_generate(cached, tmp_path, monkeypatch):
    _, job, data, _ = cached
    data['cached_only'] = True
    monkeypatch.setattr(qa, 'review', lambda *a: verdict(['broken 0.8']))
    monkeypatch.setattr(li.sdcpp, 'Server', lambda *a: pytest.fail('cache-only refusal loaded a painter'))
    report = li.generate({'id': data['premise']}, 'original-key', tmp_path/'bad-cache', 0, 1,
                         model='test', replay=data)
    assert not report['images'][0]['ok'] and not report['images'][0]['kept']
    job['prompt'] = 'changed prompt with no exact cache'
    with pytest.raises(ValueError, match='cache-only replay lacks exact inputs'):
        li.generate({'id': data['premise']}, 'original-key', tmp_path/'changed', 0, 1,
                    model='test', replay=data)


def test_cache_only_policy_is_typed_and_bound_to_the_manifest(cached):
    path, _, _, _ = cached
    raw = json.loads(path.read_text())
    raw['cached_only'] = 'true'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='explicit boolean'):
        replay.load('.github/replay/repair.json')


def test_direct_image_refusal_survives_a_clean_numeric_review(cached, tmp_path, monkeypatch):
    _, job, data, _ = cached
    row = data['images'][job['id']][1]
    row['editorial_refusal'] = {'image_sha256': row['sha256'], 'reason': 'hedgehog spines on a lemming'}
    monkeypatch.setattr(qa, 'review', lambda *a: verdict())
    monkeypatch.setattr(replay, 'exhausted', lambda *a: True)
    monkeypatch.setattr(li.sdcpp, 'Server', lambda *a: pytest.fail('exhausted inputs were repainted'))
    report = li.generate({'id': data['premise']}, 'original-key', tmp_path / 'refused-alt',
                         0, 1, model='test', replay=data)
    entry = report['images'][0]
    assert entry['ok'] and len(entry['kept']) == 1
    assert entry['kept'][0]['cached_from']['take'] == 0
    assert 'hedgehog' in entry['takes'][1]['problems'][0]
    row['editorial_refusal']['image_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='stale image-bound'):
        replay.candidates(data, job, 'test', (8, 8))


def test_a_current_qa_refusal_repaints_without_repeating_cached_seeds(cached, tmp_path, monkeypatch):
    _, _, data, exposure = cached
    seeds = []
    class Painter:
        load_seconds = 1
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def generate(self, prompt, seed, w, h):
            seeds.append(seed)
            blob = io.BytesIO()
            Image.new("RGB", (w, h), (0, 180, 0)).save(blob, format="PNG")
            return blob.getvalue()
    monkeypatch.setattr(li.sdcpp, "Server", lambda *a: Painter())
    monkeypatch.setattr(li, "expose", lambda img, target: (img, exposure))
    monkeypatch.setattr(qa, "review", lambda img, scene: verdict(["text 0.9"] if img.getpixel((0, 0))[0] else []))
    report = li.generate({"id": data["premise"]}, "original-key", tmp_path / "out", 0, 1, model="test", replay=data)
    assert seeds == [li._seed("original-key", "o3", k) for k in (2, 3)]
    entry = report["images"][0]
    assert entry["ok"] and len(entry["takes"]) == 4
    assert all("cached_from" not in row for row in entry["kept"])


def test_replay_manifest_is_pinned_and_confined_to_the_repo(cached, tmp_path):
    path, _, _, _ = cached
    sha = replay.digest(path)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after planning"):
        replay.load(".github/replay/repair.json", sha)
    with pytest.raises(ValueError, match="inside .github/replay"):
        replay.load(tmp_path / "elsewhere.json")


def test_an_image_replay_cannot_use_the_direct_publishing_mode(cached):
    with pytest.raises(ValueError, match="dry-run only"):
        ladder_cli.plan(Namespace(mode="publish", replay=".github/replay/repair.json", premise=""))


@pytest.mark.parametrize("cached_only,expected_shards", [(False, 9), (True, 1)])
def test_dry_plan_pins_the_manifest_and_selects_its_premise(tmp_path, monkeypatch, capsys,
                                                         cached_only, expected_shards):
    from studio import ledger

    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "WORK", tmp_path / "work")
    monkeypatch.setattr(ledger, "load", lambda: json.loads(json.dumps(ledger.EMPTY)))
    monkeypatch.setattr(li.ladder, "now_day", lambda: date(2030, 1, 2))
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    path = tmp_path / ".github/replay/repair.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": 1, "premise": "how_deep-01", "images": {}, "cached_only": cached_only,
                               "source": {"run_id": 123, "sha": "a" * 40, "key": "source-key"}}))
    ladder_cli.plan(Namespace(mode="dry", replay=".github/replay/repair.json", premise="", slot="A", shards=9))
    plan = json.loads((config.WORK / "ladder/A/plan.json").read_text())
    assert plan["premise"]["id"] == "how_deep-01" and plan["mode"] == "dry"
    assert plan["image_replay"] == {"path": ".github/replay/repair.json", "sha256": replay.digest(path)}
    # A replay crossing midnight keeps its proven seed history while calendar/rotation checks stay current.
    assert plan["key"] == "source-key" and plan["day"] == "2030-01-02"
    matrix = json.loads(next(line.removeprefix("images=") for line in capsys.readouterr().out.splitlines()
                             if line.startswith("images=")))
    assert matrix == [{"slot": "A", "shard": f"{i}/{expected_shards}"} for i in range(expected_shards)]


def test_original_exhausted_seeds_are_not_repainted_and_current_qa_still_binds(cached, tmp_path, monkeypatch):
    path, job, data, _ = cached
    data['images'][job['id']] = data['images'][job['id']][:1]
    report = path.parent / 'original-shard.json'
    report.write_text(json.dumps({'premise': data['premise'], 'model': 'test',
                                  'images': [{'id': job['id'], 'takes': [{}] * li.TAKES}]}))
    data['completed_images'] = {job['id']: {'file': report.name, 'sha256': replay.digest(report),
                                           'artifact_id': 99, 'prompt': job['prompt'], 'model': 'test'}}
    monkeypatch.setattr(li.sdcpp, 'Server', lambda *a: pytest.fail('exhausted deterministic seeds were repainted'))
    monkeypatch.setattr(qa, 'review', lambda *a: verdict())
    good = li.generate({'id': data['premise']}, 'original-key', tmp_path / 'good', 0, 1, model='test', replay=data)
    assert good['images'][0]['ok'] and len(good['images'][0]['kept']) == 1
    monkeypatch.setattr(qa, 'review', lambda *a: verdict(['text 0.9']))
    bad = li.generate({'id': data['premise']}, 'original-key', tmp_path / 'bad', 0, 1, model='test', replay=data)
    assert not bad['images'][0]['ok']
    assert not replay.exhausted(data, {**job, 'prompt': 'new scene'}, 'test', 'original-key', li.TAKES)
    assert not replay.exhausted(data, job, 'test', 'different-key', li.TAKES)
    report.write_text(report.read_text() + ' ')
    with pytest.raises(ValueError, match='original shard report'):
        replay.exhausted(data, job, 'test', 'original-key', li.TAKES)


def test_duplicate_cached_takes_cannot_count_as_alternatives(cached):
    _, job, data, _ = cached
    data['images'][job['id']][1]['take'] = data['images'][job['id']][0]['take']
    with pytest.raises(ValueError, match='distinct'):
        replay.candidates(data, job, 'test', (8, 8))


def test_opening_reprise_is_an_exact_scene_and_does_not_change_narration():
    from studio import ladder
    p=ladder.find(ladder.load_banks(strict=False),'how_deep-01')
    script=ladder.script_text(p)
    jobs={j['id']:j for j in ladder.image_jobs(p)}
    assert jobs['end']['prompt']==jobs['o1']['prompt']
    assert jobs['end']['scene']==jobs['o1']['scene']
    assert script==ladder.script_text({k:v for k,v in p.items() if k!='end_image'})
    with pytest.raises(ValueError,match='only the opening'):
        ladder.image_jobs({**p,'end_image':'o8'})


def test_winning_choice_reprise_preserves_answer_scene_and_narration():
    from studio import ladder

    original = ladder.find(ladder.load_banks(strict=False), 'countdown-03')
    p = {**original, 'end_image': f"o{original['answer_n']}"}
    jobs = {job['id']: job for job in ladder.image_jobs(p)}
    winning = jobs[p['end_image']]
    assert jobs['end'] == {**winning, 'id': 'end', 'n': 0, 'luma': (62.0, 198.0)}
    assert ladder.script_text(p) == ladder.script_text(original)
    assert not ladder.premise_problems(p)
    with pytest.raises(ValueError, match='only the opening image or the winning choice'):
        ladder.image_jobs({**p, 'end_image': 'o5'})


def test_end_reprise_keeps_honest_original_slot_and_one_currently_reviewed_image(cached, tmp_path, monkeypatch):
    _, original, data, _ = cached
    job={**original,'id':'end'}
    data['images']={'end':[{**data['images']['o3'][0],'source_image':'o3'}]}
    monkeypatch.setattr(li.ladder,'image_jobs',lambda p:[job])
    monkeypatch.setattr(qa,'review',lambda *a:verdict())
    monkeypatch.setattr(li.sdcpp,'Server',lambda *a:pytest.fail('valid end reprise was repainted'))
    report=li.generate({'id':data['premise']},'original-key',tmp_path/'end',0,1,model='test',replay=data)
    kept=report['images'][0]['kept']
    assert len(kept)==1 and kept[0]['cached_from']['source_image']=='o3'
    data['images']['end'][0]['source_image']='unknown'
    with pytest.raises(ValueError,match='original image slot'):
        replay.candidates(data,job,'test',(8,8))


@pytest.fixture
def local_cached(cached, monkeypatch):
    path, job, data, _ = cached
    cand = SimpleNamespace(width=8, height=8, steps=4, cfg=1.0, sampler='euler',
                           weights=[SimpleNamespace(repo='model/repo', file='weights.gguf')])
    monkeypatch.setitem(li.sdcpp.CANDIDATES, 'test', cand)
    raw_path = path.parent / 'raw.png'
    raw = Image.fromarray(np.arange(192, dtype=np.uint8).reshape((8, 8, 3)))
    raw.save(raw_path)
    graded, exposure = li.expose(raw, sum(job['luma']) / 2)
    row = data['images']['o3'][0]
    graded.save(path.parent / row['file'])
    row.update(sha256=replay.digest(path.parent / row['file']), exposure=exposure)
    row.pop('artifact_id')
    proof = {'schema': 1, 'origin': 'local-stable-diffusion.cpp', 'source_sha': 'a' * 40,
             'binary_tag': li.sdcpp.SD_TAG,
             'settings': {'model': 'test', 'prompt': job['prompt'], 'width': 8, 'height': 8,
                          'steps': 4, 'cfg': 1.0, 'sampler': 'euler',
                          'seed': li._seed('original-key', 'o3', 0)},
             'weights': [{'repo': 'model/repo', 'file': 'weights.gguf', 'sha256': 'b' * 64}],
             'raw': {'file': raw_path.name, 'sha256': replay.digest(raw_path)}}
    proof_path = path.parent / 'generation.json'
    proof_path.write_text(json.dumps(proof))
    row['generation'] = {'file': proof_path.name, 'sha256': replay.digest(proof_path)}
    data['images']['o3'] = [row]
    return path, job, data, row, proof, proof_path


def test_local_generation_is_explicit_and_reproduces_the_original_raw_grade(local_cached):
    _, job, data, _, _, _ = local_cached
    found = replay.candidates(data, job, 'test', (8, 8))
    source = found[0][3]['source']
    assert source['kind'] == 'local-stable-diffusion.cpp'
    assert 'artifact_id' not in found[0][3] and 'run_id' not in source


@pytest.fixture
def reference_cached(local_cached):
    path, job, data, row, proof, proof_path = local_cached
    ref = path.parent / 'reference.png'
    Image.new('RGB', (8, 8), 'green').save(ref)
    page = 'https://commons.wikimedia.org/wiki/File:Reference.png'
    url = 'https://upload.wikimedia.org/wikipedia/commons/a/ab/Reference.png'
    license_url = 'https://creativecommons.org/licenses/by/3.0/'
    credit = f'Reference: Artist, {page}, CC BY 3.0, {license_url}, AI-adapted setting.'
    evidence = path.parent / 'license.json'
    evidence.write_text(json.dumps({'query': {'pages': {'1': {
        'title': 'File:Reference.png', 'imageinfo': [{'url': url, 'descriptionurl': page,
        'extmetadata': {'Artist': {'value': '<b>Artist</b>'},
                        'LicenseShortName': {'value': 'CC BY 3.0'},
                        'LicenseUrl': {'value': license_url}}}]}}}}))
    proof.update(schema=2, origin='local-stable-diffusion.cpp-reference', conditioning={
        'mode': 'reference', 'endpoint': '/v1/images/edits',
        'request_extra_args': {'seed': proof['settings']['seed'], 'sample_params': {
            'sample_steps': 4, 'sample_method': 'euler', 'guidance': {'txt_cfg': 1.0}}},
        'references': [{'file': ref.name, 'sha256': replay.digest(ref), 'title': 'File:Reference.png',
                        'author': 'Artist', 'page': page, 'url': url, 'license': 'CC BY 3.0',
                        'license_url': license_url, 'credit': credit,
                        'license_evidence': {'file': evidence.name, 'sha256': replay.digest(evidence)}}]})
    proof_path.write_text(json.dumps(proof))
    row['generation']['sha256'] = replay.digest(proof_path)
    data['cached_only'] = True
    return local_cached, credit


def test_reference_pixels_and_license_survive_as_explicit_provenance(reference_cached, tmp_path, monkeypatch):
    (_, job, data, _, _, _), credit = reference_cached
    found = replay.candidates(data, job, 'test', (8, 8))
    assert found[0][3]['source']['kind'] == 'local-stable-diffusion.cpp-reference'
    assert found[0][3]['source']['required_credits'] == [credit]
    with pytest.raises(ValueError, match='attribution is missing'):
        li.generate({'id': data['premise']}, 'original-key', tmp_path/'uncredited', 0, 1, model='test', replay=data)
    monkeypatch.setattr(qa, 'review', lambda *a: verdict(['broken 0.8']))
    monkeypatch.setattr(li.sdcpp, 'Server', lambda *a: pytest.fail('reference gates loaded a painter'))
    report = li.generate({'id': data['premise'], 'description': credit}, 'original-key',
                         tmp_path/'credited-but-broken', 0, 1, model='test', replay=data)
    assert not report['images'][0]['ok']  # attribution never waives the current picture gates


@pytest.mark.parametrize('changed', ['reference_bytes', 'license', 'credit', 'request', 'text_only'])
def test_reference_conditioning_cannot_hide_unlicensed_or_changed_inputs(reference_cached, changed):
    (path, job, data, row, proof, proof_path), _ = reference_cached
    ref = proof['conditioning']['references'][0]
    if changed == 'reference_bytes':
        (path.parent / ref['file']).write_bytes(b'changed reference')
    elif changed == 'license':
        ref['license'] = 'public domain'
    elif changed == 'credit':
        ref['credit'] = 'Artist'
    elif changed == 'request':
        proof['conditioning']['request_extra_args']['seed'] += 1
    else:
        proof.update(schema=1, origin='local-stable-diffusion.cpp')
    proof_path.write_text(json.dumps(proof))
    row['generation']['sha256'] = replay.digest(proof_path)
    with pytest.raises(ValueError):
        replay.candidates(data, job, 'test', (8, 8))


def test_local_closing_reprise_keeps_original_seed_and_receives_current_checks(local_cached, tmp_path, monkeypatch):
    _, original, data, row, _, _ = local_cached
    job = {**original, 'id': 'end'}
    data['images']['end'] = [{**row, 'source_image': original['id']}]
    monkeypatch.setattr(li.ladder, 'image_jobs', lambda p: [job])
    seen = []
    monkeypatch.setattr(qa, 'review', lambda image, scene: seen.append(scene) or verdict())
    monkeypatch.setattr(li.sdcpp, 'Server', lambda *a: pytest.fail('exact local reprise was repainted'))
    report = li.generate({'id': data['premise']}, 'original-key', tmp_path / 'local-end', 0, 1,
                         model='test', replay=data)
    kept = report['images'][0]['kept']
    assert seen == [job['scene']] and len(kept) == 1
    assert kept[0]['cached_from']['source_image'] == original['id']
    data['images']['end'][0]['source_image'] = 'o4'
    with pytest.raises(ValueError, match='exact original image'):
        replay.candidates(data, job, 'test', (8, 8))


@pytest.mark.parametrize('changed', ['seed', 'prompt', 'weights', 'binary', 'raw', 'grade', 'exposure'])
def test_local_generation_cannot_adopt_changed_or_unproven_pixels(local_cached, changed):
    path, job, data, row, proof, proof_path = local_cached
    if changed in ('seed', 'prompt'):
        proof['settings'][changed] = 123 if changed == 'seed' else 'different scene'
    elif changed == 'weights':
        proof['weights'][0]['sha256'] = ''
    elif changed == 'binary':
        proof['binary_tag'] = 'unbenchmarked-tool'
    elif changed == 'raw':
        (path.parent / 'raw.png').write_bytes(b'corrupt')
    elif changed == 'grade':
        Image.new('RGB', (8, 8), 'white').save(path.parent / row['file'])
        row['sha256'] = replay.digest(path.parent / row['file'])
    else:
        row['exposure']['raw_p99'] += 1
    proof_path.write_text(json.dumps(proof))
    row['generation']['sha256'] = replay.digest(proof_path)
    with pytest.raises(ValueError):
        replay.candidates(data, job, 'test', (8, 8))


def test_local_candidate_still_receives_current_qa_and_cannot_bypass_refusal(local_cached, tmp_path, monkeypatch):
    _, job, data, _, _, _ = local_cached
    seen = []
    monkeypatch.setattr(qa, 'review', lambda image, scene: seen.append(scene) or verdict(['text 0.9']))
    monkeypatch.setattr(replay, 'exhausted', lambda *a: True)
    monkeypatch.setattr(li.sdcpp, 'Server', lambda *a: pytest.fail('a new painter was loaded'))
    report = li.generate({'id': data['premise']}, 'original-key', tmp_path / 'rejected-local', 0, 1,
                         model='test', replay=data)
    assert seen == [job['scene']] and not report['images'][0]['ok']


def test_a_single_local_candidate_does_not_claim_all_seeds_were_attempted(local_cached):
    _, job, data, row, _, _ = local_cached
    data['local_attempts'] = {job['id']: [row]}
    with pytest.raises(ValueError, match='every allowed deterministic take'):
        replay.exhausted(data, job, 'test', 'original-key', 4)


def test_local_exhaustion_checks_each_raw_and_seed_instead_of_trusting_a_summary(local_cached):
    path, job, data, row, proof, _ = local_cached
    from copy import deepcopy

    attempts = []
    for take in range(4):
        attempt = deepcopy(row)
        attempt['take'] = take
        receipt = deepcopy(proof)
        receipt['settings']['seed'] = li._seed('original-key', job['id'], take)
        proof_path = path.parent / f'proof-t{take}.json'
        proof_path.write_text(json.dumps(receipt))
        attempt['generation'] = {'file': proof_path.name, 'sha256': replay.digest(proof_path)}
        attempts.append(attempt)
    data['local_attempts'] = {job['id']: attempts}
    assert replay.exhausted(data, job, 'test', 'original-key', 4)
    (path.parent / 'raw.png').write_bytes(b'corrupt original')
    with pytest.raises(ValueError, match='corrupt'):
        replay.exhausted(data, job, 'test', 'original-key', 4)


def test_cached_exhausted_soft_only_picture_keeps_the_existing_fallback_and_its_source(cached,tmp_path,monkeypatch):
    _,job,data,_=cached
    monkeypatch.setattr(replay,'exhausted',lambda *a:True)
    def soft(*a):
        v=verdict(['mushy 7.1'])
        v.sharpness=7.1
        return v
    monkeypatch.setattr(qa,'review',soft)
    monkeypatch.setattr(li.sdcpp,'Server',lambda *a:pytest.fail('exhausted soft picture was repainted'))
    report=li.generate({'id':data['premise']},'original-key',tmp_path/'cached-soft',0,1,model='test',replay=data)
    entry=report['images'][0]
    assert entry['ok'] and len(entry['kept'])==1
    assert entry['kept'][0]['fallback'] is True
    assert entry['kept'][0]['cached_from']['source']['run_id']==123
    monkeypatch.setattr(qa,'review',lambda *a:verdict(['mushy 7.1','text 0.9']))
    refused=li.generate({'id':data['premise']},'original-key',tmp_path/'cached-unsafe',0,1,model='test',replay=data)
    assert not refused['images'][0]['ok']
