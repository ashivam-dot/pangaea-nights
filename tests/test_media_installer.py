"""Exercise stalled-mirror recovery without installing packages or using the network."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / '.github/scripts/install-media-tools.sh'


def run_installer(tmp_path, *, installed=False, final_missing=False):
    bins = tmp_path / 'bin'
    bins.mkdir()
    for command in ('mktemp', 'chmod', 'cat', 'rm'):
        (bins / command).symlink_to(shutil.which(command))
    scripts = {
        'timeout': '#!/bin/sh\nshift\nexec "$@"\n',
        'sudo': '#!/bin/sh\nexec "$@"\n',
        'lsb_release': '#!/bin/sh\nif [ "$1" = -is ]; then echo Ubuntu; else echo noble; fi\n',
        'apt-get': f'''#!{sys.executable}
import json,os,sys
from pathlib import Path
root=Path(os.environ['INSTALL_TEST_ROOT']);args=sys.argv[1:]
with (root/'calls').open('a') as f:f.write(json.dumps(args)+'\\n')
source=next((a.split('=',1)[1] for a in args if a.startswith('Dir::Etc::sourcelist=')),None)
if source:
    with (root/'sources').open('a') as f:f.write(Path(source).read_text())
if not source:sys.exit(124)
if 'install' in args and os.environ.get('FINAL_MISSING')!='1':
    for name in ['ffmpeg','ffprobe','tesseract']:
        p=root/'bin'/name;p.write_text('#!/bin/sh\\nexit 0\\n');p.chmod(0o755)
''',
    }
    for name, body in scripts.items():
        path = bins / name
        path.write_text(body)
        path.chmod(0o755)
    if installed:
        for name in ('ffmpeg', 'ffprobe', 'tesseract'):
            p = bins / name
            p.write_text('#!/bin/sh\nexit 0\n')
            p.chmod(0o755)
    env = dict(os.environ, PATH=str(bins), INSTALL_TEST_ROOT=str(tmp_path),
               FINAL_MISSING='1' if final_missing else '0')
    result = subprocess.run(['/bin/bash', str(SCRIPT), 'ffmpeg', 'tesseract-ocr'],
                            env=env, capture_output=True, text=True, timeout=10)
    calls = [json.loads(l) for l in (tmp_path/'calls').read_text().splitlines()] \
        if (tmp_path/'calls').exists() else []
    return result, calls


def test_stalled_runner_mirror_recovers_using_same_signed_archive_for_update_and_install(tmp_path):
    result, calls = run_installer(tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 3
    assert 'install' in calls[0] and 'update' in calls[1] and 'install' in calls[2]
    sources = [next(a for a in row if a.startswith('Dir::Etc::sourcelist=')) for row in calls[1:]]
    assert sources[0] == sources[1]
    source_text = (tmp_path/'sources').read_text()
    assert 'https://archive.ubuntu.com/ubuntu noble' in source_text
    assert 'signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg' in source_text
    assert 'trusted=yes' not in source_text
    assert not Path(sources[0].split('=', 1)[1]).exists()


def test_available_tools_need_no_package_operations(tmp_path):
    result, calls = run_installer(tmp_path, installed=True)
    assert result.returncode == 0 and not calls


def test_successful_package_command_cannot_hide_missing_tools(tmp_path):
    result, calls = run_installer(tmp_path, final_missing=True)
    assert result.returncode != 0 and len(calls) == 3
    assert 'still unavailable' in result.stderr
