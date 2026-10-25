"""Compare unprompted recognizer settings on exact failed production WAVs; no release or threshold changes."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from studio.ladder_voice import wer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('voice', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    report = json.loads((a.voice / 'voice.json').read_text())
    result = {'platform': platform.platform(), 'source_report_sha256': hashlib.sha256(
        (a.voice / 'voice.json').read_bytes()).hexdigest(), 'release_approval': False, 'variants': []}
    a.out.mkdir(parents=True, exist_ok=True)
    for threads in [4, 2]:
        model = WhisperModel('base.en', device='cpu', compute_type='int8', cpu_threads=threads)
        for mode in ['production', 'greedy', 'quiet-margin']:
            rows = []
            for line, segment in report['segments'].items():
                path = a.voice / f'{line}.wav'
                audio, sr = sf.read(path, dtype='float32')
                wav = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                if mode == 'quiet-margin':
                    wav = np.pad(wav, (2400, 2400))
                settings = {'language': 'en', 'beam_size': 1, 'vad_filter': False,
                            'condition_on_previous_text': False}
                if mode == 'greedy':
                    settings['temperature'] = 0.0
                decoded, _ = model.transcribe(wav, **settings)
                heard = ' '.join(s.text for s in decoded).strip()
                rows.append({'line': line, 'expected': segment['text'], 'heard': heard,
                             'wer': wer(segment['text'], heard),
                             'audio_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            result['variants'].append({'cpu_threads': threads, 'mode': mode, 'lines': rows})
            (a.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({'cpu_threads': threads, 'mode': mode,
                              'bad_lines': [r for r in rows if r['wer'] > .25]}), flush=True)
        del model


if __name__ == '__main__':
    main()
