"""Local Mandarin evaluation; no included datasets and no cloud cleanup."""
import argparse
import json
import time

from lipflow.bench import wer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('video')
    p.add_argument('--reference', required=True)
    p.add_argument('--mode', choices=['silent', 'whisper'], default='silent')
    p.add_argument('--mouth-roi', action='store_true')
    p.add_argument('--whisper-model', default='large-v3-turbo')
    args = p.parse_args()
    t = time.monotonic()
    if args.mode == 'whisper':
        from lipflow.av import load_audio
        from lipflow.whisper import ChineseWhisper
        hs = ChineseWhisper(args.whisper_model).hypotheses(load_audio(args.video, 0, 60))
        raw = hs[0].text if hs else ''
    else:
        from lipflow.offline import transcribe_file
        from lipflow.vsr import LipReader
        raw = transcribe_file(args.video, LipReader(language='zh', beam_size=10, personal=False),
                              mouth_roi=args.mouth_roi)
    errors, chars = wer(raw, args.reference)
    print(json.dumps({'raw': raw, 'reference': args.reference, 'character_errors': errors,
                      'reference_characters': chars, 'cer': errors/max(chars,1),
                      'seconds_including_load': time.monotonic()-t, 'mode': args.mode}, ensure_ascii=False))


if __name__ == '__main__':
    main()
