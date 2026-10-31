"""Build an allowlisted source ZIP; never copy credentials or experiments."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ('Dockerfile', '.dockerignore', 'requirements.txt', 'README.md',
         'VALIDATION.md', 'RULES.md', 'FAQ.md',
         'chameleon_design_schema_v0.9.json')


def main():
    paths = [ROOT / name for name in FILES]
    for folder in ('engine', 'renderer', 'brand_kit', 'input'):
        paths.extend(p for p in (ROOT / folder).rglob('*') if p.is_file())
    paths.extend(p for p in (ROOT / 'tests').rglob('*') if p.is_file())
    paths.extend(ROOT / 'tools' / name for name in ('synthetic.py', 'validate.py', 'package_submission.py'))
    paths.extend(p for p in (ROOT / 'output').glob('*_output.*') if p.suffix in ('.json', '.png'))
    paths = sorted(set(p for p in paths if not p.is_symlink() and
                       '__pycache__' not in p.parts and p.suffix != '.pyc' and
                       (p.name == '.dockerignore' or not any(part.startswith('.')
                        for part in p.relative_to(ROOT).parts))))
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing:
        raise ValueError(f'Missing required submission files: {missing}')
    samples = sorted((ROOT / 'input').glob('*.json'))
    for sample in samples:
        for suffix in ('.json', '.png'):
            if ROOT / 'output' / (sample.stem + '_output' + suffix) not in paths:
                raise ValueError(f'Missing sample output for {sample.name}: {suffix}')
    payload = {str(p.relative_to(ROOT)): p.read_bytes() for p in paths}
    payload['output/.gitkeep'] = b''
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(payload.items())}
    payload['MANIFEST.sha256.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    destination = ROOT / 'dist'
    destination.mkdir(exist_ok=True)
    archive = destination / 'chameleon-submission.zip'
    # Fixed metadata makes repeated packaging of identical files reproducible.
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo('chameleon-submission/' + name, date_time=(2026, 10, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_suffix('.zip.sha256')
    checksum.write_text(f'{digest}  {archive.name}\n')
    print(f'{archive}: {archive.stat().st_size:,} bytes, {len(payload)} files')
    print(f'SHA-256: {digest}')


if __name__ == '__main__':
    main()
