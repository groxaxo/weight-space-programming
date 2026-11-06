"""Frozen recipe contracts shared by training and atomic checkpoint export."""
from pathlib import Path
import shutil
from coder_common import check, digest, loads


def capture_recipe(repository, output, config):
    repository, output, config = Path(repository), Path(output), Path(config)
    files = {str(p.relative_to(repository)): p for folder in ('scripts', 'ops')
             for p in sorted((repository / folder).glob('*.py'))}
    files['recipe-config.json'] = config
    target = output / 'source'
    target.mkdir()
    hashes = {}
    for name, path in files.items():
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        hashes[name] = digest(path)
        shutil.copyfile(path, destination)
        check(digest(destination) == hashes[name], 'Recipe changed during capture: ' + name)
    return hashes


def verify_recipe(training):
    training = Path(training)
    run = loads((training / 'run-manifest.json').read_text())
    hashes = run.get('recipe_sha256', {})
    check(bool(hashes), 'Missing frozen training recipe')
    for name, expected in hashes.items():
        relative = Path(name)
        check(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe recipe filename')
        source = training / 'source' / relative
        # Historical runs captured flat source files. Authenticate, never recreate them.
        if not source.exists():
            source = training / 'source' / relative.name
        check(source.is_file() and not source.is_symlink() and digest(source) == expected,
              'Missing or changed frozen training recipe: ' + name)
    return run
