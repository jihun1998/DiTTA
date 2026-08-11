"""Minimal config loading: a config file is a python file defining ``cfg``."""

import os
import runpy


def load_config(path, overrides=None):
    """Execute a config file and return its ``cfg`` dict.

    ``overrides`` is a flat mapping whose keys may use dots to address nested
    entries, e.g. ``{'tta.lr': 1e-4}``.  Only existing keys can be overridden,
    so a typo fails loudly instead of being silently ignored.
    """
    path = os.path.abspath(path)
    ns = runpy.run_path(path)
    if 'cfg' not in ns:
        raise KeyError(f'{path} does not define `cfg`')
    cfg = ns['cfg']
    cfg['_config_path'] = path

    for key, value in (overrides or {}).items():
        if value is None:
            continue
        node, *rest = key.split('.')
        target = cfg
        while rest:
            if node not in target:
                raise KeyError(f'unknown config key: {key}')
            target = target[node]
            node, *rest = rest
        if node not in target:
            raise KeyError(f'unknown config key: {key}')
        target[node] = value
    return cfg


def format_config(cfg, indent=0):
    """Readable dump for the run log."""
    lines = []
    for key, value in cfg.items():
        if isinstance(value, dict):
            lines.append(' ' * indent + f'{key}:')
            lines.append(format_config(value, indent + 2))
        else:
            lines.append(' ' * indent + f'{key}: {value}')
    return '\n'.join(lines)
