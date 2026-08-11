"""Extract a few scalar curves from a TensorBoard run into a compact .npz.

Used to turn a reference run (e.g. the one the paper's numbers were taken from)
into something ``run_ditta.py --reference`` can plot alongside a live run.  The
original event files also carry per-class IoU curves and are gigabytes in size,
which is why this streams the file instead of loading it.

    python tools/export_reference_curves.py \
        --run /path/to/tb_run_dir --out reference/paper_q10.npz
"""

import argparse
import glob
import os

import numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

DEFAULT_TAGS = ('0_mIoU', '1_fwIoU', 'VC8', 'VC16')


def scalar_value(value):
    """Read a scalar summary, old style (``simple_value``) or new (``tensor``).

    ``SummaryWriter.add_scalar`` has written both over the years, and which one
    a reference run used depends on the torch version it was produced with.
    """
    if value.HasField('tensor'):
        tensor = value.tensor
        if tensor.float_val:
            return float(tensor.float_val[0])
        if tensor.double_val:
            return float(tensor.double_val[0])
        if tensor.tensor_content:
            return float(np.frombuffer(tensor.tensor_content, dtype=np.float32)[0])
    return float(value.simple_value)


def write_tb(npz_path, tb_dir, tags):
    """Re-emit the extracted curves as a small TensorBoard run."""
    from torch.utils.tensorboard import SummaryWriter
    data = np.load(npz_path)
    writer = SummaryWriter(tb_dir)
    for tag in tags:
        steps, values = data[f'{tag}/step'], data[f'{tag}/value']
        for step, value in zip(steps, values):
            writer.add_scalar(tag, float(value), global_step=int(step))
    writer.flush()
    writer.close()
    print(f'wrote TensorBoard run {tb_dir}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True, help='TensorBoard run directory')
    p.add_argument('--out', required=True, help='output .npz')
    p.add_argument('--tags', nargs='+', default=list(DEFAULT_TAGS))
    p.add_argument('--tb_out', default=None,
                   help='also write a compact TensorBoard run holding just these '
                        'curves, so it loads instantly next to a live run')
    p.add_argument('--from_npz', action='store_true',
                   help='skip reading --run and rewrite --tb_out from --out')
    args = p.parse_args()

    if args.from_npz:
        write_tb(args.out, args.tb_out, args.tags)
        return

    wanted = set(args.tags)
    steps = {t: [] for t in wanted}
    values = {t: [] for t in wanted}

    files = sorted(glob.glob(os.path.join(args.run, 'events.out.tfevents.*')))
    if not files:
        raise SystemExit(f'no event files in {args.run}')
    for path in files:
        print(f'reading {os.path.basename(path)} '
              f'({os.path.getsize(path) / 1e9:.2f} GB)', flush=True)
        for event in EventFileLoader(path).Load():
            for value in event.summary.value:
                if value.tag in wanted:
                    steps[value.tag].append(event.step)
                    values[value.tag].append(scalar_value(value))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    arrays = {}
    for tag in args.tags:
        arrays[f'{tag}/step'] = np.asarray(steps[tag], dtype=np.int64)
        arrays[f'{tag}/value'] = np.asarray(values[tag], dtype=np.float32)
        n = len(steps[tag])
        last = values[tag][-1] if n else float('nan')
        print(f'  {tag}: {n} points, last = {last:.6f}')
    np.savez_compressed(args.out, **arrays)
    print(f'wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)')
    if args.tb_out:
        write_tb(args.out, args.tb_out, args.tags)


if __name__ == '__main__':
    main()
