"""Select AstraLux FITS files and generate a reviewable rename script.

By default, only write rename.bash. Pass --run to execute the rename plan.
Requires numpy and astropy.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
from astropy.io import fits


def parse_args():
    """Read command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Folder containing ast*.fits files")
    parser.add_argument("--run", action="store_true", help="Execute the rename plan")
    parser.add_argument("--trash", action="store_true",
                        help="Move unselected ast*.fits files to the input folder's trash directory")
    parser.add_argument("--script", type=Path, default=Path("rename.bash"),
                        help="Output script path (default: rename.bash)")
    parser.add_argument("--verbose", action="store_true", help="Show every candidate")
    parser.add_argument("--quiet", action="store_true", help="Show only the summary")
    return parser.parse_args()


def build_plan(folder, verbose=False):
    """Choose the file with the fewest bright pixels per prefix and selection."""
    groups = defaultdict(list)
    files = sorted(folder.glob("ast*.fits"))

    for source in files:
        # Preserve the original grouping: remove '.fits' and one final character.
        prefix = source.name[:-6]
        with fits.open(source) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            selection = header["SELECTIO"]
            filename = str(header["FILENAME"]).strip()
            if not filename or filename in {".", ".."} or Path(filename).name != filename:
                raise ValueError(f"{source.name}: FILENAME must be a plain filename")
            if data is None or data.size == 0:
                raise ValueError(f"{source.name}: the primary HDU contains no image")
            # Keep the original threshold and count pixels above 90% of the peak.
            bright_pixels = int(np.count_nonzero(data > 0.9 * np.max(data)))

        destination = folder / filename
        groups[(prefix, selection)].append((bright_pixels, source, destination))
        if verbose:
            print(f"  Candidate: {source.name} | selection={selection} | "
                  f"bright pixels={bright_pixels}")

    plan = []
    for candidates in groups.values():
        # Sorted input makes ties reproducible: the first filename wins.
        _, source, destination = min(candidates, key=lambda item: item[0])
        plan.append((source, destination))
    selected = {source for source, _ in plan}
    unselected = [source for source in files if source not in selected]
    return len(files), plan, unselected


def validate_plan(plan, script):
    """Reject conflicting destinations before writing or executing any commands."""
    destinations = set()
    for source, destination in plan:
        if destination in destinations:
            raise ValueError(f"Multiple selected files target {destination.name}")
        destinations.add(destination)
        if source != destination and (destination.exists() or destination.is_symlink()):
            raise FileExistsError(f"Destination already exists: {destination}")
        if script in {source, destination}:
            raise ValueError("The output script cannot be a source or destination file")


def write_script(script, changes, trash_dir=None):
    """Quote paths safely and stop if a destination appears before execution."""
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    if trash_dir is not None:
        lines.append(f"mkdir -p -- {shlex.quote(str(trash_dir))}")
    for source, destination in changes:
        src, dst = shlex.quote(str(source)), shlex.quote(str(destination))
        message = shlex.quote(f"Destination already exists: {destination}")
        lines.extend([
            f"if [[ -e {dst} || -L {dst} ]]; then",
            f"  printf '%s\\n' {message} >&2",
            "  exit 1",
            "fi",
            f"mv -n -- {src} {dst}",
            # Detect a skipped move even on systems where mv -n returns success.
            f"if [[ -e {src} || -L {src} ]]; then",
            "  printf '%s\\n' 'Rename failed; stopping.' >&2",
            "  exit 1",
            "fi",
            "",
        ])
    script.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Build, display and optionally execute the complete rename plan."""
    args = parse_args()
    folder = args.path.expanduser().resolve()
    script = args.script.expanduser().resolve()
    try:
        if not folder.is_dir():
            raise NotADirectoryError(f"Folder not found: {folder}")
        scanned, plan, unselected = build_plan(folder, verbose=args.verbose and not args.quiet)
        trash_dir = folder / "trash"
        # Preserve selected files even when they already have the desired name.
        trash_moves = [(source, trash_dir / source.name) for source in unselected] if args.trash else []
        if trash_moves:
            if trash_dir.is_symlink() or (trash_dir.exists() and not trash_dir.is_dir()):
                raise ValueError(f"Trash must be a directory, not a file or symlink: {trash_dir}")
            if script == trash_dir or any(destination == trash_dir for _, destination in plan):
                raise ValueError("The trash directory conflicts with the script or a rename destination")
        validate_plan(plan + trash_moves, script)
        changes = [(src, dst) for src, dst in plan if src != dst]
        # Rename selected files first, then move only the original unselected files.
        write_script(script, changes + trash_moves, trash_dir if trash_moves else None)

        if not args.quiet:
            for source, destination in changes:
                print(f"  {source.name} -> {destination.name}")
            for source, destination in trash_moves:
                print(f"  {source.name} -> trash/{destination.name}")

        if args.run:
            subprocess.run(["bash", str(script)], check=True)

        action = "Renamed" if args.run else "Planned"
        summary = f"{action}: {len(changes)} | Scanned: {scanned} | Selected: {len(plan)}"
        if args.trash:
            trash_action = "Moved to trash" if args.run else "Planned to trash"
            summary += f" | {trash_action}: {len(trash_moves)}"
        print(summary)
        if not args.quiet:
            print(f"Script: {script}")
            if not args.run:
                print("Preview only. Use --run to apply these changes.")
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
