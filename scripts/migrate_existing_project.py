"""Non-destructive migration of local data and measured outputs from an old project.

Never migrate the old virtual environment, source modules, or Git metadata.
Use --yes to make copies; without it, the script only previews the plan.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def migrate(old: Path, new: Path, *, apply: bool = False) -> dict:
    old, new = old.expanduser().resolve(), new.expanduser().resolve()
    if not old.is_dir() or old == new:
        raise ValueError("Provide a different existing --from-dir folder")
    counts = {"copied": 0, "already_present": 0, "conflicts": [], "planned": 0}
    for relative in ("data/raw", "data/processed", "results"):
        src_folder = old / relative
        if not src_folder.exists():
            continue
        for src in sorted(p for p in src_folder.rglob("*") if p.is_file()):
            if src.is_symlink():
                raise ValueError(f"Symbolic links are not migrated: {src}")
            rel = src.relative_to(old)
            dst = new / rel
            counts["planned"] += 1
            if dst.exists():
                if dst.stat().st_size == src.stat().st_size:
                    counts["already_present"] += 1
                else:
                    counts["conflicts"].append(str(rel))
                continue
            if apply:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                counts["copied"] += 1
    if counts["conflicts"]:
        raise ValueError("Existing target files differ; migration did not overwrite: " + ", ".join(counts["conflicts"]))
    # Rewrite manifest paths only after copying, and only if we created a fresh copy.
    source_manifest = old / "data/raw/manifest.json"
    target_manifest = new / "data/raw/manifest.json"
    if apply and source_manifest.exists() and target_manifest.exists():
        pairs = json.loads(target_manifest.read_text())
        fixed=[]
        for item in pairs:
            record=[]
            for path in item:
                candidate=Path(path).expanduser()
                if candidate.is_absolute() and candidate.is_relative_to(old):
                    relocated = new / candidate.relative_to(old)
                    record.append(str(relocated) if relocated.exists() else path)
                else:
                    record.append(path)
            fixed.append(record)
        target_manifest.write_text(json.dumps(fixed, indent=2) + "\n", encoding="utf8")
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-dir", type=Path, required=True)
    p.add_argument("--yes", action="store_true", help="Copy files rather than only preview")
    args=p.parse_args()
    print(json.dumps(migrate(args.from_dir, ROOT, apply=args.yes), indent=2))
    if not args.yes:
        print("Dry run only. Add --yes to copy data and results.")

if __name__ == '__main__': main()
