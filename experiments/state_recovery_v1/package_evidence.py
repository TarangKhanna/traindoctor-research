"""Archive one generated run without local stderr paths or cache files."""
import argparse
import hashlib
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    checksum = args.out.with_suffix(args.out.suffix + ".sha256")
    if args.out.exists() or checksum.exists():
        ap.error("refusing to overwrite an evidence archive")
    files = [p for p in sorted(args.run.rglob("*")) if p.is_file() and not p.is_symlink()
             and "stderr" not in p.parts and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    with zipfile.ZipFile(args.out, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(args.run)))
    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    checksum.write_text(f"{sha}  {args.out.name}\n")
    print("Packaged", len(files), "generated evidence files; sha256", sha)


if __name__ == "__main__":
    main()
