from __future__ import annotations

import argparse
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
HORIZYN_REPO = "https://github.com/dayhofflabs/horizyn.git"
HORIZYN_COMMIT = "e6655e732f574c8bfa0488b9bc5068b67e382745"


@dataclass(frozen=True)
class Asset:
    group: str
    record: int
    zenodo_name: str
    target: str
    bytes: int
    md5: str
    sha256: str


ASSETS = [
    Asset(
        "reactzyme",
        11494913,
        "cleaned_uniprot_rhea.tsv",
        "data/external/reactzyme/cleaned_uniprot_rhea.tsv",
        87396123,
        "669bdd627c946114e87f06bffb4f33d9",
        "c2d1807562e1e296796820499733bdacab8d4e18ce753e499be558512a927359",
    ),
    Asset(
        "reactzyme",
        11494913,
        "rhea_molecules.tsv",
        "data/external/reactzyme/rhea_molecules.tsv",
        3497386,
        "cb5a575a08954f6d28311b9a4bef52fe",
        "98147c030b4f814da059cc9112b126f42f0239be93cf030f62103c35e122714e",
    ),
    Asset(
        "reactzyme",
        11494913,
        "enzyme_smi_split.zip",
        "data/external/reactzyme/enzyme_smi_split.zip",
        47503969,
        "e351fdb85830968fc9abe933c39f9eda",
        "80e00f03c8d934af20cab872c9b1df00f093c47524a00b1af8d7743d91841312",
    ),
    Asset(
        "horizyn",
        20348783,
        "horizyn_v1_0_dev.ckpt",
        "external/horizyn/checkpoints/horizyn_v1_0_dev.ckpt",
        201401250,
        "5b1f938f8b0a82fbe91892a3b4e2bf2c",
        "31bb9b6d73241b7807050377799de8b4bfb17f42a6cd652c8b17b65faf754c25",
    ),
    Asset(
        "horizyn",
        17957034,
        "train_rxns.csv",
        "external/horizyn/data/sota/train_rxns.csv",
        3411244,
        "7b0335ac694e4afee87e7a0a970f56e4",
        "8ffdb54bf6847c8c6e7e97545a4e455eab0a2ff8f427d01de2a50e65015b96d0",
    ),
    Asset(
        "horizyn",
        17957034,
        "test_rxns.csv",
        "external/horizyn/data/sota/test_rxns.csv",
        287620,
        "a45305ba22d4077d7a3f07d5f5d93ff5",
        "9596d2bb4c2d4033c5fdada6c563a186e235fbbdb13e06c0d59e569fe077f514",
    ),
]


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(asset: Asset) -> tuple[bool, str]:
    path = ROOT / asset.target
    if not path.is_file():
        return False, "missing"
    if path.stat().st_size != asset.bytes:
        return False, f"bytes={path.stat().st_size} expected={asset.bytes}"
    actual_md5 = digest(path, "md5")
    if actual_md5 != asset.md5:
        return False, f"md5={actual_md5} expected={asset.md5}"
    actual_sha = digest(path, "sha256")
    if actual_sha != asset.sha256:
        return False, f"sha256={actual_sha} expected={asset.sha256}"
    return True, "ok"


def ensure_horizyn_repo(verify_only: bool) -> None:
    root = ROOT / "external/horizyn"
    git_dir = root / ".git"
    if verify_only:
        if not git_dir.is_dir():
            raise RuntimeError("Horizyn source checkout missing")
        current = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if current != HORIZYN_COMMIT:
            raise RuntimeError(f"Horizyn source commit drift: {current} != {HORIZYN_COMMIT}")
        return
    if not git_dir.is_dir():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", HORIZYN_REPO, str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "fetch", "--depth", "1", "origin", HORIZYN_COMMIT], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "--detach", HORIZYN_COMMIT], check=True)


def download(asset: Asset) -> None:
    target = ROOT / asset.target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://zenodo.org/api/records/{asset.record}/files/{asset.zenodo_name}/content"
    temp = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temp.open("wb") as handle:
            for chunk in response.iter_content(1 << 20):
                if chunk:
                    handle.write(chunk)
    ok, detail = verify_temp(asset, temp)
    if not ok:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded asset failed verification: {asset.target}: {detail}")
    temp.replace(target)


def verify_temp(asset: Asset, path: Path) -> tuple[bool, str]:
    if path.stat().st_size != asset.bytes:
        return False, f"bytes={path.stat().st_size} expected={asset.bytes}"
    actual_md5 = digest(path, "md5")
    if actual_md5 != asset.md5:
        return False, f"md5={actual_md5} expected={asset.md5}"
    actual_sha = digest(path, "sha256")
    if actual_sha != asset.sha256:
        return False, f"sha256={actual_sha} expected={asset.sha256}"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore hash-locked third-party inputs needed for the current scientific reproduction chain.")
    parser.add_argument("--group", action="append", choices=["reactzyme", "horizyn", "all"], default=[])
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    groups = set(args.group or ["all"])
    if "all" in groups:
        groups = {"reactzyme", "horizyn"}
    if "horizyn" in groups:
        ensure_horizyn_repo(args.verify_only)
    failures: list[str] = []
    for asset in ASSETS:
        if asset.group not in groups:
            continue
        ok, detail = verify(asset)
        if not ok and not args.verify_only:
            download(asset)
            ok, detail = verify(asset)
        print(f"{asset.group}\t{asset.target}\t{detail}")
        if not ok:
            failures.append(f"{asset.target}: {detail}")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
