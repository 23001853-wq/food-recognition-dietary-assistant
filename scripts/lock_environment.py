"""Record the installed dependency closure; validate it with tests before accepting a lock."""

import argparse
import tomllib
from importlib.metadata import distribution
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extras", nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=Path("requirements-runtime-lock.txt"))
    args = parser.parse_args()
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = list(project["dependencies"])
    for extra in args.extras:
        requirements.extend(project["optional-dependencies"][extra])
    queue = [Requirement(req) for req in requirements]
    seen, versions = set(), {}
    while queue:
        req = queue.pop()
        key = (canonicalize_name(req.name), tuple(sorted(req.extras)))
        if key in seen:
            continue
        seen.add(key)
        dist = distribution(req.name)
        if req.specifier and not req.specifier.contains(dist.version):
            raise RuntimeError(f"Installed {dist.name} {dist.version} does not satisfy {req}")
        versions[dist.name] = dist.version
        for value in dist.requires or []:
            child = Requirement(value)
            if child.marker is None or any(
                child.marker.evaluate({"extra": extra}) for extra in ["", *req.extras]
            ):
                queue.append(child)
    args.output.write_text(
        "# Installed Python 3.11 dependency closure; see TEST_REPORT.md for verification.\n"
        + "\n".join(
            f"{name}=={version}"
            for name, version in sorted(versions.items(), key=lambda x: x[0].lower())
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Recorded {len(versions)} installed packages in {args.output}")


if __name__ == "__main__":
    main()
