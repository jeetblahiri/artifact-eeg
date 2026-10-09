"""Run the known-source counterexample without auxiliary datasets or fits."""
import csv
from common import ROOT, sha256, write_json
from target_validity import known_source, settings


def main():
    result = known_source()
    out = ROOT / "results" / "target_validity"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "known_source.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    write_json(out / "known_source_only.json", {
        "known_source": result,
        "config": settings(),
        "generator_sha256": sha256(ROOT / "src" / "target_validity.py"),
        "config_sha256": sha256(ROOT / "target_validity_config.json"),
        "scope": "Known stipulated sources; no actual benchmark impurity estimate."
    })
    print("Completed known-source control; no external recordings used.")


if __name__ == "__main__":
    main()
