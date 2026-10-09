"""Reproducible illustration of reference-plus-artifact mixing; no model fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import ROOT, config, normalize, sha256, write_json

# Fixed illustrative rows, selected with seed 20261009 from the already frozen
# held-out component pools. These choices do not enter model fitting or scoring.
ROWS = {"EEG": 927, "EOG": 994, "EMG": 2245}
EXPECTED = {
    "EEG": "89350579007ab18daf918a26c9f730484ba3755e861bbf8a6f68d1657843a748",
    "EOG": "fe31f2c22efce5da7488a1047b6f05d678e80f48d22e20a70972aaf13784fa0a",
    "EMG": "8bb16dff87582823488dd2189dba9d8fa019a8cbc9af9a8a19acb421b3fcfb16",
}
FS = 256
SNR_DB = 0.0


def build_example(source_dir=None):
    source = Path(source_dir if source_dir is not None else config()["source_dir"])
    if not source.is_absolute():
        source = ROOT / source
    waves, sources = {}, {}
    for kind, row in ROWS.items():
        path = source / f"{kind}_all_epochs.npy"
        digest = sha256(path)
        if digest != EXPECTED[kind]:
            raise ValueError(f"Source checksum mismatch for {kind}; use the pinned public NumPy bank")
        bank = np.load(path, mmap_mode="r")
        if bank.ndim != 2 or bank.shape[1] != 512:
            raise ValueError(f"Unexpected {kind} source shape")
        waves[kind] = normalize(np.asarray(bank[row])[None])[0]
        sources[kind] = {"filename": path.name, "original_row_zero_based": row,
                         "source_sha256": digest,
                         "source_url": f"https://github.com/ncclabsustech/EEGdenoiseNet/blob/master/data/{path.name}"}
    reference = waves["EEG"]
    result = {"time_s": np.arange(512) / FS, "reference": reference}
    mixtures = {}
    for kind in ["EOG", "EMG"]:
        artifact = waves[kind]
        coefficient = np.sqrt(np.mean(reference**2) / np.mean(artifact**2)) * 10**(-SNR_DB / 20)
        scaled = coefficient * artifact
        mixed = reference + scaled
        measured_snr = 10 * np.log10(np.sum(reference**2) / np.sum(scaled**2))
        assert np.allclose(mixed - reference, scaled, atol=1e-14)
        assert abs(measured_snr - SNR_DB) < 1e-12
        result[kind.lower() + "_artifact"] = scaled
        result[kind.lower() + "_mixture"] = mixed
        mixtures[kind] = {"coefficient": float(coefficient), "power_snr_db": float(measured_snr)}
    limit = max(4, int(np.ceil(1.05 * max(np.max(abs(a)) for k, a in result.items() if k != "time_s"))))
    metadata = {"purpose": "Illustrative synthetic mixing, not a denoising result or neural-purity test",
                "selection_seed": 20261009, "sampling_hz": FS, "samples": 512,
                "release": "Pinned 256-Hz NumPy banks identified by SHA-256; not the separate 512-Hz release",
                "normalization": "Each source is mean centered and scaled to unit RMS before addition; no post-mixture scaling",
                "amplitude_units": "Dimensionless; reference RMS is one; original physical amplitude is not retained",
                "sources": sources, "mixtures": mixtures, "common_axis_limit": limit}
    return result, metadata


def table_text(example):
    keys = ["time_s", "reference", "eog_artifact", "emg_artifact", "eog_mixture", "emg_mixture"]
    return " ".join(keys) + "\n" + "\n".join(
        " ".join(f"{example[k][i]:.9f}" for k in keys) for i in range(512))


def inline_table_block():
    example, _ = build_example()
    return "\\pgfplotstableread{\n" + table_text(example) + "\n}\\pmix"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Directory containing the three pinned public source banks")
    args = parser.parse_args()
    example, metadata = build_example(args.source_dir)
    out = ROOT / "results" / "mixture_example"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plot_data.txt").write_text(table_text(example) + "\n")
    metadata["generator_sha256"] = sha256(Path(__file__))
    write_json(out / "metadata.json", metadata)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 240})
    fig, axes = plt.subplots(3, 2, figsize=(8.4, 5.0), sharex=True, sharey=True)
    titles = [["(a) EEG reference T", "(b) Same EEG reference T"],
              ["(c) Scaled ocular segment", "(d) Scaled muscle segment"],
              ["(e) EEG + ocular segment", "(f) EEG + muscle segment"]]
    for col, kind in enumerate(["eog", "emg"]):
        axes[0, col].plot(example["time_s"], example["reference"], color="#32373c", lw=.8)
        axes[1, col].plot(example["time_s"], example[kind + "_artifact"], color="#ad6e22", lw=.8)
        axes[2, col].plot(example["time_s"], example[kind + "_mixture"], color="#235f8c", lw=.8, label="Mixture X")
        axes[2, col].plot(example["time_s"], example["reference"], color="#32373c", lw=.7, ls="--", label="Reference T")
        axes[2, col].set_xlabel("Time (s)")
        for row in range(3):
            ax = axes[row, col]
            ax.set_title(titles[row][col], fontsize=10, loc="left")
            ax.set_xlim(0, 2); ax.set_ylim(-metadata["common_axis_limit"], metadata["common_axis_limit"])
            ax.set_xticks([0, .5, 1, 1.5, 2]); ax.set_yticks([-4, 0, 4])
            ax.grid(alpha=.15); ax.axhline(0, color="#bbbbbb", lw=.4)
    axes[2, 0].legend(loc="lower left", ncol=2, fontsize=8, frameon=False)
    fig.supylabel("Amplitude (normalized; reference RMS = 1)", fontsize=10)
    fig.tight_layout()
    figures = ROOT / "figures"; figures.mkdir(exist_ok=True)
    for suffix in ["svg", "png"]:
        fig.savefig(figures / f"mixture_construction.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print("Generated six-panel EEGdenoiseNet illustration; both additions verified at 0 dB power-SNR.")


if __name__ == "__main__":
    main()
