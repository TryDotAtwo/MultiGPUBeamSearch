from pathlib import Path
import json
import sys


def write_zeros(path: Path, count: int) -> None:
    path.write_bytes(b"\0\0" * count)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dummy_weights")
    out.mkdir(parents=True, exist_ok=True)

    state_len = 72
    num_classes = 72
    hidden1 = 64
    hidden2 = 32
    residual_count = 1
    output_dim = 1

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "state_len": state_len,
                "num_classes": num_classes,
                "hidden1": hidden1,
                "hidden2": hidden2,
                "residual_count": residual_count,
                "output_dim": output_dim,
                "dtype": "fp16",
                "normalization": "none",
            }
        )
    )

    write_zeros(out / "input_weight_hxk.fp16", state_len * num_classes * hidden1)
    write_zeros(out / "input_bias.fp16", hidden1)
    write_zeros(out / "hidden_weight_hxk.fp16", hidden1 * hidden2)
    write_zeros(out / "hidden_bias.fp16", hidden2)
    write_zeros(out / "residual0_fc1_weight_hxk.fp16", hidden2 * hidden2)
    write_zeros(out / "residual0_fc1_bias.fp16", hidden2)
    write_zeros(out / "residual0_fc2_weight_hxk.fp16", hidden2 * hidden2)
    write_zeros(out / "residual0_fc2_bias.fp16", hidden2)
    write_zeros(out / "output_weight_hxk.fp16", hidden2 * output_dim)
    write_zeros(out / "output_bias.fp16", output_dim)


if __name__ == "__main__":
    main()
