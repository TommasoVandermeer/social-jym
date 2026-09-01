#!/usr/bin/env python3
"""Generate a JESSI-S2R randomization manifest from existing robot logs."""

import argparse

from socialjym.utils.sim2real_calibration import (
    calibrate_controller_files,
    write_calibration_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="controller.pkl files")
    parser.add_argument("--output", required=True, help="output JSON manifest")
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--max-delay-steps", type=int, default=12)
    args = parser.parse_args()
    calibration = calibrate_controller_files(
        args.inputs,
        margin=args.margin,
        max_delay_steps=args.max_delay_steps,
    )
    write_calibration_json(calibration, args.output)
    print(f"Wrote calibration manifest to {args.output}")


if __name__ == "__main__":
    main()
