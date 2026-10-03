"""Time the read-only UtauTTS speech-timing train() loop on an existing cache."""
import argparse
import importlib.util
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--valid", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("speech_timing_reference", args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    data = torch.load(args.cache, weights_only=False, map_location="cpu")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    _, best, step = reference.train(data, args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"pytorch train_loop_seconds={time.perf_counter()-start:.3f} best_valid_l1={best:.6f} best_step={step}")


if __name__ == "__main__":
    main()
