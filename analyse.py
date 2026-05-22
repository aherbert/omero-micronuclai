#!/usr/bin/env python3
"""Analyse images using micronuclAI."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Analyse images using micronuclAI.")
    _ = parser.add_argument(
        "image",
        nargs="+",
        help="Image",
    )
    _ = parser.add_argument(
        "--mask",
        default="_cp_masks.tif",
        help="Suffix for corresponding mask (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--device",
        help="Torch device (default to auto-detect)",
    )
    _ = parser.add_argument(
        "--overwrite",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing results (default: %(default)s)",
    )
    # Adapted from src.model.micronuclai_predict.py
    options = parser.add_argument_group(title="micronuclAI arguments")
    _ = options.add_argument(
        "-mod",
        "--model",
        dest="model",
        required=True,
        help="Pathway to prediction model.",
    )
    _ = options.add_argument(
        "-s",
        "--size",
        dest="size",
        action="store",
        default=(256, 256),
        type=int,
        nargs=2,
        help="Size of images for training (default: %(default)s)",
    )
    _ = options.add_argument(
        "-rf",
        "--resizing-factor",
        dest="resizing_factor",
        default=0.6,
        help="Resizing factor for images (default: %(default)s)",
    )
    _ = options.add_argument(
        "-e",
        "--expansion",
        dest="expansion",
        default=25,
        type=int,
        help="Expansion factor for images (default: %(default)s)",
    )
    _ = options.add_argument(
        "-log",
        "--log-level",
        dest="log_level",
        default="info",
        choices=["debug", "info"],
        help="Set the logging level (default: %(default)s)",
    )
    args = parser.parse_args()

    import os
    import time

    import torch
    from src.model.augmentations import get_transforms
    from src.model.dataset import micronuclAI_inference
    from src.model.logger import set_logger
    from src.model.micronuclai_predict import inference, summarize
    from torch.utils.data import DataLoader

    log = set_logger(log_level=args.log_level)

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    log.info(f"Using device = {device}")

    log.info("Loading model")
    model = torch.jit.load(args.model)
    # Load data transformations
    transform = get_transforms(resize=args.size, training=False, prediction=True)

    start = time.time()

    # Process all images
    for i, image_file in enumerate(args.image):
        log.info(f"Image [{i + 1}/{len(args.image)}] {image_file}")
        # Find mask
        basename, ext = os.path.splitext(image_file)
        mask_file = basename + args.mask

        if not os.path.exists(mask_file):
            log.warning("No mask image %s", mask_file)
            continue
        log.info("Mask image %s", mask_file)

        # Skip existing output files
        prediction_file = f"{basename}_predictions.csv"
        if os.path.exists(prediction_file) and not args.overwrite:
            log.info(f"Skipping {image_file} (already processed)")
            continue

        # Run micronuclAI
        # Adapted from src.model.micronuclai_predict.py

        # Load dataset
        dataset = micronuclAI_inference(
            image_file,
            mask_file,
            resizing_factor=args.resizing_factor,
            expansion=args.expansion,
            size=args.size,
            transform=transform,
        )

        # Create a data loader
        loader = DataLoader(dataset, batch_size=1, num_workers=0)

        # Inference step
        log.info("Predicting micronuclei")
        df_predictions = inference(model, loader, device, log=log)

        # Get micronuclei group by counts
        log.info("Calculating summary")
        df_mn_counts = df_predictions["micronuclei"].value_counts()

        # Get summary
        log.info("Summarizing predictions")
        df_summary = summarize(df_mn_counts, log=log)

        # Save predictions
        log.info("Saving predictions")
        df_predictions.to_csv(prediction_file, index=False)
        df_mn_counts.to_csv(f"{basename}_counts.csv", index=True)
        df_summary.to_csv(f"{basename}_summary.csv", index=True, header=False)

    log.info(f"Image processing time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
