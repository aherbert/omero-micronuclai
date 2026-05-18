#!/usr/bin/env python3
"""Analyse OMERO images using micronuclAI."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Analyse OMERO images using micronuclAI.")
    _ = parser.add_argument(
        "id",
        type=int,
        nargs="+",
        help="Image/Well/Plate ID",
    )
    _ = parser.add_argument(
        "--ch",
        type=int,
        default=0,
        help="Nuclei channel (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--device",
        help="Torch device (default to auto-detect)",
    )
    # _ = parser.add_argument(
    #     "--results",
    #     help="Result filename (default to use input IDs)",
    # )
    _ = parser.add_argument(
        "--overwrite",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing results (default: %(default)s)",
    )
    # Adapted from src.model.micronuclai_predict.py
    options = parser.add_argument_group(title="micronuclAI arguments")
    _ = options.add_argument("-mod", "--model", dest="model", action="store", required=True,
                       help="Pathway to prediction model.")
    _ = options.add_argument("-s", "--size", dest="size", action="store", required=False, default=(256, 256),
                         type=int, nargs="+", help="Size of images for training (default: %(default)s)")
    _ =options.add_argument("-rf", "--resizing-factor", dest="resizing_factor", action="store", required=False,
                         default=0.6, type=float, help="Resizing factor for images (default: %(default)s)")
    _ = options.add_argument("-e", "--expansion", dest="expansion", action="store", required=False, default=25,
                         type=int, help="Expansion factor for images (default: %(default)s)")
    _ = options.add_argument("-log", "--log-level", dest="log_level", action="store", default="info",
                         choices=["debug", "info"],
                         help="Set the logging level (default: %(default)s)")
    _ = options.add_argument("-o", "--out", dest="out", action="store", required=True,
                        help="Path to the output data folder")
    args = parser.parse_args()

    import os

    # fn = args.results
    # if fn is None:
    #     if len(args.id) == 1:
    #         fn = f"results_{args.id[0]}.csv"
    #     else:
    #         fn = f"results_{args.id[0]}_and_{len(args.id) - 1}.csv"
    # if os.path.exists(fn):
    #     if args.overwrite:
    #         print(f"Warning: results file {fn} already exists, will be overwritten")
    #     else:
    #         print(f"Warning: results file {fn} already exists")
    #         exit(1)
    # print(f"Results file: {fn}")

    import time
    import logging
    import contextlib

    import numpy as np
    import pandas as pd

    import torch
    from torch.utils.data import DataLoader
    from omero.cli import cli_login
    from omero.gateway import BlitzGateway, MapAnnotationWrapper

    from src.model.dataset import micronuclAI_inference
    from src.model.augmentations import get_transforms
    from src.model.logger import set_logger
    from src.model.micronuclai_predict import inference, summarize
    from pathlib import Path
    from tifffile import imwrite

    args.out = Path(args.out).resolve()
    args.out.mkdir(parents=True, exist_ok=True)

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

    def get_image_ids(conn, id):
        dataset = conn.getObject("Dataset", id)
        if dataset is not None:
            return [img.getId() for img in dataset.listChildren()]
        image = conn.getObject("Image", id)
        if image is None:
            log.warning(f"Image {id} not found")
            return []
        return [image.getId()]

    def get_mask(conn, image):
        annotations = image.listAnnotations()
        for ann in annotations:
            if isinstance(ann, MapAnnotationWrapper):
                for k, v in ann.getValue():
                    if k == "Segmentation_Mask":
                        return conn.getObject("Image", int(v))
        return None

    # Working mask
    mask_file = f"tmp_mask_{os.getpid()}.tif"

    log.info("Connecting to OMERO")
    with cli_login() as cli:
        # cli_login causes double logging by configuring the root logger
        logging.getLogger().handlers.clear()

        start = time.time()
        conn = BlitzGateway(client_obj=cli._client)
        # Set group to -1 for all groups
        conn.SERVICE_OPTS.setOmeroGroup(-1)

        log.info("Processing images")

        # Process all images
        for i, input_id in enumerate(args.id):
            log.info(f"Input ID [{i + 1}/{len(args.id)}] {input_id}")
            image_ids = get_image_ids(conn, input_id)
            for j, image_id in enumerate(image_ids):
                image = conn.getObject("Image", image_id)

                sizeZ = image.getSizeZ()
                sizeC = image.getSizeC()
                sizeT = image.getSizeT()
                log.info(
                    f"Image [{j + 1}/{len(image_ids)}] {image_id}: z={sizeZ}, c={sizeC}, t={sizeT}"
                )

                if sizeZ * sizeT != 1:
                    log.warning("Require 2D image")
                    continue

                mask = get_mask(conn, image)
                if mask is None:
                    log.warning("No annotation for Segmentation_Mask")
                    continue

                img = image.getPrimaryPixels().getPlane(0, args.ch, 0)
                m = mask.getPrimaryPixels().getPlane(0, 0, 0)

                image_name = image.getName()
                log.info(
                    f"Image [{j + 1}/{len(image_ids)}] {image_id} [{image_name}]: shape={img.shape}"
                )

                # Save mask to file.
                # No API for loading a mask from numpy array
                # but the image can be a numpy array.
                imwrite(mask_file, m)

                # Run micronuclAI
                # Adapted from src.model.micronuclai_predict.py

                # Load dataset
                dataset = micronuclAI_inference(img,
                                                mask_file,
                                                resizing_factor=args.resizing_factor,
                                                expansion=args.expansion,
                                                size=args.size,
                                                transform=transform)

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
                df_predictions.to_csv(args.out.joinpath(f"{image_id}_predictions.csv"), index=False)
                df_mn_counts.to_csv(args.out.joinpath(f"{image_id}_counts.csv"), index=True)
                df_summary.to_csv(args.out.joinpath(f"{image_id}_summary.csv"), index=True, header=False)

        log.info(f"Image processing time: {time.time() - start:.2f} seconds")

    with contextlib.suppress(FileNotFoundError):
        os.remove(mask_file)

if __name__ == "__main__":
    main()
