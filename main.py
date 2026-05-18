#!/usr/bin/env python3
"""Analyse OMERO images using micronuclAI."""

import argparse

from omero.gateway import WellSampleWrapper


def main():
    parser = argparse.ArgumentParser(
        description="Analyse OMERO images using micronuclAI."
    )
    _ = parser.add_argument(
        "id",
        type=int,
        nargs="+",
        help="Image/Well/Plate ID",
    )
    _ = parser.add_argument(
        "--object",
        default="image",
        choices=["image", "well", "plate"],
        help="OMERO object type (default: %(default)s)",
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
    _ = options.add_argument(
        "-o",
        "--out",
        dest="out",
        default="output",
        help="Path to the output data folder (default: %(default)s)",
    )
    args = parser.parse_args()

    import contextlib
    import logging
    import os
    import time
    from pathlib import Path

    import omero
    import torch
    from omero.cli import cli_login
    from omero.gateway import BlitzGateway, MapAnnotationWrapper
    from omero.rtypes import unwrap
    from src.model.augmentations import get_transforms
    from src.model.dataset import micronuclAI_inference
    from src.model.logger import set_logger
    from src.model.micronuclai_predict import inference, summarize
    from tifffile import imwrite
    from torch.utils.data import DataLoader

    args.out = Path(args.out).resolve()

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

    def get_image_ids(conn, object_type, id):
        q = conn.getQueryService()
        params = omero.sys.ParametersI()
        params.addLong("oid", id)
        data = []

        query = """select pt.id, well.row, well.column, img.id from Well as well
                  left join well.plate as pt
                  left outer join well.wellSamples as ws
                  left outer join ws.image as img"""

        if object_type == "plate":
            query += " where pt.id = :oid"
        elif object_type == "well":
            query += " where well.id = :oid"
        else:
            query += " where ws.image.id = :oid"
        query += " order by pt.id, well.row, well.column, ws.image.id"

        for r in q.projection(query, params):
            data.append([unwrap(x) for x in r])
        return data

    def get_mask(conn, image):
        annotations = image.listAnnotations()
        for ann in annotations:
            if isinstance(ann, MapAnnotationWrapper):
                for k, v in ann.getValue():
                    if k == "Segmentation_Mask":
                        return conn.getObject("Image", int(v))
        return None

    def _row_col_to_well_pos(row: int, col: int) -> str:
        """Convert 0-based row/column to well position string (e.g. 'A1')."""
        return f"{chr(65 + row)}{col + 1}"

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

        log.info("Processing images as %s", conn.getUser().getName())

        # Process all images
        for i, input_id in enumerate(args.id):
            log.info(f"Input ID [{i + 1}/{len(args.id)}] {input_id}")
            image_ids = get_image_ids(conn, args.object, input_id)
            if not image_ids:
                log.warning("No images for %s: %s", args.object, input_id)
                continue

            for j, (plate_id, row, col, image_id) in enumerate(image_ids):
                well_pos = _row_col_to_well_pos(row, col)
                basename = args.out / str(plate_id) / well_pos / str(image_id)
                basename.parent.mkdir(parents=True, exist_ok=True)

                # Skip existing output files
                prediction_file = f"{basename}_predictions.csv"
                if os.path.exists(prediction_file) and not args.overwrite:
                    log.info(f"Skipping {image_id} (already processed)")
                    continue

                image = conn.getObject("Image", image_id)

                sizeZ = image.getSizeZ()
                sizeC = image.getSizeC()
                sizeT = image.getSizeT()
                log.info(
                    f"{well_pos} Image [{j + 1}/{len(image_ids)}] {image_id}: z={sizeZ}, c={sizeC}, t={sizeT}"
                )

                if sizeZ * sizeT != 1:
                    log.warning("Require 2D image")
                    continue

                mask = get_mask(conn, image)
                if mask is None:
                    log.warning("No annotation for Segmentation_Mask")
                    continue

                m = mask.getPrimaryPixels().getPlane(0, 0, 0)
                if not m.any():
                    log.warning(f"{well_pos} Image [{j + 1}/{len(image_ids)}] {image_id}: No mask objects")
                    # Create empty prediction file to avoid reprocessing
                    Path(prediction_file).touch()
                    continue

                img = image.getPrimaryPixels().getPlane(0, args.ch, 0)
                image_name = image.getName()
                log.info(
                    f"{well_pos} Image [{j + 1}/{len(image_ids)}] {image_id} [{image_name}]: shape={img.shape}"
                )

                # Save mask to file.
                # No API for loading a mask from numpy array
                # but the image can be a numpy array.
                imwrite(mask_file, m)

                # Run micronuclAI
                # Adapted from src.model.micronuclai_predict.py

                # Load dataset
                dataset = micronuclAI_inference(
                    img,
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

    with contextlib.suppress(FileNotFoundError):
        os.remove(mask_file)


if __name__ == "__main__":
    main()
