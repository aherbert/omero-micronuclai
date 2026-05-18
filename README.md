# omero-micronuclai

OMERO micronuclAI uses [micronuclAI](https://github.com/SchapiroLabor/micronuclAI) to analyse images
in [OMERO](https://www.openmicroscopy.org/omero/).

## Installation

```bash
# Install uv
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows (not tested!)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone the repository
git clone https://github.com/aherbert/omero-micronuclai.git
# Change into the project directory
cd omero-micronuclai
# Create the virtual environment
uv sync

# Activate the virtual environment.
# This avoids using the 'uv run' prefix for all executed programs.
source .venv/bin/activate
```

`micronuclAI` requires a model file to be specified. This can
be provided from a download of the [micronuclAI](https://github.com/SchapiroLabor/micronuclAI) repository.

```
git clone https://github.com/SchapiroLabor/micronuclAI
```

## Analysis

The script will connect to an OMERO server using the provided credentials.

Analysis is performed using 2D images with a single timepoint.
The nucleus channel for analysis can be specified.

Mask images are assumed to exist in OMERO. A map annotation on the image should contains the mask image ID, e.g. `Segmentation_Mask:123`.
The first channel of the mask image is used as the nucleus mask.

Analysis of a single image:

```
# Show options
uv run ./main.py -h

# Analyse
uv run ./main.py ID [ID ...] -o output --model ../micronuclAI/models/micronuclai.pt
```

Connection to OMERO uses the [OMERO.py](https://github.com/ome/omero-py)
Python bindings. The script will ask for your OMERO server
URL and username and password. Repeat invocations will reuse an active
session or reconnect if it has timed out.

Multiple IDs can be provided to analyse selected images.

Results are saved to the `output` directory for each image:

- `[ID]_counts.csv`: Summary count of number of micronuclei across all nuclei.
- `[ID]_predictions.csv`: Score and count of micronuclei per nucleus.
- `[ID]_summary.csv`: Summary of analysis results.
