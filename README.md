# Acned

![Acned detection interface](Acne%20detection.png)

A local image detection app that highlights acne classes with colored bounding boxes and shows total and per-class counts above the results. Upload an image, compare the original with the annotated output, and download the result.

## Features

- Single-page, responsive interface with drag-and-drop upload and image preview.
- JPG, PNG and WebP uploads up to 20 MB.
- Six classes read from `data.yaml`: blackheads, dark spot, nodules, papules, pustules and whiteheads.
- Distinct box colors and labels for each class, including zero counts in the summary.
- Automatic CUDA selection when available, with CPU fallback.
- Local processing with temporary files cleaned up after each request.

## Setup

Use Python 3.12, the version used during development. Run these commands from the project directory.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, activate the environment with `source .venv/bin/activate` instead.

### Model checkpoint

Place your trained Ultralytics detection checkpoint at `best.pt` beside `inference.py`. Model weights and datasets are excluded from Git; cloning this repository alone does not provide the trained model. Obtain the checkpoint separately from the project owner or use your own compatible checkpoint.

Only load checkpoints from a trusted source. The app uses `torch.load(..., weights_only=False)` to deserialize the full model, which can execute code from the checkpoint.

The class IDs and names in `data.yaml` must match the checkpoint's training labels. Only its `names` mapping is used for inference; dataset training paths are not required to run the app.

## Run

```powershell
python inference.py
```

Open **http://127.0.0.1:8000**. Choose or drop an image, click **Analyze image**, and use **Download image** to save the annotated result. Press `Ctrl+C` in the terminal to stop the server.

Keep the HTML, CSS and JavaScript files beside `inference.py`; open the app through the Python server rather than opening the HTML file directly.

## Project files

```text
.
├── inference.py       # HTTP server, model loading and image processing
├── index.html         # Page structure
├── styles.css         # Responsive styles
├── app.js             # Upload, preview, counts and download behavior
├── data.yaml          # Class names
├── requirements.txt   # Python dependencies
├── .gitignore
├── README.md
└── best.pt            # Local checkpoint; excluded from Git
```

## Configuration

In `inference.py`, change `MODEL_PATH` to use another checkpoint, `MAX_UPLOAD` to adjust the upload size, or the server address in `main()` to change the port. The default confidence threshold is `0.25` and the NMS IoU threshold is `0.45`, set in `postprocess()`.

The server listens on `127.0.0.1` for local use. GitHub hosts the source code; running inference requires a Python environment and the trained checkpoint. GitHub Pages cannot run this Python backend.

## Troubleshooting

- **Checkpoint not found:** put `best.pt` in the project root or update `MODEL_PATH`.
- **Port already in use:** stop the existing app instance or change port `8000` in `main()`.
- **CUDA unavailable:** the app automatically uses CPU. GPU acceleration requires a compatible PyTorch installation and NVIDIA driver.
- **Missing dependencies:** install requirements using the same Python environment used to run the app.

## Before publishing

The `.gitignore` excludes weights, datasets, local sample images, environments and generated outputs. Files already tracked by Git remain tracked even after adding an ignore rule; review your staged files before pushing. Share the trained checkpoint separately if others should be able to run your model.
