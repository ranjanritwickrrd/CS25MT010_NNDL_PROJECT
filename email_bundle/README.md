# NNDL_Project

Academic project workspace for:

`Image Dehazing using AOD-Net with Residual Learning and Skip Connections`

## Layout

- `docs/image_dehazing_aodnet_residual_skip_notebook.ipynb`
  The main step-by-step notebook for presentation.
- `docs/image_dehazing_notebook_requirements.txt`
  Python dependencies for the notebook and web demo.
- `app.py`
  Launches the interactive web interface for image dehazing.
- `dehaze_app/`
  Modular training, inference, and web UI package used by the notebook-aligned demo.
- `tools/train_web_models.py`
  Trains the saved checkpoints used by the web interface.
- `tools/run_web_app.ps1`
  Starts the interactive web app from the local virtual environment.
- `tools/generate_aod_notebook.py`
  Regenerates the notebook file.
- `tools/execute_notebook.py`
  Executes the notebook and saves an executed copy.
- `tools/run_dehazing_notebook.ps1`
  Opens the notebook server from the local virtual environment.
- `reference/AOD-Net-PyTorch-src`
  Local copy of the original repository used as the source reference.

## Typical workflow

1. Create or activate `.venv` inside `NNDL_Project`.
2. Install dependencies from `docs/image_dehazing_notebook_requirements.txt`.
3. Regenerate the notebook if you edit the generator.
4. Execute the notebook with `tools/execute_notebook.py`.
5. Train the reusable checkpoints with `tools/train_web_models.py`.
6. Open the notebook interactively with `tools/run_dehazing_notebook.ps1`.
7. Launch the web demo with `tools/run_web_app.ps1`.

## Web Demo

The web interface lets you:

- upload your own hazy image
- choose between the baseline, residual, and skip-connection AOD-Net variants
- preview a dehazed output image
- inspect simple run details such as the model name and input size

The demo is trained on the Kaggle-backed RESIDE 6K image pairs already prepared in `data/clean` and `data/hazy`.
It is designed for haze removal, so it can still accept blurred images, but the strongest results will come from hazy or low-contrast outdoor scenes.
