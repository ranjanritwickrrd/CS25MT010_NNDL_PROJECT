# NNDL Project - Full Pipeline Commands

## Current Status
- ✅ Virtual environment configured (Python 3.14.3)
- ✅ Dependencies installed
- ✅ Dataset verified (7000 image pairs in data/clean and data/hazy)
- ✅ Config updated for full dataset (MAX_KAGGLE_PAIRS = 7000)
- ⏳ **TRAINING IN PROGRESS** - All 3 models training on full dataset

## Training Progress
Training started at: 2026-04-20 15:45+
Expected completion: ~30-120 minutes

Models being trained:
- baseline_best.pt
- residual_best.pt  
- skip_best.pt

## Commands to Run After Training Completes

### 1. Verify Training Completed
```powershell
cd d:\JetAcker_RoadSim\NNDL_Project
ls checkpoints\
```
Expected output: 6 files (3 .pt models + 3 .json metrics)

### 2. Execute Notebook with New Results
Regenerates all graphs and metrics with the newly trained models.
```powershell
cd d:\JetAcker_RoadSim\NNDL_Project
d:\JetAcker_RoadSim\.venv\Scripts\python.exe tools\execute_notebook.py
```
Output: `docs\image_dehazing_aodnet_residual_skip_notebook.executed.ipynb`

### 3. Launch Web-Based Application
Starts the interactive dehazing web interface on localhost.
```powershell
cd d:\JetAcker_RoadSim\NNDL_Project
d:\JetAcker_RoadSim\.venv\Scripts\python.exe app.py
```
The web interface will open on:
- http://127.0.0.1:7860 (or next available port 7861-7875)

### 4. Open Notebook Interactively (Optional)
For live interaction with the notebook:
```powershell
cd d:\JetAcker_RoadSim\NNDL_Project
.\tools\run_dehazing_notebook.ps1
```

## Full Automation Script
Run everything at once:
```powershell
# Navigate to project
cd d:\JetAcker_RoadSim\NNDL_Project

# 1. Train all models (already running)
# d:\JetAcker_RoadSim\.venv\Scripts\python.exe tools\train_web_models.py

# 2. Execute notebook to generate results
d:\JetAcker_RoadSim\.venv\Scripts\python.exe tools\execute_notebook.py

# 3. Verify checkpoints created
Write-Host "Training complete! Checkpoints:"
ls checkpoints\ | Format-Table

# 4. Launch web app
d:\JetAcker_RoadSim\.venv\Scripts\python.exe app.py
```

## Configuration Changes Made
- Updated `dehaze_app/config.py`:
  - MAX_KAGGLE_PAIRS: 600 → 7000 (use full dataset)
  
## Output Files Generated
- `checkpoints/baseline_best.pt` - Baseline model (trained)
- `checkpoints/residual_best.pt` - Residual model (trained)
- `checkpoints/skip_best.pt` - Skip connections model (trained)
- `checkpoints/*_metrics.json` - Performance metrics for each model
- `docs/image_dehazing_aodnet_residual_skip_notebook.executed.ipynb` - Executed notebook with graphs
- `outputs/` - Generated visualization plots

## Presentation & Report Updates
After executing the notebook, the following will be auto-updated:
- **Graphs**: MSE, SSIM, PSNR metrics comparison
- **Model Variants**: Baseline vs Residual vs Skip-Connections performance
- **Dataset**: Full RESIDE 6K (7000 pairs) results
- **Inference Examples**: Sample dehazed outputs

## Next Steps After Everything Completes
1. Review `image_dehazing_aodnet_residual_skip_notebook.executed.ipynb` for final results
2. Export/present the metrics and graphs
3. Access web demo at localhost to test live dehazing
4. Compare all 3 model variants in the web interface

## Troubleshooting
If training fails:
```powershell
# Check if packages are installed
d:\JetAcker_RoadSim\.venv\Scripts\python.exe -m pip list | grep -E "torch|torchvision"

# Reinstall if needed
d:\JetAcker_RoadSim\.venv\Scripts\python.exe -m pip install torch torchvision
```

If web app won't start:
```powershell
# Kill any previous instances
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# Retry
d:\JetAcker_RoadSim\.venv\Scripts\python.exe app.py
```
