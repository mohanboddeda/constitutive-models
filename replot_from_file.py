# replot_from_file.py
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.posttrain_flow import plot_all_losses

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Path to the SAVED history file
FILE_PATH = "trained_models/maxwellflow/single_stage/seed_42/biaxial_extension/10ksamples/maxwell_B_1.0_1.2/loss_history.npz"

# 2. Output folder (Optional - defaults to same folder as .npz)
OUTPUT_DIR = None 
# ==========================================

def main():
    if not os.path.exists(FILE_PATH):
        print(f"❌ File not found: {FILE_PATH}")
        return

    print(f"📂 Loading history from: {FILE_PATH}")
    data = np.load(FILE_PATH, allow_pickle=True)

    # Extract Data
    train_d = data['train_d']
    val_d   = data['val_d']
    train_p = data['train_p']
    val_p   = data['val_p']
    Y_std   = data['Y_std']
    
    # Extract Metadata (handle missing keys gracefully)
    flow_type  = str(data['flow_type']) if 'flow_type' in data else "Unknown"
    stage_tag  = str(data['stage_tag']) if 'stage_tag' in data else "Replot"
    model_type = str(data['model_type']) if 'model_type' in data else "Model"
    n_samples  = int(data['n_samples']) if 'n_samples' in data else None

    # Determine Output Directory
    fig_dir = OUTPUT_DIR if OUTPUT_DIR else os.path.join(os.path.dirname(FILE_PATH), "figures_replot")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"📊 Generating Plot in: {fig_dir}")
    
    # CALL YOUR PLOTTING FUNCTION
    plot_all_losses(
        train_d, val_d, train_p, val_p, 
        Y_std, fig_dir, model_type, stage_tag, 
        flow_type=flow_type, 
        n_samples=n_samples
    )
    print("✅ Done!")

if __name__ == "__main__":
    main()