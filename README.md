# Electrochromic Characterization using Video Analysis

Repository for extraction of reflectance curves of electrochromic materials reducing the effect of gamma correction using computer vision techniques and LEGO color calibration patches.

## 🎯 Overview

This project provides tools for analyzing electrochromic materials using video recordings and standardized LEGO color patches. The system includes:

- **Interactive LEGO patch selector** for precise region selection
- **Gamma correction optimization** for improved color accuracy
- **Automated video analysis** with ROI mask generation
- **Reflectance curve extraction** with temporal analysis

## 🛠️ Installation

### Quick Start with uv (Recommended)
```bash
# Clone the repository
git clone git@github.com:pgalantec/electrochromic_characterization.git
cd electrochromic_characterization

# Setup environment (installs uv if needed)
make init

# Activate environment
source .venv/bin/activate
```

### Alternative Installation (Standard Python)
```bash
# Clone the repository
git clone git@github.com:pgalantec/electrochromic_characterization.git
cd electrochromic_characterization

# Create virtual environment
make create_env

# Activate environment
source env/bin/activate
```

### Available Commands
```bash
make help           # Show all available commands
make init           # Setup environment with uv (fast)
make create_env     # Setup environment with standard venv
make clean          # Remove all virtual environments
```

## 📁 Project Structure

```
electrochromic_characterization/
├── assets/                     # Logo images
├── dataset/                    # Video files (.mp4, .avi)
├── output/                     # Experiment results
├── src/                        # Source code
│   ├── lego_app_v4.py         # Interactive configuration app
│   ├── experiment.py          # Analysis pipeline
│   ├── opt_gamma.py           # Gamma optimization
│   └── utils/                 # Utility functions
├── params.yaml                # Experiment configuration
├── Makefile                   # Development commands
└── README.md
```

## 🚀 Usage

### Step 1: Prepare Your Data
1. Place your video files in the `dataset/` directory
2. Supported formats: `.mp4`, `.avi`, `.mov`

### Step 2: Configure Experiments (Interactive Method - Recommended)

Launch the interactive LEGO patch selector:

```bash
# Using uv environment
make run_config_app_uv

# Or using standard environment
make run_config_app
```

The application will open in your browser at `http://localhost:8050` and provides:

- **📹 Video Selection**: Choose from available videos in dataset
- **🧱 LEGO Patch Selection**: Interactive selection of 4 color patches (black, dark grey, clear grey, white)
- **🎯 ROI Definition**: Draw region of interest for analysis
- **⚡ Gamma Calibration**: Enable/disable gamma correction
- **💾 Configuration Export**: Automatic generation of `params.yaml`

### Step 3: Configure Experiments (Manual Method)

Alternatively, you can manually edit `params.yaml`:

```yaml
videos:
  - video_path: "dataset/your_video.mp4"
    exp_name: "your_experiment_name"
    calibration: false                    # Enable gamma correction
    frame_dtype: Y                       # Color space: Y/gray/L
    patches: [                           # LEGO patch coordinates [x1,y1], [x2,y2]
        [[20, 1200], [100, 1380]],      # Black patch
        [[20, 950], [100, 1150]],       # Dark grey patch  
        [[20, 660], [100, 880]],        # Clear grey patch
        [[20, 400], [100, 600]]         # White patch
    ]

lego_srgb:                              # Nominal LEGO sRGB values
  black: [18, 18, 21]
  dark_grey: [98, 101, 102]  
  clear_grey: [160, 163, 164]
  white: [244, 238, 228]
```

### Step 4: Run Analysis

Execute the experiment analysis:

```bash
# Using uv environment
make run_experiment_uv

# Or using standard environment  
make run_experiment
```

The analysis pipeline will:
1. **🎨 Extract color patches** from the first frame
2. **⚡ Optimize gamma correction** (if enabled)
3. **🎭 Generate ROI mask** automatically or load existing
4. **📊 Analyze temporal changes** throughout the video
5. **📈 Generate reflectance curves** and statistics

## 📊 Output Results

Results are stored in `output/{video_name}/{experiment_name}/`:

```
output/ExpCalY-OptGamma/
├── config.json                 # Experiment configuration
├── patches.png                 # Visualization of selected patches
├── Mask.png                   # ROI binary mask
├── gamma_linearization.jpg    # Gamma correction plot
├── r2_optimization.jpg        # R² optimization curve
├── reflectance_curves.png     # Main results visualization
├── statistics.txt             # Numerical analysis
└── temporal_analysis/         # Frame-by-frame data
```

### Key Output Files:

- **`reflectance_curves.png`**: Main visualization showing temporal evolution
- **`statistics.txt`**: Numerical summary of the analysis
- **`config.json`**: Complete experiment configuration for reproducibility
- **`gamma_linearization.jpg`**: Gamma correction validation plots
- **`Mask.png`**: ROI mask used for analysis

## 🔧 Development Tools

```bash
# Code formatting
make format

# Code linting  
make lint

# Clean all environments
make clean

# Show help
make help
```

## 📝 LEGO Color Patches

The system uses 4 standardized LEGO color patches for calibration:

| Patch | Color | Nominal sRGB | Purpose |
|-------|--------|--------------|---------|
| 🖤 | Black | [18, 18, 21] | Dark reference |
| ⚫ | Dark Grey | [98, 101, 102] | Low-mid reference |
| ⚪ | Clear Grey | [160, 163, 164] | Mid-high reference |  
| ⬜ | White | [244, 238, 228] | Bright reference |

**Selection Order**: Always select patches in this order for consistent results.

## 🎨 Interactive Configuration Features

The LEGO patch selector application provides:

- **🖱️ Click-and-drag selection**: Draw rectangles around each LEGO patch
- **🎯 ROI drawing tools**: Rectangle and freehand selection for analysis region
- **📊 Real-time preview**: See your selections as you make them
- **⚡ Configuration validation**: Automatic verification of patch order and completeness
- **💾 One-click export**: Generate `params.yaml` automatically
- **🔄 Reset functionality**: Start over easily if needed

## 🚀 Advanced Features

### Gamma Optimization
- Automatically finds optimal gamma value for color linearization
- Generates R² optimization curves
- Validates correction effectiveness

### Automated ROI Detection
- Intelligent region of interest detection
- Fallback to manual mask loading
- Temporal stability validation

### Multi-format Support
- Various video formats (.mp4, .avi, .mov)
- Different color spaces (Y, gray, L*)
- Flexible resolution handling

## 🆘 Troubleshooting

### Common Issues:

1. **Video not found**: Ensure video files are in `dataset/` directory
2. **Permission errors**: Check file permissions and virtual environment activation
3. **Memory issues**: Use smaller video files or reduce resolution
4. **LEGO patch detection fails**: Ensure good lighting and clear patch visibility

### Getting Help:

```bash
# Check environment
make help

# Verify installation
python -c "import dash; print('Dash installed successfully')"

# Check video files
ls dataset/
```

## 👨‍💻 Contributors

- **Pablo Galán** - Tecnalia Research & Innovation
- **Artzai Picón** - Tecnalia Research & Innovation  
- **Jon Velasco** - BCMaterials

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Citation

If you use this work in your research, please cite:

```bibtex
@software{electrochromic_characterization,
  title={Electrochromic Characterization using Video Analysis},
  author={Galán, Pablo and Picón, Artzai and Velasco, Jon},
  year={2024},
  institution={Tecnalia Research \& Innovation, BCMaterials}
}
```

---

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| **Setup** | `make init` |
| **Configure** | `make run_config_app_uv` |
| **Analyze** | `make run_experiment_uv` |
| **Help** | `make help` |
| **Clean** | `make clean` |

**Happy analyzing! 🎬✨**

