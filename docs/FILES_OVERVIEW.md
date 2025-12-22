# 📁 Project Files Overview

## Complete File Structure

```
artwork-classifier/
│
├── 📱 GUI APPLICATION
│   ├── gui/
│   │   ├── app.py                              ⭐ Main Streamlit application
│   │   └── pages/
│   │       ├── 1_📊_Model_Comparison.py        📊 Model performance comparison
│   │       ├── 2_📈_Evaluation_Metrics.py      📈 Confusion matrices & metrics
│   │       ├── 3_🎭_Multi-Task_Learning.py     🎭 Bonus: Multi-task feature
│   │       └── 4_📚_Documentation.py           📚 In-app documentation
│
├── 🧠 MODELS
│   └── models/
│       ├── vgg16_top6_finetuned.pth           🔵 VGG16 weights (528 MB)
│       ├── resnet50_top6_finetuned.pth        🟢 ResNet50 weights (98 MB)
│       ├── efficientnet_b1_resumed_10ep.pth   🟡 EfficientNet-B1 weights (30 MB)
│       ├── swin_artist_model.pth              🟣 Swin Transformer (335 MB)
│       └── multitask_swin_model.pth           🔴 Multi-task model (335 MB)
│
├── 📓 NOTEBOOKS
│   └── notebooks/
│       ├── Artwork-Classification.ipynb        📓 Main training notebook
│       ├── Multi-Task.ipynb                   📓 Multi-task training
│       └── wikiart.ipynb                      📓 Dataset exploration
│
├── 🖼️ TEST DATA
│   └── test_images/
│       └── test_images/                       🖼️ Sample artwork images (50+)
│
├── 📚 DOCUMENTATION
│   ├── docs/
│   │   ├── PROJECT_REPORT.md                  📄 Complete academic report
│   │   ├── PROJECT_SUMMARY.md                 📄 Executive summary
│   │   └── ARCHITECTURE.md                    📄 System architecture
│   │
│   ├── README.md                              ⭐ Main project readme
│   ├── SETUP.md                               🔧 Installation & setup guide
│   ├── QUICK_REFERENCE.md                     📖 Quick reference guide
│   ├── CONTRIBUTING.md                        🤝 Contribution guidelines
│   └── FILES_OVERVIEW.md                      📁 This file
│
├── 🧪 TESTING & UTILITIES
│   ├── test_setup.py                          ✅ Setup verification script
│   ├── run_app.bat                            ⚡ Windows launch script
│   └── run_app.sh                             ⚡ Linux/Mac launch script
│
├── ⚙️ CONFIGURATION
│   ├── requirements.txt                       📦 Python dependencies
│   ├── .gitignore                             🚫 Git ignore rules
│   └── Term project.pdf                       📋 Project requirements
│
└── 📝 PROJECT METADATA
    └── (Future files)
        ├── LICENSE                            📜 MIT License
        ├── CHANGELOG.md                       📝 Version history
        └── CONTRIBUTORS.md                    👥 Contributors list
```

---

## File Descriptions

### 🎯 Core Application Files

#### `gui/app.py` ⭐
**Purpose:** Main Streamlit application  
**Size:** ~400 lines  
**Key Features:**
- Model selection interface
- Image upload functionality
- Top-5 predictions display
- Grad-CAM visualization
- Download capabilities

**Dependencies:** streamlit, torch, torchvision, timm, opencv, matplotlib

---

#### `gui/pages/1_📊_Model_Comparison.py`
**Purpose:** Compare performance of all models  
**Size:** ~300 lines  
**Key Features:**
- Performance metrics table
- Interactive charts (accuracy, radar, bar charts)
- Model complexity analysis
- Best model recommendations

---

#### `gui/pages/2_📈_Evaluation_Metrics.py`
**Purpose:** Detailed evaluation metrics  
**Size:** ~350 lines  
**Key Features:**
- Confusion matrices (static & interactive)
- Per-class precision, recall, F1-score
- Misclassification analysis
- Top-10 errors display

---

#### `gui/pages/3_🎭_Multi-Task_Learning.py`
**Purpose:** Multi-task learning interface (BONUS)  
**Size:** ~400 lines  
**Key Features:**
- Simultaneous style + artist prediction
- Dual-task model loading
- Combined visualizations
- Analysis report generation

---

#### `gui/pages/4_📚_Documentation.py`
**Purpose:** In-app documentation hub  
**Size:** ~500 lines  
**Key Features:**
- Project overview
- Architecture details
- Dataset information
- Team contributions
- Usage guides

---

### 🧠 Model Files

| File | Size | Parameters | Purpose |
|------|------|------------|---------|
| `vgg16_top6_finetuned.pth` | 528 MB | 138M | VGG16 baseline model |
| `resnet50_top6_finetuned.pth` | 98 MB | 25.6M | ResNet50 model |
| `efficientnet_b1_resumed_10ep.pth` | 30 MB | 7.8M | Best efficiency model |
| `swin_artist_model.pth` | 335 MB | 87.8M | Artist classification |
| `multitask_swin_model.pth` | 335 MB | 87.8M | Multi-task model (bonus) |

**Total Model Size:** ~1.3 GB

---

### 📓 Jupyter Notebooks

#### `notebooks/Artwork-Classification.ipynb`
**Purpose:** Main training notebook  
**Contents:**
- Data loading and preprocessing
- VGG16, ResNet50, EfficientNet training
- Model evaluation
- Results visualization

---

#### `notebooks/Multi-Task.ipynb`
**Purpose:** Multi-task learning experiments  
**Contents:**
- Multi-task model architecture
- Joint training procedure
- Dual-task evaluation

---

#### `notebooks/wikiart.ipynb`
**Purpose:** Dataset exploration  
**Contents:**
- Dataset statistics
- Class distribution
- Sample visualizations
- Preprocessing experiments

---

### 📚 Documentation Files

#### `README.md` ⭐
**Length:** ~300 lines  
**Purpose:** Main project overview  
**Sections:**
- Project introduction
- Features list
- Installation guide
- Quick start
- Results summary
- Team information

---

#### `SETUP.md`
**Length:** ~400 lines  
**Purpose:** Detailed setup instructions  
**Sections:**
- Prerequisites
- Step-by-step installation
- Troubleshooting guide
- Configuration options
- Performance tips

---

#### `QUICK_REFERENCE.md`
**Length:** ~200 lines  
**Purpose:** Quick reference guide  
**Contents:**
- Common commands
- Model specifications
- Keyboard shortcuts
- Performance benchmarks
- Quick fixes

---

#### `docs/PROJECT_REPORT.md`
**Length:** ~800 lines  
**Purpose:** Complete academic report  
**Sections:**
- Executive summary
- Introduction & objectives
- Methodology
- Results & analysis
- Grad-CAM analysis
- Conclusion & future work

---

#### `docs/PROJECT_SUMMARY.md`
**Length:** ~400 lines  
**Purpose:** Executive summary  
**Contents:**
- Project highlights
- Key achievements
- Team contributions
- Grading breakdown

---

#### `docs/ARCHITECTURE.md`
**Length:** ~300 lines  
**Purpose:** System architecture  
**Contents:**
- High-level architecture
- Data flow diagrams
- Model architectures
- Technology stack

---

#### `CONTRIBUTING.md`
**Length:** ~600 lines  
**Purpose:** Contribution guidelines  
**Sections:**
- Code of conduct
- Development workflow
- Coding standards
- Testing guidelines
- PR process

---

### 🧪 Testing & Utilities

#### `test_setup.py`
**Length:** ~250 lines  
**Purpose:** Verify installation  
**Tests:**
- Package imports
- File structure
- Model files presence
- CUDA availability
- Model loading

**Usage:** `python test_setup.py`

---

#### `run_app.bat` / `run_app.sh`
**Length:** ~15 lines each  
**Purpose:** Launch application  
**Platform:** Windows / Linux-Mac  
**Usage:** 
- Windows: `run_app.bat`
- Linux/Mac: `./run_app.sh`

---

### ⚙️ Configuration Files

#### `requirements.txt`
**Length:** ~20 lines  
**Purpose:** Python dependencies  
**Key Packages:**
- torch>=2.0.0
- torchvision>=0.15.0
- streamlit>=1.28.0
- timm>=0.9.0
- opencv-python>=4.8.0
- matplotlib, seaborn, plotly
- scikit-learn, pandas, numpy

---

#### `.gitignore`
**Length:** ~60 lines  
**Purpose:** Git ignore rules  
**Ignores:**
- Python cache files
- Virtual environments
- Jupyter checkpoints
- IDE files
- Logs and temporary files

---

## File Statistics

### By Category

| Category | Files | Total Lines | Total Size |
|----------|-------|-------------|------------|
| GUI Application | 5 | ~2,000 | ~200 KB |
| Model Weights | 5 | N/A | ~1.3 GB |
| Notebooks | 3 | ~1,500 | ~5 MB |
| Documentation | 8 | ~3,500 | ~500 KB |
| Testing/Utils | 3 | ~300 | ~30 KB |
| Config | 2 | ~80 | ~10 KB |
| **TOTAL** | **26** | **~7,380** | **~1.3 GB** |

### By Language

| Language | Files | Lines | Percentage |
|----------|-------|-------|------------|
| Python | 8 | ~2,300 | 31% |
| Markdown | 9 | ~3,500 | 47% |
| Jupyter | 3 | ~1,500 | 20% |
| Shell | 2 | ~30 | <1% |
| Config | 2 | ~50 | <1% |

---

## File Dependencies

### Main Application Dependencies
```
app.py
├── requires: torch, streamlit, timm, cv2
├── loads: models/*.pth
└── imports: PIL, numpy, matplotlib

pages/*.py
├── requires: streamlit, plotly, pandas
└── imports: numpy, seaborn, sklearn
```

### Documentation Dependencies
```
*.md files
├── no dependencies (plain markdown)
└── viewable in any markdown reader
```

---

## File Usage Guide

### For End Users
**Essential files:**
1. `requirements.txt` - Install dependencies
2. `gui/app.py` - Run the application
3. `models/*.pth` - Pre-trained weights
4. `README.md` - Getting started

**Optional:**
- `test_setup.py` - Verify installation
- `QUICK_REFERENCE.md` - Quick commands
- `run_app.bat/sh` - Easy launcher

---

### For Developers
**Essential files:**
1. All GUI files - Application code
2. `CONTRIBUTING.md` - Development guidelines
3. `docs/ARCHITECTURE.md` - System design
4. `test_setup.py` - Testing script

**Optional:**
- Notebooks - Training experiments
- Documentation - Project details

---

### For Instructors/Reviewers
**Essential files:**
1. `README.md` - Project overview
2. `docs/PROJECT_REPORT.md` - Full report
3. `docs/PROJECT_SUMMARY.md` - Quick summary
4. GUI application - Working demo
5. Notebooks - Training process

---

## File Maintenance

### Regular Updates
- `README.md` - When features change
- `requirements.txt` - When dependencies change
- Documentation - When significant changes occur

### Version Control
- All files tracked in Git (except `.gitignore` rules)
- Model files: Optional (due to size)
- Dataset: Not tracked (too large)

---

## Missing Files (Future Work)

### Planned Additions
- [ ] `LICENSE` - MIT License file
- [ ] `CHANGELOG.md` - Version history
- [ ] `CONTRIBUTORS.md` - Contributors list
- [ ] `tests/` directory - Unit tests
- [ ] `Dockerfile` - Docker deployment
- [ ] `.github/workflows/` - CI/CD pipelines

---

## File Access Patterns

### High-Traffic Files (Frequently Accessed)
1. `gui/app.py` - Every app run
2. `models/*.pth` - Model loading
3. `requirements.txt` - Installation
4. `README.md` - First-time users

### Medium-Traffic Files
1. Documentation pages
2. Test setup script
3. Page files (during navigation)

### Low-Traffic Files
1. Notebooks (training phase)
2. Contributing guide (developers)
3. Architecture docs (advanced users)

---

## Backup & Recovery

### Critical Files (Must Backup)
✅ All GUI application files
✅ Model weight files (1.3 GB)
✅ Documentation files
✅ Configuration files

### Regenerable Files (Optional Backup)
- Test images (can re-download)
- Cache files
- Temporary files

---

## File Size Optimization

### Already Optimized
✅ EfficientNet-B1 (only 30 MB)
✅ Documentation (markdown format)
✅ Code (no unnecessary imports)

### Can Be Optimized (Future)
- Model quantization (reduce to INT8)
- Image compression for samples
- Code minification (if needed)

---

This overview provides a complete picture of all files in the project, their purposes, and how they relate to each other.
