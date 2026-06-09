# Classical Feature Matching: SIFT Implementation

A scratch implementation of the Scale-Invariant Feature Transform (SIFT) algorithm as specified in Lowe (2004), with comprehensive robustness experiments.

## Environment Setup

### Prerequisites
- Python 3.12+
- `uv` package manager (https://docs.astral.sh/uv/)

### Installation

```bash
# Clone or navigate to the repository
cd classical-feature-matching

# uv will automatically create a virtual environment and install dependencies
# from pyproject.toml when you first run a command
uv run python --version
```

**Dependencies** (managed by `pyproject.toml`):
- `numpy`: numerical computing
- `opencv-python`: image I/O and basic transformations
- `matplotlib`: visualization
- `pytest`: testing framework

## Project Structure

```
.
├── sift/                          # Main SIFT implementation module
│   ├── __init__.py                # Module exports
│   ├── scale_space.py             # Gaussian pyramid & DoG construction
│   ├── keypoints.py               # Extrema detection & refinement
│   ├── orientation.py             # Dominant orientation assignment
│   ├── descriptors.py             # 128D descriptor generation
│   ├── matching.py                # BBF kd-tree matching
│   └── hough.py                   # Hough voting & affine fitting
├── experiments/
│   ├── inputs/                    # Input images for experiments
│   │   ├── source1.jpg            # Feature-rich test image
│   │   └── source2.jpg            # Feature-poor image (self-test)
│   ├── rotation_experiment.py      # Rotation invariance evaluation
│   ├── translation_experiment.py   # Translation invariance evaluation
│   ├── brightness_experiment.py    # Brightness change robustness
│   ├── scale_experiment.py         # Scale invariance evaluation
│   └── noise_experiment.py         # Noise robustness evaluation
├── tests/
│   ├── test_sift.py               # Unit tests (14 tests)
│   └── test_hough.py              # Geometric verification tests (14 tests)
├── outputs/                        # Experiment results (auto-generated)
│   ├── rotation/
│   ├── translation/
│   ├── brightness/
│   ├── scale/
│   └── noise/
├── main.py                        # SIFT feature matching demo
└── README.md                      # This file
```

## Running Tests

### Run all tests

```bash
uv run python -m pytest tests/ -v
```

Output:
```
28 passed in 0.74s
```

### Run specific test module

```bash
# SIFT construction tests
uv run python -m pytest tests/test_sift.py -v

# Geometric verification tests
uv run python -m pytest tests/test_hough.py -v
```

### Run with coverage (if pytest-cov is installed)

```bash
uv run python -m pytest tests/ --cov=sift --cov-report=html
```

## Using main.py

### Basic Usage

Perform SIFT feature detection, matching, and geometric verification on two images:

```bash
uv run python main.py <image1> <image2> [options]
```

### Examples

#### Match two images and display result

```bash
uv run python main.py experiments/inputs/source1.jpg experiments/inputs/source1.jpg
```

#### Save visualization to file

```bash
uv run python main.py img1.jpg img2.jpg --save matches.png
```

#### Adjust SIFT parameters

```bash
uv run python main.py img1.jpg img2.jpg --octaves 3 --scales 4 --sigma 1.5
```

#### Run without displaying (headless)

```bash
uv run python main.py img1.jpg img2.jpg --no-show
```

### Options

```
--save PATH              Save visualization to file (default: display only)
--octaves N              Number of octaves (default: 4)
--scales N               Scales per octave (default: 3)
--sigma FLOAT            Initial Gaussian sigma (default: 1.6)
--threshold FLOAT        Affine verification threshold in pixels (default: 6.0)
--no-show                Suppress display; only save/print (for headless mode)
--help                   Show this help message
```

### Output

For identical images, expect 100% inlier rate and affine identity transform:

```
Loading images...
  Image 1: source1.jpg  500×500
  Image 2: source1.jpg  500×500

Running SIFT (octaves=4, scales=3, σ=1.6)...
  Keypoints: 1464 (image 1)  1464 (image 2)  [8.0s]
  Matches after ratio test: 1458  [0.9s]
  Hough inliers: 1458/1458  [0.1s]
  Affine M:
[[ 1.  0.]
 [-0.  1.]]
  Affine t: [0.  0.]
```

## Running Experiments

All experiments evaluate SIFT robustness to image transformations and perturbations.  
Experiments use **source1 only** (source2 is feature-poor and used for diagnostic self-test).

### SIFT Configuration (all experiments)

```python
num_octaves = 4          # 4 scale-space levels
num_scales = 3           # 3 Gaussian levels per octave
sigma = 1.6              # Initial Gaussian blur
ratio_threshold = 0.75   # Lowe ratio test
affine_threshold = 6.0   # Geometric verification threshold (pixels)
```

### 1. Rotation Invariance Experiment

Test SIFT invariance to image rotation about the image center.

```bash
uv run python experiments/rotation_experiment.py
```

**Setup:**
- img1: 500×500 source image (no rotation)
- img2: Source image rotated by θ ∈ {0°, 15°, 30°, 60°, 90°, 120°, 180°}
- Evaluation region: Central disk (radius 250px, rotation-symmetric)

**Output:**
- `outputs/rotation/source1_angle{θ:03d}.png`: Per-angle visualization
- `outputs/rotation/summary.png`: Inlier rate vs angle line chart
- `outputs/rotation/stats.json`: Full statistics (JSON)

**Expected result:** ~98–100% inlier rate across all angles

---

### 2. Translation Invariance Experiment

Test SIFT invariance to image translation in four cardinal directions.

```bash
uv run python experiments/translation_experiment.py
```

**Setup:**
- img1: 500×500 source image (no translation)
- img2: Source image translated by d ∈ {0, 25, 50, 75, 100} pixels
- Directions: right, left, up, down
- Evaluation region: Central 300×300 window (fixed in img1, shifted in img2)

**Output:**
- `outputs/translation/{dir}_d{d:03d}.png`: Per-(direction, distance) visualization
- `outputs/translation/summary.png`: Inlier rate vs distance (averaged over 4 directions) line chart
- `outputs/translation/stats.json`: Full statistics (JSON)

**Expected result:** 100% inlier rate for all translations within evaluation region

---

### 3. Brightness Change Robustness Experiment

Test SIFT robustness to multiplicative brightness scaling.

```bash
uv run python experiments/brightness_experiment.py
```

**Setup:**
- img1: 500×500 source image (α = 1.0)
- img2: img1 × α (clipped to [0, 255]), where α ∈ {0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0}
- Evaluation region: Full 500×500 image (no geometric transformation)

**Output:**
- `outputs/brightness/source1_alpha{α_str}.png`: Per-alpha visualization
- `outputs/brightness/summary.png`: Inlier rate vs alpha line chart
- `outputs/brightness/stats.json`: Full statistics (JSON)

**Expected result:** 
- α ∈ [0.6, 2.0]: 90–100% inlier rate
- α = 0.4 (very dark): 40–50% (severe contrast loss)

---

### 4. Scale Invariance Experiment

Test SIFT invariance to zoom (scaling with respect to image content).

```bash
uv run python experiments/scale_experiment.py
```

**Setup:**
- img1: 500×500 source image (s = 1.0, reference)
- img2: Source image scaled by s ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}, output kept at 500×500
  - s > 1 (zoom in): Resize to 500·s × 500·s, then centre-crop to 500×500
  - s < 1 (zoom out): Resize to 500·s × 500·s, then centre-pad with black to 500×500
- Evaluation region: Content-overlap region (dynamically sized based on s)

**Output:**
- `outputs/scale/source1_scale{s_str}.png`: Per-scale visualization
- `outputs/scale/summary.png`: Inlier rate vs scale factor line chart
- `outputs/scale/stats.json`: Full statistics (JSON)

**Expected result:** 89–100% inlier rate across all scales

---

### 5. Noise Robustness Experiment

Test SIFT robustness to additive Gaussian noise.

```bash
uv run python experiments/noise_experiment.py
```

**Setup:**
- img1: 500×500 source image (no noise)
- img2: img1 + N(0, σ_n²) (clipped to [0, 255]), where σ_n ∈ {0, 5, 10, 20, 30, 40}
- Evaluation region: Full 500×500 image (no geometric transformation)
- RNG seed: 42 (reproducible)

**Output:**
- `outputs/noise/source1_sigma{σ_n:02d}.png`: Per-sigma visualization
- `outputs/noise/summary.png`: Inlier rate vs noise std line chart
- `outputs/noise/stats.json`: Full statistics (JSON)

**Expected result:** 94–100% inlier rate across all noise levels

---

### Running All Experiments at Once

```bash
cd /home/akira/univ/cv/classical-feature-matching && \
uv run python experiments/rotation_experiment.py && \
uv run python experiments/translation_experiment.py && \
uv run python experiments/brightness_experiment.py && \
uv run python experiments/scale_experiment.py && \
uv run python experiments/noise_experiment.py
```

**Typical runtime:** ~30–40 minutes total (depends on machine)

### Clearing Previous Results

```bash
rm -rf outputs/*
```

## Algorithm Overview

### SIFT Pipeline (Lowe 2004)

1. **Scale-Space Construction** (scale_space.py)
   - 2× image upsampling with σ_init = √(1.6² - 1.0²) ≈ 1.249
   - Gaussian pyramid: 4 octaves, 3 scales/octave, σ₀ = 1.6
   - DoG pyramid: 5 DoG levels per octave (for 3 internal scales)

2. **Keypoint Detection** (keypoints.py)
   - 26-neighborhood 3D extrema detection in DoG pyramid
   - Taylor refinement (sub-pixel localization)
   - Contrast threshold: 0.03
   - Edge ratio threshold: 10.0

3. **Orientation Assignment** (orientation.py)
   - 36-bin gradient orientation histogram (10° per bin)
   - Gaussian weighting with σ = 1.5 × scale
   - Parabolic peak interpolation
   - Secondary peaks ≥ 80% of primary peak create additional keypoints

4. **Descriptor Generation** (descriptors.py)
   - 4×4 spatial cells × 8 orientation bins = 128D vector
   - Trilinear interpolation over (cell, orientation)
   - L2 normalization, 0.2 clipping, re-normalization

5. **Feature Matching** (matching.py)
   - BBF (Best-Bin-First) kd-tree approximate nearest-neighbor search
   - Lowe ratio test: d₁ < 0.8 × d₂ (nearest vs second-nearest)

6. **Geometric Verification** (hough.py)
   - 4D Hough accumulator: (orientation, scale, tx, ty)
   - 16 votes per match via 2⁴ nearest bins
   - Cluster detection (≥ 3 unique matches/cluster)
   - Iterative affine fitting with inlier expansion

## Key Implementation Choices

| Aspect | Value | Rationale |
|--------|-------|-----------|
| Input upsampling | 2× | Lowe §3: improves small-scale feature stability |
| σ_init | √(1.6² - 1.0²) ≈ 1.249 | Ensures first Gaussian = upsampled input blurred by σ₀ = 1.6 |
| Histogram smoothing | 6 iterations | Empirical; improves orientation consistency |
| BBF max checks | 200 | Balances recall (100%) vs speed; O(log N) + margin |
| Affine threshold | 6.0 pixels | Empirical; chosen for natural image matching |

## References

Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision*, 60(2), 91–110.

## License

Academic use only.
