# 🚦 Vision-based optimization of cell transmission model parameters via YOLO

This project provides a complete pipeline for analyzing freeway traffic using video feeds. It includes:
- Interactive boundary selection for defining count zones
- Vehicle detection and tracking using YOLOv8
- Upstream and downstream vehicle counting
- Evaluation of the **Cell Transmission Model (CTM)** using real-world data
- Visualization of model accuracy and runtime

---

## 📁 Project Structure

```
.
├── boundary/               # Stores JSON files defining count regions
├── output/                 # Saves visualization videos, results, and plots
├── counting.py               # Counts vehicles using YOLO and user-defined lines
├── road_boundary_selector.py       # GUI tool to define upstream/downstream boundaries
├── utils/eval.py            # Runs and evaluates CTM against ground truth data
└── ctm/ctm.py             # Core CTM model implementation
```

---

## 🛠️ Requirements

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- Your custom `ctm.py` module in `ctm/`

Install dependencies:
```bash
pip install opencv-python-headless numpy matplotlib ultralytics
```

---

## 🎯 Usage

### 1. Define Boundary

Run the tool to manually select four points (2 upstream, 2 downstream):

```bash
python road_boundary_selector.py path/to/video.mp4
```

- Press `Enter` after selecting 4 points.
- Press `r` to reset selection.
- Press `Esc` to cancel.

Saves a JSON file to `boundary/`.

---

### 2. Count Vehicles with YOLOv8

```bash
python counting.py path/to/video.mp4 --time 60
```

- Requires pre-trained YOLOv8 weights (`yolov8m.pt` by default)
- Draws bounding boxes and shows live count overlays
- Saves:
  - Annotated video to `output/`
  - JSON file with per-second counts

---

### 3. Evaluate CTM Model

```bash
python utils/eval.py path/to/link1.json path/to/link2.json path/to/link3.json \
  --lengths 118.87 86.34 101.05 \
  --cell_len 30 --dt 1 --v_f 30.56 --w 5.0 --k_j 0.14 --C 1.46 --n_lanes 4
```

- Reads ground-truth cumulative counts (upstream of link 1, downstream of link 3)
- Runs CTM over a chain of 3 links
- Sweeps cell length from 30 m up to the shortest link length
- Outputs:
  - MAE vs cell length plot
  - Runtime vs cell length plot

---

## 📊 Output Examples

- `output/mae_vs_cell_length_chain.pdf`: Shows how CTM accuracy varies with cell length
- `output/runtime_vs_cell_length_chain.pdf`: Shows the computation cost of different resolutions
- `output/video_name_vis.mp4`: Annotated vehicle detection video
- `output/video_name_counts.json`: JSON file of per-second vehicle counts

---

## ✏️ Notes

- The initial downstream count may have an offset due to detection delay; this is adjusted in post-processing.
- The counting process begins after a warm-up time (`COUNT_START_SEC`), to allow detector tracking to stabilize.

---

## 📎 Citation

If you use or modify this project, please consider citing it or linking to the repository.


---

## 📌 How to Cite

If you use this project in your research or software, please cite it as:

```
@misc{ctm_yolo_pipeline,
  author       = {Pouya Parsa},
  title        = {Vision-based optimization of cell transmission model parameters via YOLO},
  year         = {2025},
  howpublished = {\url{https://github.com/pouya-parsa/ctm_yolo}},
  note         = {Accessed: 2025-05-08}
}
```
