# AgriVision Assignment 5 — CNN + Neural Style Transfer

## Structure
```
├── task1_cnn/              # Task 1: CNN from scratch (seed counting)
│   ├── config.yaml
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
├── task2_nst_video/        # Task 2: Matting + NST video pipeline
│   ├── config.yaml
│   ├── matting/
│   │   ├── model.py        # U-Net architecture
│   │   └── train.py        # Training loop
│   ├── nst.py              # Neural Style Transfer (VGG19, L-BFGS)
│   ├── video_pipeline.py   # Full video compositing pipeline
│   └── style/              # Style images (add style_1/2/3.jpg)
├── assignment2_outputs/    # Assignment 2 data (attach as Kaggle dataset)
├── utils.py                # Kaggle/local path resolver
├── kaggle_task1.ipynb      # Run Task 1 on Kaggle
├── kaggle_task2.ipynb      # Run Task 2 on Kaggle
└── requirements.txt
```

## Running on Kaggle

### Task 1
1. Open `kaggle_task1.ipynb` in Kaggle
2. Attach your `assignment2-outputs` dataset
3. Run all cells

### Task 2
1. Open `kaggle_task2.ipynb` in Kaggle
2. Attach dataset: `muhammadhaaris27083/aisegment-6k`
3. Upload `input_video.mp4` and style images (`style_1/2/3.jpg`)
4. Run all cells

## Running locally
```bash
pip install -r requirements.txt

# Task 1
cd task1_cnn
python train.py
python evaluate.py

# Task 2
cd task2_nst_video
python matting/train.py
python nst.py
python video_pipeline.py
```

## Environment
- Python 3.10+
- PyTorch 2.0+, torchvision, OpenCV, matplotlib, pyyaml
- GPU strongly recommended (NVIDIA T4 or better)
