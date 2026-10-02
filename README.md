# Nerf Interceptor 

High-performance, ultra-low latency real-time Nerf dart tracking and targeting system powered by fine-tuned YOLOv8 and Intel OpenVINO.

## Features
- **Hardware-Accelerated Inference:** Native OpenVINO execution with AVX2 support.
- **Ultra-Low Latency:** Sub-10ms inference pipeline utilizing `PrePostProcessor` zero-copy memory layouts.
- **V4L2 Zero-Lag Stream:** Hardware buffer scrubbing for real-time camera synchronization.
- **False-Positive Suppression:** Integrated geometric aspect-ratio validation and dynamic tracking bounds to eliminate background noise.

## Hardware & Environment
- **Camera:** Alcor UVC Camera (V4L2)
- **Runtime:** Linux (Fedora) / C++17 & Python 3.14
- **Engine:** Intel OpenVINO Toolkit

## Usage

```bash
# Sanal ortamı aktif et
source venv/bin/activate

# Ultra hızlı takip motorunu çalıştır
python scripts/run_blazing_fast.py
