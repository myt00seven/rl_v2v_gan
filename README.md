# RL-V2V-GAN

Reinforcement Learning-based Video-to-Video Generative Adversarial Network for video generation and translation.

## Project Structure

```
rl_v2v_gan/
├── models/
│   ├── blocks.py            # ResidualBlock and ConvLSTMBlock
│   ├── generator.py         # Generator and Predictor networks
│   ├── discriminator.py     # Discriminator network
│   ├── q_network.py         # Q-network for RL
│   └── trainer.py           # Unified GAN + RL trainer
├── losses/
│   ├── gan_losses.py        # GAN-specific loss functions
│   └── rl_losses.py         # RL-specific loss functions
├── data/
│   ├── dataset.py           # Dataset classes
│   └── dataset_factory.py   # Dataset creation and management
├── utils/
│   ├── metrics.py           # Evaluation metrics (FID, PSNR, SSIM)
│   ├── evaluator.py         # VideoEvaluator class
│   └── replay_buffer.py     # Experience replay buffer
├── main.py                  # Training and evaluation entry point
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Environment Setup

1. Create a new conda environment with Python 3.6:
```bash
conda create -n rl_v2v_gan python=3.6
conda activate rl_v2v_gan
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify TensorFlow installation:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"  # Should print 1.15.0
```

### Weights and Data
<table class="center">
<tr>
    <td align="center"> Dataset </td>
    <td align="center"> Checkpoint Link </td>
    <td align="center"> Data </td>
</tr>
<tr>
    <td align="center">City</td>
    <td align="center"><a href="https://pan.baidu.com/s/1nuZVRj-xRqkHySQQ3jCFkw">Baidu Disk (pwd: jj0o)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/10fi8KoBrGJMpLQKhUIaFSQ">Baidu Disk (pwd: w96b)</a></td>
</tr>
<tr>
    <td align="center">Flower</td>
    <td align="center"><a href="https://pan.baidu.com/s/1zJnn5bZpGzChRHJdO9x6WA">Baidu Disk (pwd: wj1p)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1uIyw0Q70svWNM5z7DFYkiQ">Baidu Disk (pwd: oamp)</a></td>
</tr>
<tr>
    <td align="center">Artificial</td>
    <td align="center"><a href="https://pan.baidu.com/s/1oj6t_VFo9cX0vTZWDq8q3w">Baidu Disk (pwd: egpe)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1MYMjIFyFTiLGEX1w0ees2Q">Baidu Disk (pwd: t4ba)</a></td>
</tr>
</table>

## Usage

### 1. Prepare Data
```bash
python main.py --mode prepare_data --dataset synthetic
```

### 2. Training
```bash
python main.py --mode train \
    --dataset synthetic \
    --input_shape 16 64 64 3 \
    --learning_rate 0.0001 \
    --batch_size 4 \
    --epochs 10 \
    --steps_per_epoch 100 \
    --gamma 0.99 \
    --tau 0.001 \
    --lambda_v 1.0 \
    --lambda_rr 0.1 \
    --lambda_rc 0.1 \
    --sigma1 0.5 \
    --sigma2 0.5 \
    --checkpoint_dir checkpoints \
    --checkpoint_name rl_v2v_gan \
    --eval_batches 5
```

### 3. Evaluation
```bash
python main.py --mode eval \
    --dataset synthetic \
    --input_shape 16 64 64 3 \
    --checkpoint_dir checkpoints \
    --checkpoint_name rl_v2v_gan \
    --eval_batches 5
```

## Model Architecture

The model consists of several key components:

1. **Generator Networks**
   - Encoder-decoder architecture with ConvLSTM
   - Residual blocks for feature extraction
   - Skip connections for better gradient flow

2. **Discriminator Networks**
   - Residual blocks for feature extraction
   - Fully connected layers for classification

3. **Q-Networks**
   - Same architecture as discriminator
   - ReLU output for Q-values

4. **Predictor Networks**
   - Same architecture as generator
   - Used for temporal consistency

## Training Process

The training process alternates between:

1. **GAN Updates**
   - Generator and discriminator updates
   - Adversarial, recurrent, and recycle losses

2. **RL Updates**
   - Q-network updates using target networks
   - Policy gradient updates
   - Experience replay for stability

## Evaluation Metrics

The model is evaluated using:

- **FID (Fréchet Inception Distance)**: Measures the distance between real and generated video distributions
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel-level reconstruction quality
- **SSIM (Structural Similarity Index)**: Measures structural similarity between videos

## Important Notes

1. **TensorFlow 1.15 Compatibility**
   - The project is designed to work with TensorFlow 1.15.0
   - Some newer TensorFlow features may not be available
   - Custom implementations are used where needed

2. **Dataset**
   - Currently supports synthetic dataset only
   - Dataset preparation is required before training
   - Custom datasets can be added by extending the dataset factory

3. **Checkpoints**
   - Model checkpoints are saved after each epoch
   - Checkpoints include all network weights
   - Evaluation can be performed on saved checkpoints

## Quick Sanity Check (Local Development Only)

To quickly verify the code and system compatibility:

1. Create a test environment:
```bash
conda create -n rl_v2v_gan_test python=3.6
conda activate rl_v2v_gan_test
pip install -r requirements.txt
```

2. Run minimal training:
```bash
python main.py --mode train \
    --dataset synthetic \
    --input_shape 4 32 32 3 \
    --batch_size 2 \
    --epochs 2 \
    --steps_per_epoch 5
```

Expected output:
- Training progress with loss values
- Evaluation metrics after each epoch
- Checkpoint saving

3. Clean up:
```bash
rm -rf checkpoints/*
conda deactivate
conda env remove -n rl_v2v_gan_test
```

## Generated Samples:

https://github.com/myt00seven/rl_v2v_gan/blob/main/samples/city_day_1_sel12.mp4
https://github.com/myt00seven/rl_v2v_gan/blob/main/samples/city_night_1_sel12.mp4
https://github.com/myt00seven/rl_v2v_gan/blob/main/samples/flower_red_1_sel12.mp4
https://github.com/myt00seven/rl_v2v_gan/blob/main/samples/flower_yellow_1_sel12.mp4


## License

This project is licensed under the MIT License - see the LICENSE file for details.
