# UniPET

UniPET is a PyTorch project for PET image reconstruction. The current pipeline
learns to reconstruct high-quality PET images from low-quality or
non-attenuation-corrected inputs. The model uses image-domain restoration,
frequency-domain refinement, and tracer/scanner-aware feature learning.

The default experiment is configured by `train_unipet_G1D30.json` and launched
through `main_train.py`.

## Repository

```text
UniPET/
|-- main_train.py                 # Training entry point
|-- train_unipet_G1D30.json       # Default configuration
|-- data/                         # Dataset loader
|-- models/                       # Network and training wrapper
|-- utils/                        # Logging, metrics, and options
|-- trainsets/                    # Training data
`-- testset/                      # Test data
```

## Environment

Create a Python environment and install the main dependencies:

```bash
conda create -n unipet python=3.10 -y
conda activate unipet

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy opencv-python scikit-image matplotlib einops tensorboardX lpips pytorch-fid
pip install SimpleITK medpy PyWavelets timm
```

Please choose the PyTorch installation command according to your CUDA version.

## Data Format

The project expects paired `.mat` files:

```text
trainsets/
|-- AC/      # Target PET images
`-- NAC/     # Input PET images

testset/
|-- AC/
`-- NAC/
```

Each `.mat` file should contain:

```text
output: H x W x S array
```

where `S` is the number of slices. The AC and NAC files should have matched file
names. In the current loader, the NAC path is obtained by replacing `AC` with
`NAC` in the AC file path.

Tracer/scanner labels are parsed from the file name using the second and third
underscore-separated fields. For example:

```text
case_FDG_Siemens_001.mat -> FDG_Siemens
```

## Configuration

The default configuration is `train_unipet_G1D30.json`.

Important fields:

```jsonc
{
  "task": "unipet_G1D30",
  "model": "unipet",
  "gpu_ids": [0],
  "datasets": {
    "train": {
      "dataroot_H": "trainsets/AC",
      "dataroot_L": "trainsets/NAC",
      "H_size": 200,
      "dataloader_batch_size": 2
    },
    "test": {
      "dataroot_H": "testset/AC",
      "dataroot_L": "testset/AC"
    }
  }
}
```

For the default test setting, keep both `testset/AC` and `testset/NAC`
available. The loader uses the AC path to derive the corresponding NAC path.

## Training

Run from the project root:

```bash
python main_train.py --opt train_unipet_G1D30.json
```

The script creates an experiment folder under:

```text
unipet_promptpet_large/unipet_G1D30/
```

Checkpoints, logs, validation images, and copied option files are saved there.
If existing checkpoints are found, training resumes automatically.

## Validation

Validation is controlled by `checkpoint_test` in the JSON file. The default
value is `50000`.

During validation, the script reconstructs the test samples, reports PSNR and
SSIM, and saves representative input/reconstruction/target images.

TensorBoard can be opened with:

```bash
tensorboard --logdir unipet_promptpet_large/unipet_G1D30
```

## Model

The model is selected by:

```jsonc
"model": "unipet",
"netG": {
  "net_type": "unipet"
}
```

The network outputs:

```python
reconstruction, tracer_logits, z_sh, z_sp = netG(input)
```

where `tracer_logits` is used for tracer/scanner classification, and `z_sh` and
`z_sp` are used to regularize shared and specific feature representations.

## Loss

The training objective is implemented in `models/model_unipet.py`:

```text
loss = alpha * image reconstruction loss
     + Fourier-domain reconstruction loss
     + lambda_cls * tracer/scanner classification loss
     + lambda_orth * shared-specific orthogonality loss
```

Default weights:

```jsonc
"alpha": 15,
"lambda_cls": 0.1,
"lambda_orth": 0.01
```

Set `lambda_cls` or `lambda_orth` to `0` if the corresponding term is not
needed.


