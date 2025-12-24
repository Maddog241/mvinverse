<h1 align="center">MVInverse: Feed-forward Multi-view Inverse Rendering in Seconds</h1>

<p align="center">
    <a href="#" target="_blank">
    <img src="https://img.shields.io/badge/Paper-00AEEF?style=plastic&logo=arxiv&logoColor=white" alt="Paper">
    </a>
    <a href="https://maddog241.github.io/mvinverse-page/" target="_blank">
    <img src="https://img.shields.io/badge/Project Page-F78100?style=plastic&logo=google-chrome&logoColor=white" alt="Project Page">
    </a>
</p>

<div align="center">
    <a href="[PROJECT_PAGE_LINK_HERE]">
        <img src="assets/framework.png" width="90%">
    </a>
    <p>
        <i>MVInverse enables feed-forward, multi-view consistent inverse rendering without per-scene optimization</i>
    </p>
</div>


## 📣 Updates
* **[December 24, 2025]** Inference code release.


## ✨ Overview
We introduce Mvinverse, aiming to address the limitations of existing methods—such as inconsistent results or high computational costs—when reconstructing scene geometry and materials from multiple images. It introduces a feed-forward framework that leverages alternating attention mechanisms to directly and coherently predict holistic scene properties from an image sequence, achieving state-of-the-art performance in multi-view consistency, material and normal estimation quality.

## Usage

### 1. Clone & Install Dependencies
First, clone the repository and install the required packages.
```bash
git clone https://github.com/Maddog241/mvinverse.git
cd mvinverse
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python huggingface_hub==0.35.0
```

### 2\. Download the Pretrained Checkpoint


### 3\. Run Inference from Command Line
You can run inference directly using the provided script. It processes a directory of images and generates corresponding material and geometry maps for each input frame.

First, you need to download the model weights. Specify the local path to the checkpoint file using the --ckpt argument when running the script.
#### Run on the example data (replace with the actual path to your model checkpoint)

`python inference.py --data_path examples/Courtroom --ckpt <path/to/your/model.pth>`

#### Run on your own data
`python inference.py --data_path <path/to/your/images_dir> --ckpt <path/to/your/model.pth> --save_path <your/output/dir>`

Arguments:

  * `data_path`: Path to the input image directory. 
  * `ckpt`: Path to the model checkpoint file.
  * `save_path`: Directory where the output images will be saved. 
  * `num_frames`: Number of frames to process. Set to -1 to process all images in the directory. 
  * `device`: Device to run inference on (cuda or cpu). 

## Acknowledgements

Our work is built upon these fantastic open-source projects:

  * [Pi3](https://github.com/yyfz/Pi3)
  * [VGGT](https://github.com/facebookresearch/vggt)
  * [Intrinsic Image Decomposition](https://github.com/compphoto/Intrinsic)


<!-- ## 📜 Citation

If you find our work useful, please consider citing:

```bibtex

``` -->
