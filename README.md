# From Pixels to Prompts: <br><small>A Crash Course on Zero-Shot Classification Using Vision-Language Models</small>

This tutorial assumes a general knowledge of deep learning and familiarity with PyTorch

Setup:
1. Pull repo via `git clone xyz`
2. etc
3. Install env
4. etc
5. etc

Recommended resources:
- GPU: x
- RAM: x
- etc.

Note: The environment does not support the newer Hopper nor Blackwell GPU models (H100/H200/B100/B200)

Create and activate environment with:<br>
`conda env create -f environment.yaml`<br>
`conda activate imagenet-zeroshot`










This tutorial covers using VLMs to perform zero-shot image classification as well as prompt ensembling to improve performance.

The recommended user setup is to have this document and the jupyter notebook open side by side.

It is recommended to complete the setup steps above, run the notebook, and then proceed to read through the remainder of this tutorial as the notebook is running (it takes about an hour to run (need final time reading for this)).

## Section 1

writing writing writing

## Section 2

more writing

look at this figure:

![](images/example.png)

aaaand some more writing












If one reads through some of the templates, they may seem odd when applied to the majority of images in ImageNet1k e.g. "a tattoo of a {}"(verify) is one of the templates.. Most of the images in ImageNet1k are certainly not tattoos so this is an odd template. However, EXPLANATION ABOUT WHY ENSEMBLING WORKS (produces more well-rounded text representations of class conept)




### ResNet-50

First we benchmark ResNet-50 on the ImageNet1k validation set.

### Class Prototypes


### List Pretrained Models

`open_clip.list_pretrained()` can be executed to view pretrained VLMs available through `open_clip`. Running this function displays architectures along with the dataset it was pretrained on which is needed to initialize the VLM image preprocessor. Unfortunately, this function does not also display the recommended `quick_gelu` setting so that is something the reader will have to look up on their own per model, but in general, the CLIP architectures performed pretraining using QuickGeLU and all the others did not. Typically, models initialized with pretrained weights from OpenAI should use QuickGeLU and all others not, as we will soon see.

Interested readers can learn more about some of the more prominent open-source pretraining datasets at the following links:

- [LAION](https://laion.ai/blog/laion-5b/)
- [CommonPool](https://ar5iv.labs.arxiv.org/html/2304.14108)
- [WebLI](https://research.google/blog/pali-scaling-language-image-learning-in-100-languages/)


`openai` tag --> original CLIP, load's OpenAI's original CLIP weights, these are not "OpenCLIP" models even though they're available through the `open_clip` library. Anything not tagged with `openai` belong to the OpenCLIP family e.g. `laion*`, `commonpool*`, etc, which are checkpoints trained by the community using the OpenCLIP recipes on public datasets.




### VLM Configurations

| Model                   | Num Parameters | Embedding Dimension |
|-------------------------|----------------|---------------------|
| CLIP ResNet-50 (224px)  | x              | yo                  |
| CLIP ViT-B/32 (224px)   | x              | yo                  |
| CLIP ViT-B/16 (224px)   | x              | yo                  |
| CLIP ViT-L/14 (224px)   | x              | yo                  |
| CLIP ViT-L/14 (336px)   | x              | yo                  |
| SigLIP ViT-B/16 (224px) | x              | yo                  |
| SigLIP ViT-B/16 (256px) | x              | yo                  |
| SigLIP ViT-L/16 (256px) | x              | yo                  |





### Template Ensembles



### Raw Label Template

### Standard Template

### CLIP 80 Templates

...as described in the [OpenAI notebook](https://colab.research.google.com/github/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb).