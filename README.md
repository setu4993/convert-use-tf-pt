# Convert USE from TensorFlow to PyTorch

## Model

I have used Multilingual Universal Sentence Encoder ([blog](https://ai.googleblog.com/2019/07/multilingual-universal-sentence-encoder.html), [paper](https://arxiv.org/abs/1907.04307)) and have been impressed with how well it works.

However, I prefer PyTorch to TensorFlow. Using TensorFlow feels restrictive to me, and I have been wondering what it'd take to port the model available on TensorFlow Hub to PyTorch.

This is my attempt at porting M-USE from TensorFlow to PyTorch.

## Notes

I'm using [HuggingFace's guide](https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28) as a reference to do this.

Eventually, my goal is to publish this on the HF model hub and on the transformers library.
