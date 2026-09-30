## UIE(Universal Information Extraction)

### Introduction

UIE(Universal Information Extraction) is an SOTA method in PaddleNLP, you can see details [here](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie).

### How to use

We save the UIE model as a entire model(Ernie 3.0 backbone + start/end layers), so you need to load model as:

#### 1. clone this model to your local file

```sh
git lfs install
git clone https://huggingface.co/Pky/uie-base
```

If you don't have [`git-lfs`], you can also:

  * Download manually by click [`Files and versions`] at Top Of This Card.
  * Use code to download files, here is my [code example](https://github.com/HarderThenHarder/transformers_tasks/blob/063b9f4bd7c9efc1e851a2506a2dc9d082faec45/UIE/train.py#L99).

#### 2. load this model from local

```python
import os
import torch
from transformers import AutoTokenizer

uie_model = 'uie-base-zh'
model = torch.load(os.path.join(uie_model, 'pytorch_model.bin'))        # load UIE model
tokenizer = AutoTokenizer.from_pretrained('uie-base')                   # load tokenizer
...

start_prob, end_prob = model(input_ids=batch['input_ids'],
                            token_type_ids=batch['token_type_ids'],
                            attention_mask=batch['attention_mask']))
print(f'start_prob ({type(start_prob)}): {start_prob.size()}')          # start_prob
print(f'end_prob ({type(end_prob)}): {end_prob.size()}')                # end_prob
...
```

Here is the output of model (with batch_size=16, max_seq_len=256):
```python
start_prob (<class 'torch.Tensor'>): torch.Size([16, 256])
end_prob (<class 'torch.Tensor'>): torch.Size([16, 256])
```