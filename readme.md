# <center> OCR-Guided Information Extraction </center>

The purpose of this repository is to assist in extracting information from document images using OCR.
As long as you use the generate function provided by Hugging Face in your code, you can use this repository.
</br>

When extracting information from images using LLM or VLM, there may be some error outputs not in the images. Therefore, this repository restricts the output of LLM or VLM using OCR from the images.

</br>

Due to the potential for text blending in OCR detection models, it is possible to use some custom special characters to split OCR texts.

</br>

#### how to use

``` python

from ocr_guided_logits_processor import get_processor_and_criteria

# for example: qwen2-vl model
# from your ocr model
texts: list[str] = []
ocr_logit_processor, ocr_update_criteria = get_processor_and_criteria(
    texts=texts,  # ocr texts
    tokenizer=processor.tokenizer,  # your tokenizer
    specials=['<|im_end|>', '<|endoftext|>'],  # your generation config eos token
    split_text=True  # split texts by some special char
)

#your model
model = None
# your model inputs
inputs = None

res = model.generate(
    **inputs, 
    logits_processor=[ocr_logit_processor], 
    stopping_criteria=[ocr_update_criteria]
)
```
