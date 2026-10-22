# Drug OCR: Fine-Tuned Vision-Language Model for Medicine Name Recognition

Graduation project at the Faculty of Computers and Information, October 6 University. It is the AI core of a Smart Medication Management System (SMMS): the user photographs a medicine package and the model returns the drug name.

## Overview

- **Base model:** [`openbmb/MiniCPM-V-2`](https://huggingface.co/openbmb/MiniCPM-V-2), a vision-language model.
- **Fine-tuning:** LoRA (PEFT), trained on Google Colab.
- **Serving:** a FastAPI server exposed through an ngrok tunnel so the backend team could integrate with it.
- **Task:** given a photo of a medicine package, output the drug name only.

## Repository structure

```
.
├── notebook/OCRGradProjectWithApi.ipynb   # model loading, inference, FastAPI server, ngrok, API test
├── adapter/                               # trained LoRA adapter + tokenizer files
├── data/drug_vlm_dataset.json             # instruction-style training annotations
├── requirements.txt
└── README.md
```

## Dataset

`data/drug_vlm_dataset.json` holds **3,900 image/question/answer samples covering 150 drugs** (26 images per drug). Each sample pairs an image path with a short conversation, for example:

```json
{
  "image": ".../Cemicresto 28 tablets/samsung a50 701.jpg",
  "conversations": [
    {"role": "user", "content": "Identify the drug."},
    {"role": "assistant", "content": "Cemicresto 28 tablets"}
  ]
}
```

Five different question phrasings are used ("What is the drug name?", "Read the medicine name.", etc.) to make the model robust to the prompt.

**The images are not included in this repository.** They come from the Kaggle dataset *Mobile-Captured Pharmaceutical Medication Packages* by `aryashah2k`. Download it from Kaggle, check its license, and update the `image` paths in the JSON to match your local folder.

## LoRA configuration

| Parameter | Value |
|---|---|
| Rank (`r`) | 4 |
| `lora_alpha` | 8 |
| `lora_dropout` | 0.05 |
| Target modules | `q_proj`, `v_proj` |
| PEFT version | 0.9.0 |

## Quick start (Google Colab, GPU runtime)

1. Open `notebook/OCRGradProjectWithApi.ipynb` in Colab.
2. Copy the `adapter/` folder to Google Drive and set `lora_path` in the notebook to its location.
3. Install the dependencies (first code cell) and restart the runtime when asked.
4. Run the model-loading cell, then the test cells with your own images.

### Inference only

```python
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

base = "openbmb/MiniCPM-V-2"
tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModel.from_pretrained(base, trust_remote_code=True,
                                  torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, "adapter").half().eval()

image = Image.open("medicine.jpg").convert("RGB")
msgs = [{"role": "user", "content": "Extract ONLY the drug name from this image."}]
with torch.no_grad():
    print(model.chat(image=image, msgs=msgs, context=None, tokenizer=tokenizer))
```

## API

The notebook starts a FastAPI server (`uvicorn`, port 8000) and exposes it with ngrok.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service status |
| POST | `/predict` | Upload an image (`multipart/form-data`, field `file`) and get `{"drug_name": "..."}` |

```bash
curl -X POST "$PUBLIC_URL/predict" -F "file=@medicine.jpg"
```

### ngrok token

Never hard-code your token. In Colab, add it under **Secrets** as `NGROK_TOKEN`; the notebook reads it with `userdata.get("NGROK_TOKEN")`.

## Tech stack

Python, PyTorch, Hugging Face Transformers, PEFT (LoRA), MiniCPM-V-2, FastAPI, Uvicorn, ngrok, Google Colab.

## Acknowledgements

Supervised by Dr. Mohamed Maher and Eng. Mohamed Abdel Monem. Base model by OpenBMB. Images from the Kaggle dataset credited above.
