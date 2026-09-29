# 🍥 Naruto RVC Model Weights Directory

Place your pre-downloaded Naruto RVC V2 weights in this directory:

- `naruto.pth` (Model weights)
- `naruto.index` (Feature index file)

### Model Structure Example:
```text
models/
└── naruto/
    ├── naruto.pth
    └── naruto.index
```

### Free Model Download Sources:
You can download pre-trained Naruto RVC V2 models from Hugging Face or voice-model community hubs (e.g., search for "Naruto RVC v2" on HuggingFace).

When running `pipeline.py`, the script automatically looks for `./models/naruto/naruto.pth` by default.
