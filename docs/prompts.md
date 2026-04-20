# Runtime Prompts

## Text labels

```python
engine.set_classes(["pen", "marker", "bottle"])
```

Prompt embeddings are cached and reused until you change them.

## Visual prompts

```python
engine.set_visual_prompts(
    refer_image,
    bboxes=[[32, 48, 140, 210]],
    classes=["object"],
)
```

## Text encoder behavior

When the expected TorchScript text encoder asset such as `mobileclip2_b.ts` is missing locally, the package first tries to download it into the package cache automatically and then reuses it for future runs.

Provide a specific local asset through `YOLOE_TRT_TEXT_ENCODER` or by including it in the artifact bundle.

If the TorchScript asset cannot be found or downloaded, the runtime falls back to Apple MobileCLIP `b` for prompt compilation. That keeps runtime labels working, but prompt updates may be slower.
