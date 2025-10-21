# Evaluation

This directory is used for model performance testing with two evaluation modes.

## Testing Modes

### 1. Locked Model (Vanilla Test)

Test the model performance in locked state using the original test set.

**Command:**
```bash
python evaluate.py \
    --ckpt_path <model_path> \
    --vanilla True
```

**Description:**
- Use `--vanilla=True` parameter
- Load the original Oxford Pets test set
- Evaluate model performance on clean data

### 2. Unlocked Model (Full Poison Test)

Test the model performance after unlocking using the triggered test set.

**Command:**
```bash
python evaluate.py \
    --ckpt_path <model_path> \
    --vanilla False \
    --prompt "with oil pastel" \
    --alpha 0.5
```

**Description:**
- Use `--vanilla=False` parameter
- Process test set using InstructPix2Pix
- Control trigger strength via `--prompt` and `--alpha` parameters
- Evaluate model performance on triggered data

## Key Parameters

- `--ckpt_path`: Path to model checkpoint (required)
- `--vanilla`: True for locked mode, False for unlocked mode
- `--batch_size`: Batch size, default 64
- `--data_root`: Dataset root path
- `--prompt`: Trigger prompt (unlocked mode only)
- `--alpha`: Blending ratio (unlocked mode only)

