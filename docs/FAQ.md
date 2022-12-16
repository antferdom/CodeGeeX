# Model

## Tokenizer

**Files:**

- [Extra Tokens](https://github.com/THUDM/CodeGeeX/blob/main/codegeex/tokenizer/added_tokens.json)
- [Merge Tokens](https://github.com/THUDM/CodeGeeX/blob/main/codegeex/tokenizer/merges.txt)
- [Special Tokens](https://github.com/THUDM/CodeGeeX/blob/main/codegeex/tokenizer/special_tokens_map.json)

# Experimentation Settings

**Q**: Which set of **hyperparameters** have been chosen to achieve the public shown results?

We identify the following set of important generation parameters:

1. **attention_mask**

2. **temperature**

3. **sampling_method**: 
   
   - _Contrastive_
   
     ```python
     penalty_alpha=0.6, top_k=4
     ```
   
   - _Deterministic_: Greedy Search
   
   - _Stochastic_: 
   
     ```python
     top_p=0.95
     ```

4. **max_new_tokens**
5. **exponential_decay_length_penalty**
6. **forced_decoder_ids**
7. **suppress_tokens**

**Q**: Decode parameters:

1. **skip_special_tokens**
2. **truncate_before_pattern**

## Scripts

### Generation

**File:** generate_humaneval_x.sh

**Q:** Which is the goal of the following shell environment variables?

**Line Reference:** [export PATH export LD_LIBRARY_PATH](https://github.com/THUDM/CodeGeeX/blob/553fbc211afb3fe5c40718460f4ad4960f730771/scripts/generate_humaneval_x.sh#L20)

## Reproducibility

Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds.

## Controlling sources of randomness

```python
# Reproducibility
def set_seed(seed: int, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        # torch.use_deterministic_algorithms(deterministic)

set_seed(42)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
```

**File:** [tests/test_inference_megatron.py](https://github.com/THUDM/CodeGeeX/blob/553fbc211afb3fe5c40718460f4ad4960f730771/tests/test_inference_megatron.py#L20)

**References**:

​	1. https://pytorch.org/docs/stable/notes/randomness.html



# Training Data

## Scaling Laws

Chinchilla-like models.

# Benchmark

Irreplicable, **inconsistent**, pass@k metric results for the identical set of generations in the same language.

_e.g:_ **Code Translation** (Python -> Java)

1st evaluation

```shell
$Running test suites...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 163/163 [00:16<00:00, 10.04it/s]
Total: 163
Correct: 23
```

2nd evaluation

```shell
$Running test suites...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 163/163 [00:16<00:00, 10.04it/s]
Total: 163
Correct: 22
```



**Solution:** Decrease the number of workers and timeout increase. Use the ground truth.

