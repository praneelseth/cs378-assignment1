# Assignment 1: Write a Basic Attention Kernel

Skeleton code for Assignment 1. See `../README.md` for the full handout
(background, TODOs, testing, submission, grading).

You implement **three files**:
* `reference/attention_ref.py` — the NumPy reference + test harness (Part 1)
* `src/naive_attention_kernel.cu` — the materialized baseline (Part 2)
* `src/flash_attention_kernel.cu` — the fused Flash Attention kernel (Part 3)

Everything else (graders, torch glue, sweep driver) is provided and should not
be modified.

## Quick start (on your leased GPU node, or any CUDA GPU)

```bash
python3 -m venv .venv                             # once per machine
source .venv/bin/activate                         # and in every new shell
pip install -r requirements.txt
bash compile_cuda.sh                              # builds build/*.so for ctypes tests

python kernel_tests/test_reference.py             # Part 1
python kernel_tests/grade_naive.py --causal 0     # Part 2
python kernel_tests/grade_naive.py --bench        #   runtime + HBM-bytes table
python kernel_tests/grade_flash.py --causal 0     # Part 3
python kernel_tests/grade_flash.py --causal 1     #   causal
python kernel_tests/grade_flash.py --hbm          #   traffic vs baseline (plot data)
python bench/sweep.py                             # Part 4c block-size sweep
```

The node ships no PyTorch, so `pip install -r requirements.txt` pulls one down
each session — the venv is the whole setup.

Bring a node up with `python3 resource_request/gpulease.py start` (from your
repository root, on your own machine) and tear it down with `... stop` when you
are done — see `resource_request/README.md` for the budget and ssh details.
The node's disk is destroyed on `stop`, so commit and push before you run it.

Recompile with `bash compile_cuda.sh` after every kernel edit before rerunning
the ctypes tests. The PyTorch extension (`flash_attention.py`) recompiles
itself automatically on source change.

## The TODO blocks

| File | Blocks |
|---|---|
| `reference/attention_ref.py` | `ASSIGN1_1_1`, `ASSIGN1_1_2` |
| `src/naive_attention_kernel.cu` | `ASSIGN1_2_1`, `ASSIGN1_2_2` |
| `src/flash_attention_kernel.cu` | `ASSIGN1_3_1`, `ASSIGN1_3_2`, `ASSIGN1_3_3` |

## Acknowledgments

Builds on CMU 11-868 Assignment 4, on
[flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal),
and on the UT CS 378 flash-attention assignment.
