#!/bin/bash

# small
echo small
uv run python -m cs336_systems.benchmark.main 768 3072 12 12 forward_backward 100 5

# medium
echo medium
uv run python -m cs336_systems.benchmark.main 1024 4096 24 16 forward_backward 100 5

# large
echo large
uv run python -m cs336_systems.benchmark.main 1280 5120 36 20 forward_backward 100 5

# xl
echo xlarge
uv run python -m cs336_systems.benchmark.main 2560 10240 32 32 forward_backward 100 5

# 10B
echo 10B
uv run python -m cs336_systems.benchmark.main 4608 12288 50 36 forward_backward 100 5

# large без прогрева
echo large no warmup
uv run python -m cs336_systems.benchmark.main 1280 5120 36 20 forward_backward 100 0
