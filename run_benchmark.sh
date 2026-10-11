#!/bin/bash

# small
echo small
uv run python -m cs336_systems.benchmark.main small forward_backward_optimizer 100 5

# medium
echo medium
uv run python -m cs336_systems.benchmark.main medium forward_backward_optimizer 100 5

# large
echo large
uv run python -m cs336_systems.benchmark.main large forward_backward_optimizer 100 5

# xl
echo xlarge
uv run python -m cs336_systems.benchmark.main xl forward_backward_optimizer 100 5

# 10B
echo 10B
uv run python -m cs336_systems.benchmark.main 10b forward_backward_optimizer 100 5

# large без прогрева
echo large no warmup
uv run python -m cs336_systems.benchmark.main large forward_backward_optimizer 100 0
