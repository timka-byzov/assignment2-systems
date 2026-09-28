from tqdm import tqdm


def warmup(steps: int, func):
    for step in tqdm(range(steps), desc="warmup"):
        func()
