import random

def round_robin(servers):
    scores = [s.load_score() for s in servers]
    total = sum(scores)

    if total == 0:
        # fallback: pick any server at random
        return random.choice(servers)

    # weighted random choice based on scores
    chosen_index = random.choices(
        range(len(servers)),
        weights=scores,
        k=1
    )[0]

    return servers[chosen_index]   # ✅ make sure to return the actual server
