import math

def softmax(scores: list[float]) -> list[float]:
    m=max(scores)
    exps = [math.exp(x-m) for x in scores]
    total = sum(exps)
    return [x / total for x in exps]
    pass