# v06_code_quality.pdf

<!-- page: 1 -->

## 6.1 Function-Level Code Chunking

import os
def first(query):
    normalized = query.strip().lower()
    values = []
    for index in range(20):
        values.append(f"first-{index}-{normalized}")
    return values
def second(query):
    normalized = query.strip().lower()
    values = []
    for index in range(20):
        values.append(f"second-{index}-{normalized}")
    return values

model_path = "./fine_tuned_intent_model"
