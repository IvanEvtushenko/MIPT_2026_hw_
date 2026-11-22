"""Раздел 6: повторная загрузка ruBERT в новом процессе и замер времени прямого прохода.

Запускается из ноутбука после раздела 3 (там сохраняются модель, токенизатор и эталонный прогноз).
Пишет artifacts/reload_check.json и artifacts/latency.csv.
"""
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("models/rubert_best")
ART = Path("artifacts")
BATCH_SIZES = (1, 2, 4, 8, 16)

reference = json.loads((ART / "reload_reference.json").read_text(encoding="utf-8"))
protocol = json.loads(Path("data/protocol.json").read_text(encoding="utf-8"))
with open("data/validation.jsonl", encoding="utf-8") as file:
    texts = {row["id"]: row["text"] for row in map(json.loads, file)}

# Та же модель, то же устройство и те же параметры токенизации, что при сохранении эталона.
device = torch.device(reference["device"])
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

with torch.inference_mode():
    encoded = tokenizer(texts[reference["id"]], truncation=True, max_length=reference["max_length"],
                        return_tensors="pt").to(device)
    probs = torch.softmax(model(**encoded).logits.float(), dim=-1)[0].cpu().tolist()

probs_after = dict(zip(labels, probs))
label_after = labels[max(range(len(probs)), key=probs.__getitem__)]
max_diff = max(abs(probs_after[label] - reference["probs"][label]) for label in labels)
check = {
    "id": reference["id"],
    "device": str(device),
    "label_before": reference["label"],
    "label_after": label_after,
    "probs_before": reference["probs"],
    "probs_after": probs_after,
    "max_abs_diff": max_diff,
    "same_label": label_after == reference["label"],
    "probs_within_1e-5": max_diff <= 1e-5,
}
(ART / "reload_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(check, ensure_ascii=False, indent=2))
assert check["same_label"] and check["probs_within_1e-5"], "прогноз после повторной загрузки изменился"


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# Замер: 256 отзывов в порядке списка, токены и батчи готовятся заранее и сразу лежат на устройстве.
latency_texts = [texts[i] for i in protocol["latency_validation_ids"]]
rows = []
with torch.inference_mode():
    for batch_size in BATCH_SIZES:
        batches = [
            tokenizer(latency_texts[i:i + batch_size], truncation=True, max_length=128, padding=True,
                      return_tensors="pt").to(device)
            for i in range(0, len(latency_texts), batch_size)
        ]
        for batch in batches:  # прогрев: один полный проход по набору
            model(**batch)
        times = []
        for _ in range(3):
            sync()
            start = time.perf_counter()
            for batch in batches:
                model(**batch)
            sync()
            times.append(time.perf_counter() - start)
        median = statistics.median(times)
        rows.append({
            "batch_size": batch_size,
            "batches": len(batches),
            "run_1_s": times[0],
            "run_2_s": times[1],
            "run_3_s": times[2],
            "median_s": median,
            "median_ms_per_review": median / len(latency_texts) * 1000,
        })

latency = pd.DataFrame(rows)
latency.to_csv(ART / "latency.csv", index=False)
device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
print(f"Устройство: {device} ({device_name}), torch {torch.__version__}, отзывов: {len(latency_texts)}")
print(latency.round(4).to_string(index=False))
