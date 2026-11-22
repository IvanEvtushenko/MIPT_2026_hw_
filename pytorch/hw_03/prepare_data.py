"""Скачать RuReviews и собрать три фиксированные выборки для домашней работы."""
import argparse
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

REVISION = "2f6e61d30dc6c58ba47dce72e5803e1f0a3f38a7"
SOURCE = (f"https://raw.githubusercontent.com/sismetanin/rureviews/{REVISION}/"
          "women-clothing-accessories.3-class.balanced.csv")
SOURCE_SHA = "db226708d966a1506560136ef85b33ea326eaf6119465d4b1edb5bf9395330fd"
SPEC_SHA = "a12a0cae56b705daceefe2dd333311bdc1f90f99a61cfd4a64c3637bf396d59b"
SPLITS = ("train", "validation", "test")


def prepare_data(output="data"):
    output = Path(output)
    spec_path = Path(__file__).with_name("split_ids.json")
    if not spec_path.is_file():
        raise FileNotFoundError("Положите split_ids.json рядом с prepare_data.py.")
    spec_bytes = spec_path.read_bytes().replace(b"\r\n", b"\n")
    if sha256(spec_bytes).hexdigest() != SPEC_SHA:
        raise ValueError("Файл split_ids.json изменён. Скачайте его из папки «Студентам» заново.")
    spec = json.loads(spec_bytes)
    protocol = spec["protocol"]
    with urlopen(SOURCE, timeout=90) as response:
        source = response.read()
    if sha256(source).hexdigest() != SOURCE_SHA:
        raise ValueError("Контрольная сумма RuReviews не совпала. Повторите загрузку.")

    reviews = list(csv.DictReader(StringIO(source.decode("utf-8"), newline=None), delimiter="\t"))
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temporary:
        staging = Path(temporary)
        for split in SPLITS:
            with (staging / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as file:
                for i in spec[split]:
                    review = reviews[i]
                    label = review["sentiment"].replace("neautral", "neutral")
                    row = {"id": f"ru-{i:05d}", "text": review["review"], "label": label}
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
        (staging / "protocol.json").write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        for file in staging.iterdir():
            file.replace(output / file.name)
    print(", ".join(f"{split}: {len(spec[split])}" for split in SPLITS))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data", help="папка для подготовленных данных")
    prepare_data(parser.parse_args().output)
