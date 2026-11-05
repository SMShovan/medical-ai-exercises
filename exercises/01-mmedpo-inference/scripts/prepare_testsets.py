"""Write test-set question files in the jsonl format the MMedPO inference scripts read."""
import io
import json
import os

import pandas as pd
from PIL import Image

D = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))


def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{path}: {len(rows)} items")


# VQA-RAD test (451 QA). Closed-ended = yes/no answers, as in LLaVA-Med evaluation.
df = pd.read_parquet(f"{D}/vqa-rad/data/test-00000-of-00001-e5bc3d208bb4deeb.parquet")
os.makedirs(f"{D}/vqa-rad/images", exist_ok=True)
rows = []
for i, r in df.iterrows():
    name = f"{i:04d}.png"
    Image.open(io.BytesIO(r["image"]["bytes"])).save(f"{D}/vqa-rad/images/{name}")
    ans = str(r["answer"]).strip()
    rows.append({"question_id": i, "image": name, "question": r["question"], "answer": ans,
                 "answer_type": "CLOSED" if ans.lower() in ("yes", "no") else "OPEN"})
dump(f"{D}/vqa-rad/test.jsonl", rows)

# SLAKE test, English questions only (1061 QA).
rows = [{"question_id": x["qid"], "image": x["img_name"], "question": x["question"],
         "answer": str(x["answer"]), "answer_type": x["answer_type"]}
        for x in json.load(open(f"{D}/slake/test.json")) if x["q_lang"] == "en"]
dump(f"{D}/slake/test.jsonl", rows)

# IU-Xray test (590 studies, R2GenGPT split). MMedPO feeds the first view (0.png) only.
rows = []
for i, line in enumerate(open(f"{D}/iuxray/test.jsonl")):
    x = json.loads(line)
    rows.append({"question_id": i, "image": x["images"][0].replace("/iu_xray/image/", ""),
                 "gt_report": x["response"]})
dump(f"{D}/iuxray/test_mmedpo.jsonl", rows)
