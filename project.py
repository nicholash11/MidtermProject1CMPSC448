import argparse
import copy
import hashlib
import json
import platform
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / ".runtime").is_dir():
    sys.path.insert(0, str(ROOT / ".runtime"))

REVISION = "cd543a149df89434d8a54582c0151c0b945c3d20"
MODELS = {"GPT": "gpt-4o-mini-2024-07-18",
          "Claude": "claude-3-sonnet-20240229",
          "Llama": "Meta-Llama-3-8B-Instruct"}
LABELS = list(MODELS)
DATA = ROOT / "project_data.json"
OUTPUT = ROOT / "project_outputs.json"
SEED = 42
IDENTITY = re.compile(r"\b(?:chatgpt|gpt(?:[- ]?\d[\w.-]*)?|claude|llama|anthropic|openai|meta ai)\b", re.I)
SENSITIVE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}|\b\d{3}-\d{2}-\d{4}\b|\bsk-[A-Za-z0-9]{20,}\b")


def normalized(text):
    return " ".join(text.lower().split())


def download():
    sources, families = {}, {}
    for family, model in MODELS.items():
        url = f"https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/{REVISION}/results/{model}/model_outputs.json"
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        records = json.loads(raw)
        sources[family] = {"model": model, "url": url,
                           "sha256": hashlib.sha256(raw).hexdigest(), "source_rows": len(records)}
        by_prompt = {}
        for record in records:
            prompt, output = record["instruction"], record["output"]
            if prompt.strip() and output.strip():
                key = normalized(prompt)
                if key in by_prompt:
                    raise ValueError(f"Duplicate normalized prompt in {model}")
                by_prompt[key] = record
        families[family] = by_prompt
        print(f"Downloaded {model}: {len(records)} rows", flush=True)

    common = set.intersection(*(set(rows) for rows in families.values()))
    sensitive = {p for p in common if any(SENSITIVE.search(families[f][p][field])
                 for f in LABELS for field in ("instruction", "output"))}
    output_prompts = defaultdict(set)
    for p in common - sensitive:
        for family in LABELS:
            output_prompts[normalized(families[family][p]["output"])].add(p)
    repeated = set().union(*(ps for ps in output_prompts.values() if len(ps) > 1))
    prompts = sorted(common - sensitive - repeated)
    if len(prompts) < 30:
        raise ValueError("Not enough complete, unique prompt groups")
    random.Random(SEED).shuffle(prompts)
    n_train, n_val = int(len(prompts) * .70), int(len(prompts) * .15)
    rows = []
    for i, p in enumerate(prompts):
        split = "train" if i < n_train else "validation" if i < n_train + n_val else "test"
        prompt = families[LABELS[0]][p]["instruction"]
        for family in LABELS:
            record = families[family][p]
            rows.append({"LLM_name": family, "LLM_Input": prompt,
                         "LLM_output": record["output"], "generator": record["generator"],
                         "prompt_id": hashlib.sha256(p.encode()).hexdigest()[:16], "split": split})
    data = {"metadata": {"dataset": "AlpacaEval published model outputs",
            "revision": REVISION, "collected": "2026-10-11", "sources": sources,
            "data_license": "CC BY-NC 4.0 (AlpacaEval dataset card)",
            "common_prompts": len(common), "sensitive_groups_removed": len(sensitive),
            "repeated_output_groups_removed": len(repeated), "retained_prompts": len(prompts),
            "seed": SEED}, "rows": rows}
    DATA.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} balanced examples to {DATA.name}", flush=True)
    return data


def tokens(text):
    text = IDENTITY.sub(" modelname ", text)
    return re.findall(r"\w+|[^\w\s]|\n", text.lower())


def text_tokens(row, mode):
    prompt, output = tokens(row["LLM_Input"]), tokens(row["LLM_output"])
    if mode == "input":
        return prompt[:128]
    if mode == "output":
        return output[:256]
    return prompt[:128] + ["<sep>"] + output[:256]


def validate(data):
    groups, output_splits = defaultdict(list), defaultdict(set)
    for row in data["rows"]:
        assert row["LLM_name"] in LABELS
        assert row["split"] in ("train", "validation", "test")
        assert row["LLM_Input"].strip() and row["LLM_output"].strip()
        groups[row["prompt_id"]].append(row)
        output_splits[normalized(row["LLM_output"])].add(row["split"])
    for rows in groups.values():
        assert Counter(r["LLM_name"] for r in rows) == Counter(LABELS)
        assert len({r["split"] for r in rows}) == 1, "Prompt leakage"
        assert len({r["LLM_Input"] for r in rows}) == 1
    assert all(len(splits) == 1 for splits in output_splits.values()), "Repeated output leakage"
    assert all(any(r["split"] == split for r in data["rows"])
               for split in ("train", "validation", "test"))


def scores(y, predictions):
    import numpy as np
    cm = np.zeros((3, 3), dtype=int)
    for actual, predicted in zip(y, predictions):
        cm[actual, predicted] += 1
    precision = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
    recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {"accuracy": float(np.trace(cm) / cm.sum()), "macro_f1": float(f1.mean()),
            "confusion_matrix": cm.tolist(), "per_class": {
                f: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                for i, f in enumerate(LABELS)}}


def train(data, epochs):
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    validate(data)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    splits = {s: [r for r in data["rows"] if r["split"] == s]
              for s in ("train", "validation", "test")}

    class Classifier(nn.Module):
        def __init__(self, kind, vocab_size):
            super().__init__()
            self.kind = kind
            self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
            self.dropout = nn.Dropout(.3)
            if kind == "CNN":
                self.conv = nn.ModuleList([nn.Conv1d(64, 64, k) for k in (3, 4, 5)])
                self.linear = nn.Linear(192, 3)
            else:
                self.lstm = nn.LSTM(64, 64, batch_first=True)
                self.linear = nn.Linear(64, 3)

        def forward(self, x):
            lengths = x.ne(0).sum(1)
            embedded = self.embedding(x)
            if self.kind == "CNN":
                features = []
                for conv in self.conv:
                    activations = torch.relu(conv(embedded.transpose(1, 2)))
                    valid = torch.arange(activations.size(2))[None, :] < lengths[:, None]
                    features.append(activations.masked_fill(~valid[:, None, :], -1e9).amax(2))
                pooled = torch.cat(features, dim=1)
            else:
                hidden, _ = self.lstm(embedded)
                mask = x.ne(0).unsqueeze(2)
                pooled = (hidden * mask).sum(1) / lengths[:, None]
            return self.linear(self.dropout(pooled))

    results = []
    for mode in ("input", "output", "both"):
        counts = Counter(t for row in splits["train"] for t in text_tokens(row, mode))
        vocab = {"<pad>": 0, "<unk>": 1}
        for token, _ in counts.most_common(9998):
            vocab[token] = len(vocab)
        size = {"input": 128, "output": 256, "both": 385}[mode]
        datasets = {}
        for split, rows in splits.items():
            encoded = []
            for row in rows:
                ids = [vocab.get(t, 1) for t in text_tokens(row, mode)] or [1]
                encoded.append(ids + [0] * (size - len(ids)))
            datasets[split] = TensorDataset(torch.tensor(encoded),
                torch.tensor([LABELS.index(r["LLM_name"]) for r in rows]))

        for kind in ("CNN", "RNN"):
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            model = Classifier(kind, len(vocab))
            optimizer = torch.optim.Adam(model.parameters(), lr=.002)
            loss_fn = nn.CrossEntropyLoss()
            loader = DataLoader(datasets["train"], batch_size=32, shuffle=True,
                                generator=torch.Generator().manual_seed(SEED))

            def evaluate(split):
                model.eval()
                total_loss, predictions, actual = 0., [], []
                with torch.no_grad():
                    for x, y in DataLoader(datasets[split], batch_size=64):
                        logits = model(x)
                        total_loss += loss_fn(logits, y).item() * len(y)
                        predictions.extend(logits.argmax(1).tolist())
                        actual.extend(y.tolist())
                return total_loss / len(actual), scores(actual, predictions), predictions

            start, history, best_loss, best_state, best_epoch = time.perf_counter(), [], float("inf"), None, 0
            for epoch in range(1, epochs + 1):
                model.train()
                total = 0.
                for x, y in loader:
                    optimizer.zero_grad()
                    loss = loss_fn(model(x), y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    total += loss.item() * len(y)
                val_loss, val_scores, _ = evaluate("validation")
                history.append({"epoch": epoch, "train_loss": total / len(datasets["train"]),
                                "validation_loss": val_loss, "validation_accuracy": val_scores["accuracy"]})
                print(f"{kind:3} {mode:6} epoch {epoch:2}: train loss {history[-1]['train_loss']:.3f}, "
                      f"val loss {val_loss:.3f}, val accuracy {val_scores['accuracy']:.3f}", flush=True)
                if val_loss < best_loss - .0001:
                    best_loss, best_state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch
                if epoch - best_epoch >= 3:
                    break
            model.load_state_dict(best_state)
            _, metrics, predictions = evaluate("test")
            correct = np.array(predictions) == datasets["test"].tensors[1].numpy()
            group_correct = correct.reshape(-1, 3).mean(axis=1)
            rng = np.random.default_rng(SEED)
            boot = group_correct[rng.integers(len(group_correct), size=(2000, len(group_correct)))].mean(1)
            metrics["accuracy_ci95"] = np.quantile(boot, [.025, .975]).tolist()
            result = {"model": kind, "input": mode, "best_epoch": best_epoch,
                      "vocab_size": len(vocab), "parameters": sum(p.numel() for p in model.parameters()),
                      "seconds": time.perf_counter() - start, "history": history,
                      "test": metrics, "predictions": predictions}
            results.append(result)
            print(f"TEST {kind} {mode}: accuracy={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}", flush=True)
    data["experiment"] = {"seed": SEED, "maximum_epochs": epochs, "batch_size": 32,
        "learning_rate": .002, "environment": {"python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform()},
        "results": results}
    DATA.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_outputs(data, output=OUTPUT):
    experiment = {**data["experiment"], "results": [
        {key: value for key, value in result.items() if key != "predictions"}
        for result in data["experiment"]["results"]]}
    payload = {"metadata": data["metadata"], "labels": LABELS,
               "experiment": experiment}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                      encoding="utf-8")
    print(f"Saved experiment outputs to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("all", "download", "train"), default="all")
    parser.add_argument("--epochs", type=int, default=12, help="Maximum training epochs (default: 12)")
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="JSON output path (default: project_outputs.json beside this script)")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.command == "download" or not DATA.exists():
        data = download()
    else:
        data = json.loads(DATA.read_text(encoding="utf-8"))
    if args.command in ("all", "train"):
        train(data, args.epochs)
        save_outputs(data, args.output)


if __name__ == "__main__":
    main()
