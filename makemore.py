import hashlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


BASE_DIR = Path(__file__).parent

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

NAMES_FILE = BASE_DIR / "names.txt"

BLOCK_SIZE = 16
EMBED_DIM = 112
NUM_HEADS = 8
NUM_LAYERS = 10
HIDDEN_NEURONS = 108

BATCH_SIZE = 4096
EPOCHS = 30
LR = 0.001

SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
TOP_K = 10


def get_code_hash():
    code = Path(__file__).read_bytes()
    return hashlib.sha256(code).hexdigest()[:10]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_speed():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(EMBED_DIM)

        self.attn = nn.MultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            batch_first=True,
        )

        self.ln2 = nn.LayerNorm(EMBED_DIM)

        self.mlp = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_NEURONS),
            nn.ReLU(),
            nn.Linear(HIDDEN_NEURONS, EMBED_DIM),
        )

    def forward(self, x):
        norm_x = self.ln1(x)

        attn_out, _ = self.attn(
            norm_x,
            norm_x,
            norm_x,
            need_weights=False,
        )

        x = x + attn_out

        norm_x = self.ln2(x)

        mlp_out = self.mlp(norm_x)

        x = x + mlp_out

        return x


class NameModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)

        self.blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(NUM_LAYERS)]
        )

        self.ln_final = nn.LayerNorm(EMBED_DIM)

        self.output = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x):
        B, T = x.shape

        token_emb = self.token_embedding(x)

        positions = torch.arange(T, device=x.device)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_final(x)

        # only use final position to predict next letter
        x = x[:, -1, :]

        logits = self.output(x)

        return logits


def load_words(data_file):
    if not Path(data_file).exists():
        print("could not find names.txt here:")
        print(data_file)
        sys.exit()

    words = open(data_file, "r", encoding="utf-8").read().splitlines()

    # remove empty lines
    words = [w.strip().lower() for w in words if w.strip()]

    return words


def build_vocab(words):
    chars = sorted(list(set("".join(words))))

    stoi = {}
    stoi[SOS_TOKEN] = 0
    stoi[EOS_TOKEN] = 1

    for ch in chars:
        stoi[ch] = len(stoi)

    itos = {i: s for s, i in stoi.items()}

    return stoi, itos


def build_dataset(words, stoi):
    xs, ys = [], []

    for w in words:
        chs = [SOS_TOKEN] + list(w) + [EOS_TOKEN]

        for i in range(1, len(chs)):
            context = chs[max(0, i - BLOCK_SIZE):i]

            # no padding token
            # left-fill with SOS
            context = [SOS_TOKEN] * (BLOCK_SIZE - len(context)) + context

            x = [stoi[ch] for ch in context]
            y = stoi[chs[i]]

            xs.append(x)
            ys.append(y)

    xs = torch.tensor(xs, dtype=torch.long)
    ys = torch.tensor(ys, dtype=torch.long)

    return xs, ys


def estimate_loss(model, x, y, batch_size, device):
    model.eval()

    losses = []

    with torch.no_grad():
        for _ in range(20):
            ix = torch.randint(0, x.shape[0], (batch_size,), device=device)

            xb = x[ix]
            yb = y[ix]

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


def train_model():
    setup_speed()

    device = get_device()
    print("using device:", device)

    words = load_words(NAMES_FILE)

    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)

    xs, ys = build_dataset(words, stoi)

    ix = torch.randperm(xs.shape[0])
    xs = xs[ix]
    ys = ys[ix]

    split = int(0.9 * xs.shape[0])

    xtrain = xs[:split].to(device)
    ytrain = ys[:split].to(device)

    xtest = xs[split:].to(device)
    ytest = ys[split:].to(device)

    model = NameModel(vocab_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    use_amp = device.type == "cuda"

    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    print("vocab size:", vocab_size)
    print("training examples:", xtrain.shape[0])
    print("test examples:", xtest.shape[0])
    print("batch size:", BATCH_SIZE)
    print("epochs:", EPOCHS)

    for epoch in range(EPOCHS):
        model.train()

        perm = torch.randperm(xtrain.shape[0], device=device)

        total_loss = 0.0
        total_batches = 0

        for start in range(0, xtrain.shape[0], BATCH_SIZE):
            batch_ix = perm[start:start + BATCH_SIZE]

            xb = xtrain[batch_ix]
            yb = ytrain[batch_ix]

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)

                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_train_loss = total_loss / total_batches
        test_loss = estimate_loss(model, xtest, ytest, BATCH_SIZE, device)

        print(
            "epoch:",
            epoch + 1,
            "/",
            EPOCHS,
            "| train loss:",
            round(avg_train_loss, 4),
            "| test loss:",
            round(test_loss, 4),
        )

    final_train_loss = estimate_loss(model, xtrain, ytrain, BATCH_SIZE, device)
    final_test_loss = estimate_loss(model, xtest, ytest, BATCH_SIZE, device)

    print("final train loss:", final_train_loss)
    print("final test loss:", final_test_loss)

    code_hash = get_code_hash()
    model_path = MODEL_DIR / f"makemore_{code_hash}.pt"

    torch.save(
        {
            "model_state": model.cpu().state_dict(),
            "stoi": stoi,
            "itos": itos,
            "vocab_size": vocab_size,
            "block_size": BLOCK_SIZE,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "hidden_neurons": HIDDEN_NEURONS,
            "sos_token": SOS_TOKEN,
            "eos_token": EOS_TOKEN,
            "code_hash": code_hash,
            "train_loss": final_train_loss,
            "test_loss": final_test_loss,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        },
        model_path,
    )

    print("saved model to:", model_path)


def apply_top_k(logits, top_k):
    if top_k is None or top_k <= 0:
        return logits

    top_k = min(top_k, logits.shape[-1])

    values, _ = torch.topk(logits, top_k, dim=-1)

    cutoff = values[:, [-1]]

    logits = logits.masked_fill(logits < cutoff, float("-inf"))

    return logits


def generate_one(model, stoi, itos, vocab_size, temperature, top_k, device):
    context = [stoi[SOS_TOKEN]] * BLOCK_SIZE
    out = []

    for _ in range(30):
        x = torch.tensor([context], device=device)

        logits = model(x)

        logits = logits / temperature

        logits = apply_top_k(logits, top_k)

        probs = F.softmax(logits, dim=1)

        ix = torch.multinomial(probs, num_samples=1).item()

        if ix == stoi[EOS_TOKEN]:
            break

        if ix == stoi[SOS_TOKEN]:
            context = context[1:] + [ix]
            continue

        token = itos[ix]

        out.append(token)

        context = context[1:] + [ix]

    return "".join(out)


def generate_names():
    device = get_device()
    print("using device:", device)

    code_hash = get_code_hash()
    model_path = MODEL_DIR / f"makemore_{code_hash}.pt"

    if not model_path.exists():
        print("No model exists for this code version.")
        print("Expected file:", model_path)
        print("Choose 1 first to train the model.")
        sys.exit()

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    vocab_size = checkpoint["vocab_size"]

    model = NameModel(vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    temp_input = input("temperature? press enter for 1.0: ").strip()
    top_k_input = input("top_k? press enter for 10: ").strip()
    count_input = input("how many names? press enter for 20: ").strip()

    temperature = float(temp_input) if temp_input else 1.0
    top_k = int(top_k_input) if top_k_input else TOP_K
    count = int(count_input) if count_input else 20

    if temperature <= 0:
        raise ValueError("temperature must be above 0")

    with torch.no_grad():
        for _ in range(count):
            name = generate_one(
                model=model,
                stoi=stoi,
                itos=itos,
                vocab_size=vocab_size,
                temperature=temperature,
                top_k=top_k,
                device=device,
            )

            print(name)


def main():
    print("1 = train")
    print("2 = generate")

    choice = input("choose 1 or 2: ").strip()

    if choice == "1":
        train_model()

    elif choice == "2":
        generate_names()

    else:
        print("Invalid choice. Choose 1 or 2.")


if __name__ == "__main__":
    main()