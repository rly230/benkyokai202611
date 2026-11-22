import torch
import torch.nn.functional as F
from tokenizer import CharTokenizer

tok = CharTokenizer.load("data/vocab.json")
device = "cuda"


def load_ids(path):
    with open(path, encoding="utf-8") as f:
        return torch.tensor(tok.encode(f.read()), device=device)


def build_dataset(ids, block_size):
    n = len(ids) - block_size  # 作れる例の数
    starts = torch.arange(n, device=ids.device).unsqueeze(1)
    offsets = torch.arange(block_size, device=ids.device)
    idx = starts + offsets
    X = ids[idx]  # 表の番号で一度に取り出す
    Y = ids[block_size:]  # 各行の「次の1文字」
    # ids.unfold(0, block_size, 1)[:-1]
    return X, Y


train_ids, val_ids = load_ids("data/train.txt"), load_ids("data/val.txt")
block_size = 8

X, Y = build_dataset(train_ids, block_size)
Xv, Yv = build_dataset(val_ids, block_size)

for i in range(5):
    print(tok.decode(X[i].tolist()), "→", tok.decode([Y[i].item()]))


d = 24  # 埋め込みの次元（1文字を何個の数で表すか）
H = 512  # 隠れ層の大きさ
V = tok.vocab_size

g = torch.Generator(device=device).manual_seed(42)
C = torch.randn((V, d), generator=g, device=device)
W1 = (
    torch.randn((block_size * d, H), generator=g, device=device)
    * (5 / 3)
    / (block_size * d) ** 0.5
)
b1 = torch.zeros(H, device=device)
W2 = torch.randn((H, V), generator=g, device=device) * 0.01
b2 = torch.zeros(V, device=device)
params = [C, W1, b1, W2, b2]
for p in params:
    p.requires_grad_()
print("パラメータ数:", sum(p.numel() for p in params))


def forward(X):
    emb = C[X]  # 形 (バッチ, block_size, d)
    h = torch.tanh(emb.view(-1, block_size * d) @ W1 + b1)
    return h @ W2 + b2  # h から logits を出す。形 (バッチ, V)


# ---- 学習ループの前に1回だけ ----
eval_g = torch.Generator().manual_seed(0)  # 学習用の g とは別にする
eval_ix = torch.randperm(len(X), generator=eval_g)[:200_000].to(device)
X_eval, Y_eval = X[eval_ix], Y[eval_ix]

batch_size, lr, steps = 1024, 0.5, 60000
for step in range(steps + 1):
    ix = torch.randint(0, len(X), (batch_size,), generator=g, device=device)
    loss = F.cross_entropy(forward(X[ix]), Y[ix])
    for p in params:
        p.grad = None
    loss.backward()
    cur_lr = lr if step < 45000 else lr / 10  # 終盤は学習率を下げて仕上げる
    with torch.no_grad():
        for p in params:
            p -= cur_lr * p.grad
    if step % 5000 == 0:
        with torch.no_grad():
            tr = F.cross_entropy(
                forward(X_eval), Y_eval
            ).item()  # 全体からランダムに選んだ20万組
            va = F.cross_entropy(forward(Xv), Yv).item()
        print(f"step {step:5d}  train {tr:.3f}  val {va:.3f}")


from collections import Counter

# ---- similar() より前に置く（132行目の train_text の読み込みはここへ移して、元の行は消す） ----
with open("data/train.txt", encoding="utf-8") as f:
    train_text = f.read()

counts = Counter(train_text)  # 1. 文字ごとの出現回数
freq = torch.tensor(
    [counts.get(ch, 0) for ch in tok.itos], device=device  # 語彙の順に並べた出現回数
)


def similar(ch, k=10, min_count=300):
    i = tok.encode(ch)[0]
    if i == tok.unk_id:
        print(f"{ch} は語彙にありません")
        return
    with torch.no_grad():
        E = F.normalize(C, dim=1)
        sims = E @ E[i]
        sims = sims.masked_fill(
            freq < min_count, float("-inf")
        )  # 2・3. まれな文字を候補から外す
        sims[i] = float("-inf")  # 自分自身は直接外す
        vals, idx = sims.topk(k)
    pairs = [
        (tok.itos[j], round(v, 2))
        for v, j in zip(vals.tolist(), idx.tolist())
        if v != float("-inf")
    ]  # 候補が k 個に満たない場合の空き枠を除く
    print(f"{ch}（{counts.get(ch, 0)}回）→", pairs)


@torch.no_grad()
def generate(start="「", n=200, temperature=1.0, seed=42, ban_unk=True, pad="\n"):
    gen = torch.Generator(device=device).manual_seed(seed)
    ctx = tok.encode(start)
    n_pad = max(0, block_size - len(ctx))
    out = tok.encode(pad) * n_pad + ctx  # 足りない分を前から改行で埋める

    for _ in range(n):
        x = torch.tensor([out[-block_size:]], device=device)  # 最後の block_size 文字
        logits = forward(x)[0] / temperature
        if ban_unk:
            logits[tok.unk_id] = float("-inf")  # 〓 を選ばせない
        probs = F.softmax(logits, dim=0)
        ix = torch.multinomial(probs, 1, generator=gen).item()
        out.append(ix)

    return tok.decode(out[n_pad:])  # 埋めた改行は表示しない


def longest_copy(s, text):
    """生成文 s の中で、学習データにそのまま含まれる最長の部分を探す"""
    best = ""
    for i in range(len(s)):
        j = i + len(best) + 1
        while j <= len(s) and s[i:j] in text:
            best = s[i:j]
            j += 1
    return best


for mc in [1000, 300, 100, 30, 10]:
    print(f"--- min_count={mc}")
    for ch in ["云", "行", "赤"]:
        similar(ch, min_count=mc)


for seed in range(3):
    s = generate("「", seed=seed)
    print(s)
    print(
        f"▶ 学習データと一致した最長部分（{len(longest_copy(s, train_text))}文字）:",
        longest_copy(s, train_text),
    )
    print("-" * 40)
