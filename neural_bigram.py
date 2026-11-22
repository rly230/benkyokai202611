"""ニューラルbigram：数え上げの代わりに、勾配降下で 2400×2400 の表を学習する"""

import torch
import torch.nn.functional as F
from tokenizer import CharTokenizer

tok = CharTokenizer.load("data/vocab.json")
V = tok.vocab_size
device = "cuda"


def load_ids(path):
    with open(path, encoding="utf-8") as f:
        return torch.tensor(tok.encode(f.read()), device=device)


train_ids, val_ids = load_ids("data/train.txt"), load_ids("data/val.txt")
t_xs, t_ys = train_ids[:-1], train_ids[1:]
v_xs, v_ys = val_ids[:-1], val_ids[1:]

g = torch.Generator(device=device).manual_seed(42)
W = torch.randn((V, V), generator=g, device=device, requires_grad=True)

batch_size = 16384
lr = 200  # 動画より大きい。理由は下の「考えてほしいこと」
steps = 10000

for step in range(steps + 1):
    ix = torch.randint(0, len(t_xs), (batch_size,), generator=g, device=device)
    x, y = t_xs[ix], t_ys[ix]

    logits = W[x]  # TODO 1: W から x の行を取り出す（one-hot @ W と同じ）
    # print(logits)
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(len(y)), y].log().mean()
    # print(loss)

    W.grad = None
    loss.backward()
    W.data += -lr * W.grad  # TODO 3: W を勾配と逆の方向に lr だけ動かす

    if step % 1000 == 0:
        with torch.no_grad():  # 評価では勾配を計算しない
            tr = F.cross_entropy(W[t_xs], t_ys).item()
            va = F.cross_entropy(W[v_xs], v_ys).item()
        print(f"step {step:5d}  train {tr:.3f}  val {va:.3f}")

# ---- 数え上げ版との比較（学習ループのあとに追加） ----
with torch.no_grad():
    # 段階1と同じ数え上げの表（+1 スムージングも段階1に合わせる）
    N = torch.zeros((V, V), device=device)
    N.index_put_((t_xs, t_ys), torch.ones(len(t_xs), device=device), accumulate=True)
    P_count = (N + 1) / (N + 1).sum(1, keepdim=True)

    # ニューラル版：W の各行を softmax で確率にする
    P_neural = torch.softmax(W, dim=1)


def disp(ch):
    """改行や全角スペースなど、見えない文字を見える形にする"""
    return repr(ch)[1:-1]


def show_top(ch, k=10):
    i = tok.encode(ch)[0]
    if i == tok.unk_id:
        print(f"「{ch}」は語彙にありません")
        return
    pc, ic = P_count[i].topk(k)
    pn, in_ = P_neural[i].topk(k)
    pc, ic, pn, in_ = pc.tolist(), ic.tolist(), pn.tolist(), in_.tolist()

    print(f"「{ch}」の次に来る文字 上位{k}（左：数え上げ　右：ニューラル）")
    for r in range(k):
        print(
            f"{r + 1:>2}位  {disp(tok.itos[ic[r]]):>4} {pc[r]:.3f}"
            f"    {disp(tok.itos[in_[r]]):>4} {pn[r]:.3f}"
        )


show_top("私")
