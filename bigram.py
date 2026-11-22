"""bigramモデル"""

from pathlib import Path
from tokenizer import CharTokenizer
import torch

vocab_path = Path("data/vocab.json")
train_path = Path("data/train.txt")
val_path = Path("data/val.txt")

tok = CharTokenizer.load(vocab_path)
print(f"既存の語彙を読み込みました: {vocab_path}")

with open(train_path, "r", encoding="utf-8") as f:
    text = f.read()

# list = tok.encode(text)
# print(len(list))

ids = torch.tensor(tok.encode(text))

N = torch.zeros((tok.vocab_size, tok.vocab_size), dtype=torch.int32)
adder = torch.ones(ids.size(0) - 1, dtype=torch.int32)

# for ch1, ch2 in zip(list, list[1:]):
#     N[ch1, ch2] += 1

xs = ids[:-1]
ys = ids[1:]


N.index_put_((xs, ys), adder, accumulate=True)

idx = tok.stoi.get("私")
row = N[idx]
topk = torch.topk(row, 10, sorted=True).indices
print(topk)
topk_str = tok.decode(topk)
print(topk_str)

# 。：そ私こし「三あおす
# 「：そあいどなえお何まこ
# \n：「私三そこ先お二主与
# 私：はのにがをもとだよた

a = 0.001
p = N[0].float()
p = p / p.sum()


g = torch.Generator().manual_seed(42)
ix = tok.stoi["\n"]
out = [ix]
num = 0

P = N.float() + a
P = P / P.sum(1, keepdim=True)


for _ in range(200):
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(ix)
    num += 1

print(tok.decode(out))

# for ix1, ix2 in zip(xs[:10].tolist(), ys[:10].tolist()):
#     prob = P[ix1, ix2]
#     logprob = torch.log(prob)
#     print(f"{ix1},{ix2}: {prob:.4f}")

prob = P[xs, ys]
loss = -torch.log(prob).mean()
print(loss)

# 検証用

with open(val_path, "r", encoding="utf-8") as f:
    val_text = f.read()

val_ids = torch.tensor(tok.encode(val_text))

v_xs = val_ids[:-1]
v_ys = val_ids[1:]

v_prob = P[v_xs, v_ys]
v_loss = -torch.log(v_prob).mean()
print(v_loss)
