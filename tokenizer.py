"""文字単位のトークナイザー

使い方（ほかのファイルから）:
    from tokenizer import CharTokenizer
    tok = CharTokenizer.load("data/vocab.json")
    ids = tok.encode("吾輩は猫である")
    text = tok.decode(ids)
"""

from __future__ import (
    annotations,
)

import json
from collections import Counter
from pathlib import Path

UNK = "〓"  # 未知の文字を表す1文字（本文に出てこないこと）
MIN_COUNT = 4  # この回数未満（＝3回以下）の文字は語彙に入れない


class CharTokenizer:
    def __init__(self, itos: list[str], meta: dict | None = None):
        self.itos = list(itos)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        # 同じ文字が2回入っていると ID がずれるので確認しておく
        if len(self.stoi) != len(self.itos):
            raise ValueError("語彙に重複した文字があります")
        if UNK not in self.stoi:
            raise ValueError("語彙にUNKが含まれていません")
        self.unk_id = self.stoi[UNK]
        self.meta = meta or {}

    @classmethod
    def build(cls, text: str, min_count: int = MIN_COUNT) -> CharTokenizer:
        """学習用テキストから語彙を作る。最初の1回だけ使う。"""
        assert (
            UNK not in text
        ), f"本文に {UNK} が含まれています。別の文字を UNK にしてください"
        counts = Counter(text)
        kept = sorted(ch for ch, n in counts.items() if n >= min_count)
        itos = [UNK] + kept  # UNK を ID 0 に固定する
        meta = {
            "unk": UNK,
            "min_count": min_count,
            "chars_in_text": len(counts),  # 元の文字の種類数（記録用）
        }
        return cls(itos, meta)

    def save(self, path: str | Path) -> None:
        """語彙を JSON に保存する。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {**self.meta, "itos": self.itos}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        """保存した JSON から語彙を復元する。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("unk") == UNK, "保存時と UNK の文字が違います"
        itos = data.pop("itos")
        return cls(itos, meta=data)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        """文字列 → ID のリスト。語彙にない文字は unk_id にする。"""
        return [self.stoi.get(ch, self.unk_id) for ch in s]

    def decode(self, ids) -> str:
        """ID のリスト → 文字列。UNK は 〓 のまま出す。
        PyTorch のテンソルを渡すときは ids.tolist() にしてから渡す。"""
        return "".join(self.itos[i] for i in ids)


if __name__ == "__main__":
    vocab_path = Path("data/vocab.json")
    train_path = Path("data/train.txt")

    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()

    if vocab_path.exists():
        tok = CharTokenizer.load(vocab_path)
        print(f"既存の語彙を読み込みました: {vocab_path}")
    else:
        tok = CharTokenizer.build(text)
        tok.save(vocab_path)
        print(f"語彙を作って保存しました: {vocab_path}")

    # 1. 語彙の大きさ
    print(f"語彙数: {tok.vocab_size}（UNK を含む） 設定: {tok.meta}")

    # 2. 往復の確認：まれな文字は 〓 になるので、期待値もそれに合わせて作る
    sample = text[:200]
    expected = "".join(ch if ch in tok.stoi else UNK for ch in sample)
    assert tok.decode(tok.encode(sample)) == expected
    print("往復OK:", tok.decode(tok.encode(sample[:50])))

    # 3. 未知の文字
    assert tok.encode("𠮷") == [tok.unk_id]
    print("未知の文字:", tok.encode("𠮷"), "→", tok.decode(tok.encode("𠮷")))

    # 4. 〓 になった文字が本文の何%か
    n_unk = sum(1 for ch in text if ch not in tok.stoi)
    print(f"〓 になる文字: {n_unk} / {len(text)}（本文の {n_unk / len(text):.3%}）")
