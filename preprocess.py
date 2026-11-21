"""青空文庫テキストの前処理。

input/*.txt（Shift_JIS, CRLF）を読み、注記・ルビ・書誌情報を除いた本文を作り、
各作品の末尾 VAL_RATIO を検証用に回して data/train.txt, data/val.txt（UTF-8）に書き出す。

使い方:  python preprocess.py
作品を追加したら input/ に置いて再実行するだけでよい（input/ の原本は書き換えない）。
"""
import re
from collections import Counter
from pathlib import Path

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("data")
VAL_RATIO = 0.1  # 各作品の末尾10%を検証用にする

# --- 置換パターン（適用順に意味がある） ------------------------------------
# 1. 外字: ※［＃「てへん＋劣」、第3水準1-84-77］ → ※ごと消す。
#    注記より先に消さないと ※ だけが本文に残る。
GAIJI = re.compile(r"※［＃[^］]*］")
# 2. 入力者注: ［＃５字下げ］［＃「…」に傍点］［＃「…」は底本では「…」］など
NOTE = re.compile(r"［＃[^］]*］")
# 3. ルビ: 私《わたくし》 → 私、 ｜はルビの開始位置を示す記号
RUBY = re.compile(r"《[^》]*》")
RUBY_START = "｜"
# 4. アクセント分解された欧文: 〔Quid aliud est ...〕 → まるごと消す（日本語だけにする）
#    注記の中に 〔 が入れ子になる箇所があるので、注記を消した後に適用する。
FOREIGN = re.compile(r"〔[^〕]*〕")

# 行ごと消すもの（注記を消す前の行で判定する）
HEADING = re.compile(r"［＃「[^」]*」は[大中小]見出し］")  # 章番号・章題の行
SIGNATURE = re.compile(r"［＃地から[０-９\d]+字上げ］")    # 末尾の（明治三十九年四月）など


def extract_body(raw: str) -> str:
    """冒頭の記号説明ブロック（-----の行2本で挟まれた部分）より後、底本：より前を取り出す。"""
    lines = raw.split("\n")
    seps = [i for i, line in enumerate(lines) if line.startswith("-----")]
    if len(seps) < 2:
        raise ValueError("記号説明ブロック（-----の行2本）が見つからない")
    body = lines[seps[1] + 1:]
    for i, line in enumerate(body):
        if line.startswith("底本："):
            return "\n".join(body[:i])
    raise ValueError("底本：の行が見つからない")


def clean_line(line: str) -> str | None:
    """1行を整形する。行ごと消すときは None を返す。"""
    if HEADING.search(line) or SIGNATURE.search(line):
        return None
    line = GAIJI.sub("", line)
    line = NOTE.sub("", line)
    line = RUBY.sub("", line)
    line = line.replace(RUBY_START, "")
    line = FOREIGN.sub("", line)
    # 全角スペース: 段落頭の字下げと、？！のあとの組版上の空き（小説の体裁なので消す）
    line = line.replace("　", "").strip()
    return line or None  # 空行も消す（1段落 = 1行にそろえる）


def preprocess(raw: str) -> list[str]:
    paragraphs = []
    for line in extract_body(raw).split("\n"):
        cleaned = clean_line(line)
        if cleaned is not None:
            paragraphs.append(cleaned)
    return paragraphs


def split_train_val(paragraphs: list[str]) -> tuple[list[str], list[str]]:
    """文字数で末尾 VAL_RATIO を検証用に。段落の途中では切らない。"""
    total = sum(len(p) + 1 for p in paragraphs)
    count = 0
    for i, p in enumerate(paragraphs):
        if count >= total * (1 - VAL_RATIO):
            return paragraphs[:i], paragraphs[i:]
        count += len(p) + 1
    return paragraphs, []


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    train_all, val_all = [], []
    print(f"{'作品':<28}{'学習':>9}{'検証':>9}  （文字数）")
    for path in sorted(INPUT_DIR.glob("*.txt")):
        raw = open(path, encoding="cp932").read()
        train, val = split_train_val(preprocess(raw))
        train_all += train
        val_all += val
        n_tr = sum(len(p) + 1 for p in train)
        n_va = sum(len(p) + 1 for p in val)
        print(f"{path.name:<28}{n_tr:>9,}{n_va:>9,}")

    train_text = "\n".join(train_all) + "\n"
    val_text = "\n".join(val_all) + "\n"
    (OUTPUT_DIR / "train.txt").write_text(train_text, encoding="utf-8")
    (OUTPUT_DIR / "val.txt").write_text(val_text, encoding="utf-8")
    print(f"{'合計':<28}{len(train_text):>9,}{len(val_text):>9,}")

    # 確認用: 消し残しがないか（記号類が残っていたら警告）
    leftovers = {s: train_text.count(s) + val_text.count(s)
                 for s in ["［", "］", "＃", "《", "》", "｜", "※", "〔", "〕"]}
    leftovers = {s: n for s, n in leftovers.items() if n}
    print("消し残し:", leftovers or "なし")

    vocab = Counter(train_text + val_text)
    print(f"語彙数: {len(vocab)}")
    unseen = set(val_text) - set(train_text)
    print(f"検証にだけ出る文字: {len(unseen)}種 {''.join(sorted(unseen))}")


if __name__ == "__main__":
    main()
