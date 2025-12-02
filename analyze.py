from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --- читаем данные ---

def read_csv_simple(path: Path) -> pd.DataFrame:
    print(f"читаю {path.name}...")
    return pd.read_csv(path, sep=",")


def load_all() -> Dict[str, pd.DataFrame]:
    data_dir = Path(__file__).resolve().parent / "data" / "raw"
    if not data_dir.exists():
        raise FileNotFoundError("папка data/raw не найдена")

    names = ["train", "test", "users", "books", "genres", "book_genres", "book_descriptions"]
    dfs = {}
    for name in names:
        file = data_dir / f"{name}.csv"
        if file.exists():
            dfs[name] = read_csv_simple(file)
        else:
            print(f"{name}.csv не найден, пропуск")
    return dfs


# --- объединяем и готовим данные ---

def build_train_merged(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    train, users, books = dfs["train"].copy(), dfs["users"].copy(), dfs["books"].copy()
    for d in [train, users, books]:
        d.columns = [c.strip() for c in d.columns]

    if "has_read" in train:
        before = len(train)
        train = train[train["has_read"] == 1]
        print(f"фильтр has_read=1: {before} → {len(train)}")

    df = train.merge(users, on="user_id", how="left").merge(books, on="book_id", how="left")

    if "gender" in df:
        df["gender"] = df["gender"].map({1: "м", 2: "ж"})

    if "book_genres" in dfs:
        cnt = dfs["book_genres"].groupby("book_id")["genre_id"].count().rename("book_genres_count")
        df = df.merge(cnt, on="book_id", how="left")
    else:
        df["book_genres_count"] = np.nan

    print("итоговая таблица:", df.shape)
    return df


def make_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 12, 17, 25, 40, 60, 200]
    labels = ["дети", "подростки", "молодёжь", "взрослые", "зрелые", "пожилые"]
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
    return df


def minmax_scale(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df:
            continue
        cmin, cmax = df[c].min(), df[c].max()
        df[c] = 0 if pd.isna(cmin) or pd.isna(cmax) or cmin == cmax else (df[c] - cmin) / (cmax - cmin)
    print("нормализовано:", ", ".join(cols))
    return df


# --- рисуем ---

def save_plot(path: Path, imgs: List[str]):
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    imgs.append(path.name)


def save_hist(df, col, title, out, imgs):
    if col not in df: return
    plt.figure(figsize=(8, 5))
    df[col].dropna().hist(bins=40)
    plt.title(title)
    save_plot(out / f"hist_{col}.png", imgs)


def save_bar(df, col, title, out, imgs, topn=None):
    if col not in df: return
    vc = df[col].value_counts()
    if topn:
        top = vc.head(topn)
        other = vc.iloc[topn:].sum()
        if other > 0:
            top["другие"] = other
        vc = top
    plt.figure(figsize=(9, 4))
    vc.plot(kind="bar")
    plt.title(title)
    save_plot(out / f"bar_{col}.png", imgs)


def save_mean(df, col, title, out, imgs):
    if col not in df or "rating" not in df: return
    g = df.groupby(col)["rating"].mean().sort_values(ascending=False)
    plt.figure(figsize=(9, 4))
    g.plot(kind="bar")
    plt.title(title)
    save_plot(out / f"mean_{col}.png", imgs)


def save_scatter(df, x, title, xlabel, ylabel, out, imgs, sample=30000):
    if x not in df: return
    d = df[[x, "rating"]].dropna()
    if len(d) > sample:
        d = d.sample(sample, random_state=42)
    plt.figure(figsize=(8, 5))
    plt.scatter(d[x], d["rating"], s=8, alpha=0.2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_plot(out / f"scatter_{x}.png", imgs)


def save_corr(df, cols, out, imgs):
    cols = [c for c in cols if c in df]
    if len(cols) < 2: return
    corr = df[cols].corr()
    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr)
    plt.colorbar(im)
    plt.xticks(range(len(cols)), cols, rotation=45)
    plt.yticks(range(len(cols)), cols)
    plt.title("корреляция")
    save_plot(out / "corr.png", imgs)


# --- запуск ---

def main():
    print("начинаем рисовать 🎨")
    dfs = load_all()
    df = build_train_merged(dfs)
    df = make_age_groups(df)

    num_cols = ["publication_year", "avg_rating", "book_genres_count"]
    df = minmax_scale(df, num_cols)

    if "age" in df:
        amin, amax = df["age"].min(), df["age"].max()
        df["age_norm"] = (df["age"] - amin) / (amax - amin)

    out = Path(__file__).resolve().parent / "output" / "eda"
    out.mkdir(parents=True, exist_ok=True)
    imgs = []

    for c in ["rating", "age"] + num_cols:
        save_hist(df, c, f"{c}: распределение", out, imgs)

    for cat in ["age_group", "gender", "language", "publisher"]:
        save_bar(df, cat, f"{cat}: частоты", out, imgs, topn=10)
        save_mean(df, cat, f"{cat}: средний рейтинг", out, imgs)

    pairs = [
        ("age", "возраст vs рейтинг", "возраст", "рейтинг"),
        ("publication_year", "год публикации vs рейтинг", "год", "рейтинг"),
        ("avg_rating", "средний рейтинг книги vs рейтинг", "средний рейтинг", "рейтинг"),
        ("book_genres_count", "жанры vs рейтинг", "число жанров", "рейтинг"),
    ]
    for x, t, xl, yl in pairs:
        save_scatter(df, x, t, xl, yl, out, imgs)

    save_corr(df, ["rating", "age", "age_norm", "publication_year", "avg_rating", "book_genres_count"], out, imgs)

    print(f"\nEda отработана! картинки лежат в {out}")


if __name__ == "__main__":
    main()
