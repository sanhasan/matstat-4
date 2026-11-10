from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def _setup():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "figure.dpi": 110,
        }
    )


def _save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def overview_plot(data, directory="images/anime"):
    _setup()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    missing = data.isna().sum().sort_values(ascending=False)
    axes[0].bar(missing.index, missing.values, color="tab:red", alpha=0.75)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set(title="Пропуски в исходной таблице", ylabel="Количество")

    counts = data["type"].value_counts().sort_values(ascending=True)
    axes[1].barh(counts.index, counts.values, color="tab:blue", alpha=0.8)
    axes[1].set(title="Категориальный признак type", xlabel="Количество аниме")
    return _save(fig, directory, "01_dataset_overview")


def normality_plot(ratings, mean, std, directory="images/anime"):
    _setup()
    ratings = np.asarray(ratings)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(ratings, bins=45, density=True, alpha=0.7, color="tab:blue", label="Данные")
    grid = np.linspace(ratings.min(), ratings.max(), 600)
    axes[0].plot(grid, stats.norm.pdf(grid, mean, std), color="black", lw=2,
                 label=f"N({mean:.3f}, {std**2:.3f})")
    axes[0].set(xlabel="Средний рейтинг", ylabel="Плотность",
                title="Рейтинг и подобранный нормальный закон")
    axes[0].legend()

    theoretical, ordered = stats.probplot(ratings, dist="norm", fit=False)
    axes[1].scatter(theoretical, ordered, s=7, alpha=0.35, color="tab:orange")
    slope, intercept = np.polyfit(theoretical, ordered, 1)
    axes[1].plot(theoretical, slope * np.asarray(theoretical) + intercept, "k--")
    axes[1].set(xlabel="Теоретические квантили N(0, 1)",
                ylabel="Выборочные квантили rating", title="Q–Q график")
    return _save(fig, directory, "02_rating_normality")


def homogeneity_plot(tv, movie, directory="images/anime"):
    _setup()
    tv, movie = np.sort(tv), np.sort(movie)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].step(tv, np.arange(1, tv.size + 1) / tv.size, where="post", label="TV")
    axes[0].step(movie, np.arange(1, movie.size + 1) / movie.size, where="post", label="Movie")
    axes[0].set(xlabel="Рейтинг", ylabel="Эмпирическая функция распределения",
                title="Различие распределений рейтинга")
    axes[0].legend()

    axes[1].boxplot([tv, movie], tick_labels=["TV", "Movie"], showfliers=False,
                    patch_artist=True,
                    boxprops={"facecolor": "tab:blue", "alpha": 0.45})
    axes[1].set(ylabel="Рейтинг", title="Центр и межквартильный размах")
    return _save(fig, directory, "03_tv_movie_homogeneity")


def association_plot(ratings, members, directory="images/anime"):
    _setup()
    ratings = np.asarray(ratings)
    log_members = np.log10(np.asarray(members) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    image = axes[0].hexbin(log_members, ratings, gridsize=55, mincnt=1, cmap="viridis")
    slope, intercept = np.polyfit(log_members, ratings, 1)
    grid = np.linspace(log_members.min(), log_members.max(), 300)
    axes[0].plot(grid, intercept + slope * grid, color="tab:red", lw=2,
                 label="Линейный тренд")
    axes[0].set(xlabel="log10(members + 1)", ylabel="Рейтинг",
                title="Популярность и средний рейтинг")
    axes[0].legend()
    fig.colorbar(image, ax=axes[0], label="Число наблюдений в ячейке")

    edges = np.quantile(log_members, np.linspace(0, 1, 11))
    groups = np.clip(np.digitize(log_members, edges[1:-1]), 0, 9)
    centers = np.array([np.median(log_members[groups == i]) for i in range(10)])
    median = np.array([np.median(ratings[groups == i]) for i in range(10)])
    low = np.array([np.quantile(ratings[groups == i], 0.25) for i in range(10)])
    high = np.array([np.quantile(ratings[groups == i], 0.75) for i in range(10)])
    axes[1].plot(centers, median, "o-", color="tab:orange", label="Медиана")
    axes[1].fill_between(centers, low, high, color="tab:orange", alpha=0.2,
                         label="Межквартильный диапазон")
    axes[1].set(xlabel="log10(members + 1), децили", ylabel="Рейтинг",
                title="Монотонная тенденция по децилям")
    axes[1].legend()
    return _save(fig, directory, "04_rating_members_association")

