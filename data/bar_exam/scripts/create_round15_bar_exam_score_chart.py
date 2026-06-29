from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "round15_bar_exam_selection_scores_20260630.png"


def set_korean_font() -> None:
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    set_korean_font()

    rows = [
        ("GPT-5.4", 94, "엘박스 발표 이미지 기준"),
        ("Claude Opus 4.6", 98, "엘박스 발표 이미지 기준"),
        ("Gemini 2.5 Pro", 110, "엘박스 발표 이미지 기준"),
        ("엘박스 AI", 141, "언론 보도 기준"),
        ("인간 수석", 144, "360/375 환산, 언론 보도 기준"),
        ("슈퍼로이어", 150, "언론 보도 기준"),
        ("이번 실험", 150, "v5 RAG + Antigravity Gemini 3.1 Pro"),
    ]

    labels = [r[0] for r in rows]
    scores = [r[1] for r in rows]
    notes = [r[2] for r in rows]

    colors = []
    for label in labels:
        if label == "이번 실험":
            colors.append("#0F766E")
        elif label == "슈퍼로이어":
            colors.append("#2563EB")
        elif label == "인간 수석":
            colors.append("#D97706")
        elif label == "엘박스 AI":
            colors.append("#7C3AED")
        else:
            colors.append("#94A3B8")

    fig, ax = plt.subplots(figsize=(16, 10), dpi=160)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    bars = ax.barh(labels, scores, color=colors, height=0.62)
    ax.invert_yaxis()

    ax.set_xlim(0, 152)
    ax.set_xlabel("정답 수 / 150문항", fontsize=18, color="#334155", labelpad=16)
    fig.text(
        0.06,
        0.94,
        "제15회 변호사시험 선택형: 150문항 기준 성적 비교",
        fontsize=30,
        fontweight="bold",
        color="#0F172A",
    )
    fig.text(
        0.06,
        0.895,
        "이번 실험은 정답 번호 없이, Codex GPT-5.5가 만든 문항별 법률 근거 컨텍스트(v5)를 Antigravity Gemini 3.1 Pro가 풀이",
        fontsize=14.5,
        color="#475569",
    )

    for bar, score, note, label in zip(bars, scores, notes, labels):
        y = bar.get_y() + bar.get_height() / 2
        label_color = "white" if label in {"엘박스 AI", "인간 수석", "슈퍼로이어", "이번 실험"} else "#0F172A"
        x = score - 2.5 if score >= 120 else score + 1.4
        ha = "right" if score >= 120 else "left"
        ax.text(
            x,
            y,
            f"{score}/150",
            va="center",
            ha=ha,
            fontsize=19,
            fontweight="bold",
            color=label_color,
        )

    ax.axvline(150, color="#0F172A", linewidth=1.4, linestyle="--", alpha=0.55)
    ax.text(149.4, len(labels) - 0.12, "만점 150", ha="right", va="bottom", fontsize=13, color="#0F172A")

    ax.grid(axis="x", color="#CBD5E1", linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", labelsize=17, colors="#0F172A", length=0)
    ax.tick_params(axis="x", labelsize=13, colors="#475569")

    source_lines = [
        "출처 구분: GPT-5.4 / Claude Opus 4.6 / Gemini 2.5 Pro 점수는 엘박스 발표 이미지 기준.",
        "엘박스 AI·슈퍼로이어·인간 수석은 언론 보도 기준. 인간 수석 144는 360/375를 문항당 2.5점으로 환산한 값.",
    ]
    fig.text(0.06, 0.053, source_lines[0], fontsize=11.3, color="#64748B")
    fig.text(0.06, 0.029, source_lines[1], fontsize=11.3, color="#64748B")

    plt.subplots_adjust(left=0.17, right=0.94, top=0.81, bottom=0.14)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
