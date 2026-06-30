from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "round15_bar_exam_selection_scores_linkedin_final_20260630.png"

W, H = 1600, 1200
S = 2

FONT_REG = "/usr/share/fonts/truetype/nanum/NanumSquareR.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf"
FONT_EXTRA = "/usr/share/fonts/truetype/nanum/NanumSquareEB.ttf"


def f(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size * S)


def c(v: int) -> int:
    return v * S


def rect(values):
    return tuple(c(v) for v in values)


def rounded(draw: ImageDraw.ImageDraw, values, radius: int, fill: str, outline=None, width=1):
    draw.rounded_rectangle(rect(values), radius=c(radius), fill=fill, outline=outline, width=c(width))


def line(draw: ImageDraw.ImageDraw, values, fill: str, width=1):
    draw.line(tuple(c(v) for v in values), fill=fill, width=c(width))


def text(draw: ImageDraw.ImageDraw, x: int, y: int, value: str, font, fill: str, anchor=None):
    draw.text((c(x), c(y)), value, font=font, fill=fill, anchor=anchor)


def text_r(draw: ImageDraw.ImageDraw, x: int, y: int, value: str, font, fill: str):
    text(draw, x, y, value, font, fill, anchor="ra")


def text_lm(draw: ImageDraw.ImageDraw, x: int, y: int, value: str, font, fill: str):
    text(draw, x, y, value, font, fill, anchor="lm")


def text_rm(draw: ImageDraw.ImageDraw, x: int, y: int, value: str, font, fill: str):
    text(draw, x, y, value, font, fill, anchor="rm")


def text_mm(draw: ImageDraw.ImageDraw, x: int, y: int, value: str, font, fill: str):
    text(draw, x, y, value, font, fill, anchor="mm")


def text_c(draw: ImageDraw.ImageDraw, x: int, y: int, value: str, font, fill: str):
    text(draw, x, y, value, font, fill, anchor="ma")


def main() -> None:
    img = Image.new("RGB", (W * S, H * S), "#F6F8FB")
    draw = ImageDraw.Draw(img)

    ink = "#0F172A"
    soft = "#475569"
    muted = "#718096"
    border = "#DDE5EF"
    card = "#FFFFFF"
    track = "#E6ECF3"

    title = f(FONT_EXTRA, 56)
    h1 = f(FONT_EXTRA, 74)
    h2 = f(FONT_EXTRA, 33)
    label = f(FONT_BOLD, 26)
    body = f(FONT_REG, 21)
    small = f(FONT_REG, 16)
    tiny = f(FONT_REG, 14)
    score = f(FONT_EXTRA, 31)
    hero = f(FONT_EXTRA, 80)

    # Header
    text(draw, 74, 58, "제15회 변호사시험 선택형", f(FONT_BOLD, 25), "#0F766E")
    text(draw, 74, 108, "150문항 성적 비교", title, ink)
    text(draw, 76, 178, "정답 수 기준 · 공식 정답 대조 결과", body, soft)

    # Hero result card
    rounded(draw, (620, 58, 1526, 250), 10, card, border)
    rounded(draw, (620, 58, 630, 250), 10, "#0F766E")
    text(draw, 674, 105, "이번 실험", h2, "#0F766E")
    text(draw, 674, 158, "정답 번호 없이 v5 법률 근거 컨텍스트만 사용", body, soft)
    line(draw, (948, 88, 948, 220), "#D8E1EA", 1)
    text_rm(draw, 1462, 145, "150 / 150", hero, "#0F766E")
    text_rm(draw, 1462, 210, "정답률 100%", f(FONT_BOLD, 22), "#0F766E")

    # Subject cards
    subjects = [
        ("공법", "40 / 40"),
        ("민사법", "70 / 70"),
        ("형사법", "40 / 40"),
    ]
    x0, y0, w, h, gap = 74, 286, 468, 116, 34
    for i, (name, val) in enumerate(subjects):
        x = x0 + i * (w + gap)
        rounded(draw, (x, y0, x + w, y0 + h), 10, card, border)
        rounded(draw, (x + 26, y0 + 27, x + 88, y0 + 89), 31, "#EEF7F5")
        text_mm(draw, x + 57, y0 + 61, "§", f(FONT_EXTRA, 36), "#0F766E")
        text_lm(draw, x + 112, y0 + 61, name, label, ink)
        text_rm(draw, x + w - 34, y0 + 61, val, f(FONT_EXTRA, 36), "#0F766E")

    # Comparison card
    rounded(draw, (74, 438, 1526, 1038), 10, card, border)
    text(draw, 126, 498, "비교 점수", h2, ink)
    text(draw, 126, 537, "선택형 150문항 중 정답 수", small, muted)
    text_r(draw, 1390, 521, "정답 수", body, soft)

    # Table guide line
    line(draw, (104, 572, 1496, 572), "#E1E8F0", 1)

    rows = [
        ("1", "이번 실험", 150, "v5 RAG + Gemini 3.1 Pro", "#0F766E"),
        ("1", "슈퍼로이어", 150, "언론 보도 기준", "#2563EB"),
        ("3", "인간 수석", 144, "언론 보도 360/375 환산", "#D97706"),
        ("4", "엘박스 AI", 141, "언론 보도 기준", "#7C3AED"),
        ("5", "Gemini 2.5 Pro", 110, "엘박스 발표 이미지 기준", "#64748B"),
        ("6", "Claude Opus 4.6", 98, "엘박스 발표 이미지 기준", "#64748B"),
        ("7", "GPT-5.4", 94, "엘박스 발표 이미지 기준", "#64748B"),
    ]

    row_top = 604
    row_h = 48
    row_gap = 16
    rank_x = 126
    name_x = 178
    bar_x = 392
    bar_w = 650
    score_x = 1200
    note_box_x = 1250
    note_box_w = 226
    note_x = note_box_x + 14

    for i, (rank, name, value, note, color) in enumerate(rows):
        y = row_top + i * (row_h + row_gap)
        if i % 2 == 0:
            rounded(draw, (104, y - 4, 1496, y + row_h + 4), 8, "#F8FAFC")

        mid = y + row_h // 2
        rounded(draw, (rank_x, mid - 19, rank_x + 38, mid + 19), 7, color)
        text_mm(draw, rank_x + 19, mid + 1, rank, f(FONT_EXTRA, 20), "#FFFFFF")
        text_lm(draw, name_x, mid + 1, name, f(FONT_BOLD, 24), color if i < 4 else ink)

        rounded(draw, (bar_x, mid - 11, bar_x + bar_w, mid + 11), 11, track)
        fill_w = round(bar_w * value / 150)
        rounded(draw, (bar_x, mid - 11, bar_x + fill_w, mid + 11), 11, color)

        text_rm(draw, score_x, mid + 1, f"{value} / 150", score, color if i < 4 else ink)
        rounded(draw, (note_box_x, mid - 18, note_box_x + note_box_w, mid + 18), 6, "#F1F5F9")
        text_lm(draw, note_x, mid + 1, note, small, muted)

        if i < len(rows) - 1:
            line(draw, (104, y + row_h + 8, 1496, y + row_h + 8), "#EEF2F7", 1)

    # Footer
    text(draw, 80, 1084, "출처 구분", f(FONT_BOLD, 17), ink)
    text(draw, 80, 1116, "GPT-5.4 / Claude Opus 4.6 / Gemini 2.5 Pro 점수는 엘박스 발표 이미지 기준.", tiny, muted)
    text(draw, 80, 1142, "엘박스 AI·슈퍼로이어·인간 수석은 언론 보도 기준. 인간 수석 144는 360/375를 문항당 2.5점으로 환산.", tiny, muted)
    text_r(draw, 1520, 1142, "기준일: 2026.06.30", tiny, muted)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img = img.resize((W, H), Image.Resampling.LANCZOS)
    img.save(OUT, quality=96)
    print(OUT)


if __name__ == "__main__":
    main()
