from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 630

PAPER = "#efe6d3"
INK = "#1b2a4a"
BRASS = "#b8912a"

FONT_PATHS = {
    "bold": [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ],
    "regular": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ],
}


def load_font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS[kind]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


img = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
draw = ImageDraw.Draw(img)

# Border accents, echoing the site's header rule
draw.rectangle([(0, 0), (WIDTH - 1, HEIGHT - 1)], outline=INK, width=6)
draw.line([(80, 260), (WIDTH - 80, 260)], fill=INK, width=4)

title_font = load_font("bold", 140)
plus_font = load_font("bold", 140)
subtitle_font = load_font("regular", 40)
tagline_font = load_font("regular", 28)

title_text = "PDS"
plus_text = "+"

title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
plus_bbox = draw.textbbox((0, 0), plus_text, font=plus_font)
title_width = title_bbox[2] - title_bbox[0]
plus_width = plus_bbox[2] - plus_bbox[0]
total_width = title_width + plus_width
start_x = (WIDTH - total_width) // 2

draw.text((start_x, 90), title_text, font=title_font, fill=INK)
draw.text((start_x + title_width, 90), plus_text, font=plus_font, fill=BRASS)

subtitle_text = "PITCHER DOMINANCE SCORE"
sub_bbox = draw.textbbox((0, 0), subtitle_text, font=subtitle_font)
sub_width = sub_bbox[2] - sub_bbox[0]
draw.text(((WIDTH - sub_width) // 2, 310), subtitle_text, font=subtitle_font, fill=INK)

tagline_text = "Live pitcher leaderboard \u00b7 updated daily"
tag_bbox = draw.textbbox((0, 0), tagline_text, font=tagline_font)
tag_width = tag_bbox[2] - tag_bbox[0]
draw.text(((WIDTH - tag_width) // 2, 480), tagline_text, font=tagline_font, fill="#4a5677")

img.save("preview.png")
print("Saved preview.png")
