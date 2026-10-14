"""Render a karaoke video from word-timed lyrics and separated stems.

The lyrics JSON has {"lead": [line, ...], "backing": [line, ...]}, where each line is
{"start", "end", "words": [{"w", "s", "e"}]} (the format karaoke pages are built from).
Lyrics are written as ASS subtitles with \\kf fill tags and burned in by ffmpeg/libass.

Usage:
    python stem_test/karaoke_video.py lyrics.json --audio inst.wav backing.wav \\
        --title Searching --artist "Z by Z" --fonts-dir fonts/ --out karaoke.mp4
"""

import argparse
import subprocess
import tempfile
from pathlib import Path
import json

from PIL import Image, ImageFont

W, H = 1920, 1080
LEAD_ROWS = (560, 725)       # baseline y of the two alternating lead rows
BACKING_Y = 905
LEAD_SIZE, BACKING_SIZE = 112, 80
MAX_TEXT_W = W - 2 * 100     # lines wider than this are narrowed to fit
BG = "#101318"
PRE_ROLL = 4.0               # how early a lead line can appear before it is sung
COUNTDOWN_GAP = 6.0          # show a countdown before a line that follows a gap this long


def ass_color(hex_rgb, alpha=0):
    r, g, b = hex_rgb[1:3], hex_rgb[3:5], hex_rgb[5:7]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def ts(t):
    t = max(0.0, t)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def karaoke_text(words, appear):
    """\\kf fills each word over its sung duration; plain \\k holds through gaps."""
    parts = [f"{{\\k{max(0, round((words[0]['s'] - appear) * 100))}}}"]
    for i, w in enumerate(words):
        parts.append(f"{{\\kf{max(1, round((w['e'] - w['s']) * 100))}}}{w['w']}")
        if i + 1 < len(words):
            gap = max(0, round((words[i + 1]["s"] - w["e"]) * 100))
            parts.append(f"{{\\k{gap}}} ")
    return "".join(parts)


def shift(lyrics, offset):
    """Move every line and word later by offset seconds."""
    out = {}
    for part, lines in lyrics.items():
        out[part] = [{**l, "start": l["start"] + offset, "end": l["end"] + offset,
                      "words": [{**w, "s": w["s"] + offset, "e": w["e"] + offset} for w in l["words"]]}
                     for l in lines]
    return out


def fit_tag(text, font, size):
    """\\fscx tag that narrows a line just enough to fit MAX_TEXT_W, or "" if it already fits."""
    if font is None:
        return ""
    width = ImageFont.truetype(font, size).getlength(text)
    return f"\\fscx{int(100 * MAX_TEXT_W / width)}" if width > MAX_TEXT_W else ""


def find_font(fonts_dir, family, style):
    for f in sorted(Path(fonts_dir).glob("*.[ot]tf")):
        name = ImageFont.truetype(str(f), 10).getname()
        if name[0] == family and style in name[1]:
            return str(f)
    return None


def build_ass(lyrics, title, artist, duration, intro=0.0, credits=(), fonts_dir="."):
    lead_font = find_font(fonts_dir, "Bricolage Grotesque", "ExtraBold")
    back_font = find_font(fonts_dir, "Bricolage Grotesque", "Medium")
    lead, back = lyrics["lead"], lyrics["backing"]
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lead,Bricolage Grotesque ExtraBold,{LEAD_SIZE},{ass_color('#ec6fa6')},{ass_color('#5b6474')},{ass_color('#0b0d11')},{ass_color('#000000', 0x80)},0,0,0,0,100,100,0,0,1,3.5,0,2,100,100,0,1
Style: Backing,Bricolage Grotesque Medium,{BACKING_SIZE},{ass_color('#4fbdd1')},{ass_color('#3a4a55')},{ass_color('#0b0d11')},{ass_color('#000000', 0x80)},0,1,0,0,100,100,0,0,1,3,0,2,100,100,0,1
Style: Label,IBM Plex Mono Medium,30,{ass_color('#9aa3b0')},{ass_color('#9aa3b0')},{ass_color('#000000', 0xFF)},{ass_color('#000000', 0xFF)},0,0,0,0,100,100,2,0,1,0,0,7,0,0,0,1
Style: Count,Bricolage Grotesque ExtraBold,80,{ass_color('#ec6fa6')},{ass_color('#2a303a')},{ass_color('#000000', 0xFF)},{ass_color('#000000', 0xFF)},0,0,0,0,100,100,24,0,1,0,0,2,0,0,0,1
Style: Title,Bricolage Grotesque ExtraBold,120,{ass_color('#e8ebf0')},{ass_color('#e8ebf0')},{ass_color('#000000', 0xFF)},{ass_color('#000000', 0xFF)},0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1
Style: Kicker,IBM Plex Mono Medium,30,{ass_color('#ec6fa6')},{ass_color('#ec6fa6')},{ass_color('#000000', 0xFF)},{ass_color('#000000', 0xFF)},0,0,0,0,100,100,6,0,1,0,0,5,0,0,0,1
Style: Credit,IBM Plex Mono Medium,26,{ass_color('#7c8594')},{ass_color('#7c8594')},{ass_color('#000000', 0xFF)},{ass_color('#000000', 0xFF)},0,0,0,0,100,100,1,0,1,0,0,5,0,0,0,1
Style: Artist,Bricolage Grotesque Medium,56,{ass_color('#9aa3b0')},{ass_color('#9aa3b0')},{ass_color('#000000', 0xFF)},{ass_color('#000000', 0xFF)},0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    add = lambda start, end, style, text: ev.append(f"Dialogue: 0,{ts(start)},{ts(end)},{style},,0,0,0,,{text}")

    # header labels once the song is under way
    add(intro, duration, "Label", f"{{\\pos(110,70)}}{title.upper()}  ·  {artist.upper()}")
    add(intro, duration, "Label", f"{{\\an9\\pos({W - 110},70)}}{{\\c{ass_color('#ec6fa6')}}}●{{\\c{ass_color('#9aa3b0')}}} LEAD   "
                              f"{{\\c{ass_color('#4fbdd1')}}}●{{\\c{ass_color('#9aa3b0')}}} BACKING")

    # title card, cleared before the first line appears so the two never overlap
    first = min([l["start"] for l in lead + back] or [duration])
    card_end = first - PRE_ROLL - 0.2
    if card_end > 2:
        cx, cy = W // 2, H // 2 - 60
        add(0.3, card_end, "Kicker", f"{{\\pos({cx},{cy - 150})\\fad(500,400)}}KARAOKE")
        add(0.6, card_end, "Title", f"{{\\pos({cx},{cy - 40})\\fad(700,400)\\fscx104\\fscy104\\t(0,1200,\\fscx100\\fscy100)}}{title}")
        add(1.1, card_end, "Artist", f"{{\\pos({cx},{cy + 70})\\fad(600,400)}}{artist}")
        legend = (f"{{\\c{ass_color('#ec6fa6')}}}●{{\\c{ass_color('#9aa3b0')}}}  LEAD  your part      "
                  f"{{\\c{ass_color('#4fbdd1')}}}●{{\\c{ass_color('#9aa3b0')}}}  BACKING  sung on the track")
        add(2.0, card_end, "Label", f"{{\\an5\\pos({cx},{cy + 190})\\fad(500,400)}}{legend}")
        for i, line in enumerate(credits):
            add(2.6 + 0.25 * i, card_end, "Credit", f"{{\\pos({cx},{H - 200 + 40 * i})\\fad(500,400)}}{line}")

    # lead: alternate rows; a line appears when the previous one starts, so the next line is always visible
    for i, l in enumerate(lead):
        prev_start = lead[i - 1]["start"] if i else 0.0
        appear = max(l["start"] - PRE_ROLL, prev_start)
        nxt = lead[i + 1]["start"] if i + 1 < len(lead) else duration
        vanish = min(l["end"] + 0.5, max(nxt, l["end"] + 0.1))
        y = LEAD_ROWS[i % 2]
        fit = fit_tag(" ".join(w["w"] for w in l["words"]), lead_font, LEAD_SIZE)
        add(appear, vanish, "Lead", f"{{\\pos({W // 2},{y})\\fad(150,200){fit}}}" + karaoke_text(l["words"], appear))

        gap = l["start"] - (lead[i - 1]["end"] if i else 0.0)
        if gap >= COUNTDOWN_GAP and l["start"] >= 3:
            # three dots that fill one per second before the entry
            c0 = l["start"] - 3
            add(c0, l["start"], "Count", f"{{\\pos({W // 2},{y - 140})\\fad(200,0)}}{{\\kf100}}●{{\\kf100}}●{{\\kf100}}●")

    # instrumental breaks get a quiet marker
    for a, b in zip(lead, lead[1:]):
        if b["start"] - a["end"] > 15:
            add(a["end"] + 1.5, b["start"] - 4, "Artist", f"{{\\pos({W // 2},{LEAD_ROWS[0] + 40})\\fad(600,600)}}♪  instrumental  ♪")

    for i, l in enumerate(back):
        prev_end = back[i - 1]["end"] + 0.3 if i else 0.0
        appear = max(l["start"] - 1.5, prev_end)
        nxt = back[i + 1]["start"] - 1.5 if i + 1 < len(back) else duration
        vanish = max(min(l["end"] + 0.5, nxt), l["end"] + 0.05)
        fit = fit_tag(" ".join(w["w"] for w in l["words"]), back_font, BACKING_SIZE)
        add(appear, vanish, "Backing", f"{{\\pos({W // 2},{BACKING_Y})\\fad(150,200){fit}}}" + karaoke_text(l["words"], appear))

    return head + "\n".join(ev) + "\n"


def background(path):
    """Flat solid field."""
    Image.new("RGB", (W, H), BG).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lyrics")
    ap.add_argument("--audio", nargs="+", required=True, help="stems to mix into the backing track")
    ap.add_argument("--title", required=True)
    ap.add_argument("--artist", required=True)
    ap.add_argument("--fonts-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--intro", type=float, default=0.0, help="seconds of title card before the song starts")
    ap.add_argument("--credit", action="append", default=[], help="a line of small print for the title card")
    args = ap.parse_args()

    lyrics = json.loads(Path(args.lyrics).read_text())
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                            args.audio[0]], capture_output=True, text=True, check=True)
    song = float(probe.stdout)
    duration = song + args.intro
    lyrics = shift(lyrics, args.intro)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        background(tmp / "bg.png")
        (tmp / "lyrics.ass").write_text(build_ass(lyrics, args.title, args.artist, duration, args.intro, args.credit, args.fonts_dir), encoding="utf-8")

        inputs = ["-loop", "1", "-framerate", str(args.fps), "-i", str(tmp / "bg.png")]
        for a in args.audio:
            inputs += ["-i", a]
        n = len(args.audio)
        mix = "".join(f"[{i + 1}:a]" for i in range(n)) + (f"amix=inputs={n}:normalize=0,alimiter=limit=0.97," if n > 1 else "") \
            + f"adelay={int(args.intro * 1000)}:all=1[a]"
        bar = (f"color=c=0xec6fa6:s={W}x6:r={args.fps}[bar];"
               f"[0:v]ass={tmp / 'lyrics.ass'}:fontsdir={args.fonts_dir}[txt];"
               f"[txt][bar]overlay=x='-W+W*max(0,t-{args.intro:.3f})/{song:.3f}':y={H - 6}:shortest=1[v]")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-stats", *inputs,
               "-filter_complex", f"{mix};{bar}", "-map", "[v]", "-map", "[a]",
               "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-tune", "animation",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", args.out]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
