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

import numpy as np
from PIL import Image

W, H = 1920, 1080
LEAD_ROWS = (600, 730)       # baseline y of the two alternating lead rows
BACKING_Y = 880
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


def build_ass(lyrics, title, artist, duration):
    lead, back = lyrics["lead"], lyrics["backing"]
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lead,Bricolage Grotesque ExtraBold,84,{ass_color('#ec6fa6')},{ass_color('#5b6474')},{ass_color('#0b0d11')},{ass_color('#000000', 0x80)},0,0,0,0,100,100,0,0,1,3,0,2,110,110,0,1
Style: Backing,Bricolage Grotesque Medium,58,{ass_color('#4fbdd1')},{ass_color('#3a4a55')},{ass_color('#0b0d11')},{ass_color('#000000', 0x80)},0,1,0,0,100,100,0,0,1,2.5,0,2,110,110,0,1
Style: Label,IBM Plex Mono Medium,26,{ass_color('#9aa3b0')},{ass_color('#9aa3b0')},{ass_color('#000000', 0xFF)},{ass_color('#000000', 0xFF)},0,0,0,0,100,100,2,0,1,0,0,7,0,0,0,1
Style: Count,Bricolage Grotesque ExtraBold,64,{ass_color('#ec6fa6')},{ass_color('#2a303a')},{ass_color('#000000', 0xFF)},{ass_color('#000000', 0xFF)},0,0,0,0,100,100,24,0,1,0,0,2,0,0,0,1
Style: Title,Bricolage Grotesque ExtraBold,120,{ass_color('#e8ebf0')},{ass_color('#e8ebf0')},{ass_color('#000000', 0xFF)},{ass_color('#000000', 0xFF)},0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1
Style: Artist,Bricolage Grotesque Medium,56,{ass_color('#9aa3b0')},{ass_color('#9aa3b0')},{ass_color('#000000', 0xFF)},{ass_color('#000000', 0xFF)},0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    add = lambda start, end, style, text: ev.append(f"Dialogue: 0,{ts(start)},{ts(end)},{style},,0,0,0,,{text}")

    # header labels for the whole song
    add(0, duration, "Label", f"{{\\pos(110,70)}}{title.upper()}  ·  {artist.upper()}")
    add(0, duration, "Label", f"{{\\an9\\pos({W - 110},70)}}{{\\c{ass_color('#ec6fa6')}}}●{{\\c{ass_color('#9aa3b0')}}} LEAD   "
                              f"{{\\c{ass_color('#4fbdd1')}}}●{{\\c{ass_color('#9aa3b0')}}} BACKING")

    # title card until the first line can appear
    first = min([l["start"] for l in lead + back] or [duration])
    if first > 3:
        add(0, first - PRE_ROLL + 0.5, "Title", f"{{\\pos({W // 2},{H // 2 - 40})\\fad(400,400)}}{title}")
        add(0, first - PRE_ROLL + 0.5, "Artist", f"{{\\pos({W // 2},{H // 2 + 60})\\fad(400,400)}}{artist}")

    # lead: alternate rows; a line appears when the previous one starts, so the next line is always visible
    for i, l in enumerate(lead):
        prev_start = lead[i - 1]["start"] if i else 0.0
        appear = max(l["start"] - PRE_ROLL, prev_start)
        nxt = lead[i + 1]["start"] if i + 1 < len(lead) else duration
        vanish = min(l["end"] + 0.5, max(nxt, l["end"] + 0.1))
        y = LEAD_ROWS[i % 2]
        add(appear, vanish, "Lead", f"{{\\pos({W // 2},{y})\\fad(150,200)}}" + karaoke_text(l["words"], appear))

        gap = l["start"] - (lead[i - 1]["end"] if i else 0.0)
        if gap >= COUNTDOWN_GAP and l["start"] >= 3:
            # three dots that fill one per second before the entry
            c0 = l["start"] - 3
            add(c0, l["start"], "Count", f"{{\\pos({W // 2},{y - 110})\\fad(200,0)}}{{\\kf100}}●{{\\kf100}}●{{\\kf100}}●")

    # instrumental breaks get a quiet marker
    for a, b in zip(lead, lead[1:]):
        if b["start"] - a["end"] > 15:
            add(a["end"] + 1.5, b["start"] - 4, "Artist", f"{{\\pos({W // 2},{LEAD_ROWS[0] + 40})\\fad(600,600)}}♪  instrumental  ♪")

    for i, l in enumerate(back):
        prev_end = back[i - 1]["end"] + 0.3 if i else 0.0
        appear = max(l["start"] - 1.5, prev_end)
        nxt = back[i + 1]["start"] - 1.5 if i + 1 < len(back) else duration
        vanish = max(min(l["end"] + 0.5, nxt), l["end"] + 0.05)
        add(appear, vanish, "Backing", f"{{\\pos({W // 2},{BACKING_Y})\\fad(150,200)}}" + karaoke_text(l["words"], appear))

    return head + "\n".join(ev) + "\n"


def background(path):
    """Dark field with a soft glow behind the lyric rows."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.array([16, 19, 24], np.float32)
    glow = np.exp(-(((x - W / 2) / (W * 0.45)) ** 2 + ((y - 690) / (H * 0.32)) ** 2))
    tint = np.array([60, 30, 50], np.float32)
    img = base + glow[..., None] * tint * 0.35
    img += np.random.default_rng(0).normal(0, 1.2, img.shape)  # a little grain to stop banding
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lyrics")
    ap.add_argument("--audio", nargs="+", required=True, help="stems to mix into the backing track")
    ap.add_argument("--title", required=True)
    ap.add_argument("--artist", required=True)
    ap.add_argument("--fonts-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    lyrics = json.loads(Path(args.lyrics).read_text())
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                            args.audio[0]], capture_output=True, text=True, check=True)
    duration = float(probe.stdout)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        background(tmp / "bg.png")
        (tmp / "lyrics.ass").write_text(build_ass(lyrics, args.title, args.artist, duration), encoding="utf-8")

        inputs = ["-loop", "1", "-framerate", str(args.fps), "-i", str(tmp / "bg.png")]
        for a in args.audio:
            inputs += ["-i", a]
        n = len(args.audio)
        mix = "".join(f"[{i + 1}:a]" for i in range(n)) + (f"amix=inputs={n}:normalize=0,alimiter=limit=0.97[a]" if n > 1 else "anull[a]")
        bar = (f"color=c=0xec6fa6:s={W}x6:r={args.fps}[bar];"
               f"[0:v]ass={tmp / 'lyrics.ass'}:fontsdir={args.fonts_dir}[txt];"
               f"[txt][bar]overlay=x='-W+W*t/{duration:.3f}':y={H - 6}:shortest=1[v]")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-stats", *inputs,
               "-filter_complex", f"{mix};{bar}", "-map", "[v]", "-map", "[a]",
               "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-tune", "animation",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", args.out]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
