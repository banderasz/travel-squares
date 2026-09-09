#!/usr/bin/env python3
"""Render a card image from its JSON, the way the Illustrator script did.

Reimplements image_generator/create_card_bg_landmark.js so any generated deck
gets artwork. Until now only the old 20_25 deck had pictures, and the training
image generator pairs artwork with symbols *by index* — so pointing it at a new
deck silently produced images that disagreed with their own labels.

Per quarter it draws a "landmark" whose shape says how many symbols are there:

    1 symbol   small square        3 symbols  triangle, pointing down
    2 symbols  hexagon             4 symbols  large square

with every dimension jittered +-10% so no two cards look stamped out. The
outlines are roughened, merged into one island, and stroked once around the
whole coast; inside, one of pine/palm/tree/grass is scattered at half opacity
and clipped to the land. The card itself is parchment with wave glyphs on the
open sea. Symbols sit at fixed positions on top.

Palette, stroke weight and roughen spectrum are measured off
image_generator/example_image.ai, a printed sheet of finished cards, rather
than converted from the values in the script — those came out far too
saturated. Where the script's own numbers disagree with the sheet, the sheet
wins, and the difference is noted at the constant.

    python -m src.training_images.render_card --deck decks/grown_t20.json --out cards/
"""
import argparse
import json
import math
import os
import random

from PIL import Image, ImageChops, ImageDraw

from src.symbols import Symbols

ASSETS = 'image_generator/symbols'
SCATTER = ['bg_1', 'bg_2', 'bg_3', 'bg_4']      # pine, palm, tree, grass
QUARTERS = ('top_left', 'top_right', 'bottom_left', 'bottom_right')

# Sampled straight off image_generator/example_image.ai, a printed sheet of
# finished cards. These replace values converted from the CMYK in the script,
# which came out far too saturated — the real palette is muted.
SEA = (251, 236, 203, 255)           # the card itself: parchment
ISLAND = (199, 176, 142, 255)
OUTLINE = (80, 71, 58, 255)
SEA_MARK = (224, 208, 178, 255)      # the little wave glyphs on the open sea

# All from the Illustrator script, expressed against its 135pt card.
REF_CARD = 135.0
PADDING = 8.0
# The script's own scales are {1: 0.6, 2: 1.0, 3: 1.7, 4: 2.0}, but those were
# drawn against a deck whose quarters held at most one symbol. Ours hold up to
# four, and at scale 2.0 a quarter's shape is twice the size its symbols need,
# so the four of them merge into one slab covering the whole card instead of an
# island with sea around it. These are sized to clear the symbols they contain.
SHAPE_SCALE = {1: 0.60, 2: 1.15, 3: 1.60, 4: 1.25}
SYMBOL_SCALE = 0.45 * 0.8            # tSize = qW*0.45, then symbolScale 0.8
JITTER_PCT = 10
SCATTER_SPACING = (25 * 0.7 * 1.8, 22 * 0.7 * 1.8)
SCATTER_ICON = 15 * 0.4
SCATTER_OPACITY = 128                # the script uses opacity 50 (of 100)
STROKE_WIDTH = 0.025                 # of the card, measured off the example sheet
SUPERSAMPLE = 2                      # the outline is thick; draw it big, scale down

# Roughen, as a spectrum rather than a size and a detail. Taking the FFT of
# the island outlines in example_image.ai gives amplitudes falling off as 1/k
# from about the 4th harmonic to the 16th, at 0.42/k of the shape's mean
# radius — so a handful of broad lobes with progressively finer structure laid
# over them, and nothing above ~16 bumps around the perimeter. Synthesising
# that directly beats trying to reach it by stacking passes, which is what the
# earlier size/detail pairs were groping towards.
ROUGHEN_AMPLITUDE = 0.42
ROUGHEN_HARMONICS = (4, 16)

# Wave glyphs on the open sea, as fractions of the card. A staggered grid,
# every other row offset by half a step.
WAVE_STEP = (0.50, 0.43)
WAVE_SIZE = 0.22
WAVE_HUMPS = 4


# Asset filenames, keyed by stable symbol ID. They mostly match the display
# names, but treasure's artwork is called "gem", and only one arrow was drawn —
# the other three are rotations of it.
ASSET = {
    Symbols.CIRCLE: 'food',   Symbols.SQUARE: 'gem',    Symbols.STAR: 'weapon',
    Symbols.SUN: 'coin',      Symbols.DIAMOND: 'parrot', Symbols.TRIANGLE: 'rum',
    Symbols.X: 'rat',         Symbols.MOON: 'snake',    Symbols.SKULL: 'mask',
}
ARROW_ROTATION = {Symbols.ARROW_DOWN: 0, Symbols.ARROW_RIGHT: 90,
                  Symbols.ARROW_UP: 180, Symbols.ARROW_LEFT: 270}

# Prefer the SVG art so nothing is pixelated at any card size; fall back to the
# exported PNGs if cairo is not installed. Homebrew puts libcairo outside the
# default dyld search path, so point at it before cairosvg tries to load it.
for _lib in ('/opt/homebrew/lib', '/usr/local/lib'):
    if os.path.isdir(_lib):
        os.environ.setdefault('DYLD_FALLBACK_LIBRARY_PATH', _lib)
        break
try:
    import cairosvg
    cairosvg.svg2png(url=os.path.join(ASSETS, 'food.svg'), output_width=8)
    HAVE_SVG = True
except Exception:
    HAVE_SVG = False


def _load(name, px=None, cache={}):
    """Symbol art. SVG when cairo is installed (crisp at any size), else PNG."""
    key = (name, px if HAVE_SVG else None)
    if key not in cache:
        svg = os.path.join(ASSETS, f'{name}.svg')
        if HAVE_SVG and os.path.exists(svg):
            import io
            data = cairosvg.svg2png(url=svg, output_width=px, output_height=None)
            cache[key] = Image.open(io.BytesIO(data)).convert('RGBA')
        else:
            cache[key] = Image.open(os.path.join(ASSETS, f'{name}.png')).convert('RGBA')
    return cache[key]


def symbol_art(symbol, px):
    """The artwork for a symbol, rotated if it is an arrow."""
    if symbol in ARROW_ROTATION:
        art = _load('arrow_down', px)
        angle = ARROW_ROTATION[symbol]
        return art.rotate(angle, expand=True, resample=Image.BICUBIC) if angle else art
    return _load(ASSET[symbol], px)


def _jitter(value, rng):
    return value * (1 + (rng.random() * 2 - 1) * JITTER_PCT / 100)


def _resample_even(points, count):
    """`count` points spaced equally along the closed outline.

    Equal spacing is what lets the index double as the arc-length parameter,
    so harmonic k really is k bumps around the perimeter.
    """
    n = len(points)
    seg = [math.dist(points[i], points[(i + 1) % n]) for i in range(n)]
    perimeter = sum(seg) or 1.0
    out, i, walked = [], 0, 0.0
    for s in range(count):
        want = s / count * perimeter
        while walked + seg[i] < want and i < n - 1:
            walked += seg[i]
            i += 1
        t = (want - walked) / (seg[i] or 1.0)
        ax, ay = points[i]
        bx, by = points[(i + 1) % n]
        out.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    return out


def roughen(points, rng, amplitude=None, harmonics=None, samples=512):
    """Push the outline in and out along its own normal, smoothly.

    The displacement is a sum of sinusoids around the perimeter with random
    phases and 1/k amplitudes — the spectrum measured off the example sheet.
    Building it this way rather than by jogging individual points is the whole
    trick: neighbouring samples then move *together*. Independent per-point
    randomness is white noise, and white noise reverses direction at every
    point no matter how smoothly the result is interpolated, which is what
    made earlier attempts look torn rather than weathered.
    """
    # Read the module globals here rather than as default arguments, which bind
    # once at import and so cannot be overridden to try a different look.
    amplitude = ROUGHEN_AMPLITUDE if amplitude is None else amplitude
    harmonics = ROUGHEN_HARMONICS if harmonics is None else harmonics

    dense = _round_corners(_resample_even(points, samples), samples // 20)
    cx = sum(p[0] for p in dense) / samples
    cy = sum(p[1] for p in dense) / samples
    radius = sum(math.dist(p, (cx, cy)) for p in dense) / samples

    k_lo, k_hi = harmonics
    waves = [(k, amplitude / k * radius, rng.random() * math.tau)
             for k in range(k_lo, k_hi + 1)]

    span = max(2, samples // 32)
    out = []
    for i, (x, y) in enumerate(dense):
        t = i / samples * math.tau
        offset = sum(amp * math.sin(k * t + phase) for k, amp, phase in waves)
        (ax, ay), (bx, by) = dense[(i - 1) % samples], dense[(i + 1) % samples]
        tx, ty = bx - ax, by - ay
        length = math.hypot(tx, ty) or 1.0
        limit = _curve_radius(dense[(i - span) % samples], (x, y),
                              dense[(i + span) % samples]) * 0.6
        offset = max(-limit, min(limit, offset))
        out.append((x + ty / length * offset, y - tx / length * offset))
    return out


def _round_corners(points, window):
    """Blunt the base polygon's corners with a moving average.

    The corners are where loops come from. A normal derived from neighbouring
    samples swings through ninety degrees in the two samples either side of a
    square's corner, so displacing along it folds the outline back over itself
    and leaves a little knot in the coast. Rounded corners have a normal that
    turns gradually, and the islands on the printed sheet are rounded anyway.
    """
    n = len(points)
    if window < 1:
        return points
    return [(sum(points[(i + j) % n][0] for j in range(-window, window + 1))
             / (2 * window + 1),
             sum(points[(i + j) % n][1] for j in range(-window, window + 1))
             / (2 * window + 1)) for i in range(n)]


def _curve_radius(a, b, c):
    """Radius of the circle through three points — how sharply the coast bends.

    Displacing further than this towards the centre of the bend turns the
    outline inside out, so it is the ceiling on how deep a bay can cut.
    """
    ax, ay = a
    bx, by = b
    cx_, cy_ = c
    area = abs((bx - ax) * (cy_ - ay) - (cx_ - ax) * (by - ay)) / 2.0
    if area < 1e-9:
        return float('inf')
    return math.dist(a, b) * math.dist(b, c) * math.dist(a, c) / (4.0 * area)


def _shape_points(n, cx, cy, pw, ph, rng):
    """The landmark outline for a quarter holding n symbols."""
    k = SHAPE_SCALE[n]
    if n == 4 or n == 1:
        w = _jitter((pw * 0.5 if n == 4 else pw) * k, rng)
        h = _jitter((ph * 0.5 if n == 4 else pw) * k, rng)
        return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
                (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]
    if n == 2:
        w, h = _jitter(pw * k, rng), _jitter(ph * k, rng)
        return [(cx - w / 2, cy), (cx - w / 2, cy - h / 2), (cx, cy - h / 2),
                (cx + w / 2, cy), (cx + w / 2, cy + h / 2), (cx, cy + h / 2)]
    # 3 -> triangle with a flat top and the apex at the bottom. The script builds
    # an upward polygon then rotates it 180. That orientation is what the symbol
    # layout needs: two symbols sit along the wide top edge and one near the
    # point, so an upward triangle leaves the top two hanging outside it.
    #
    # Its scale is 2.0, not the script's 1.7. At 1.7 the triangle has already
    # narrowed to about the width of one symbol by the time it reaches the
    # bottom one, so that symbol overhangs the edge -- and the roughen then eats
    # into what little margin is left. 2.0 keeps the apex wide enough to hold it.
    r = _jitter(pw * 0.5 * k, rng)
    h = _jitter(ph * 0.5 * k, rng)
    cy2 = cy - h / 8
    return [(cx - r * math.cos(math.pi / 6), cy2 - r * math.sin(math.pi / 6)),
            (cx + r * math.cos(math.pi / 6), cy2 - r * math.sin(math.pi / 6)),
            (cx, cy2 + r)]


def _scatter(size, mask, scale, rng):
    """Tile one environment symbol over the island, clipped to `mask`."""
    target = SCATTER_ICON * scale
    icon = _load(rng.choice(SCATTER), int(max(target, 4)))
    ratio = target / max(icon.size)
    icon = icon.resize((max(1, int(icon.width * ratio)), max(1, int(icon.height * ratio))),
                       Image.LANCZOS)
    sx, sy = SCATTER_SPACING[0] * scale, SCATTER_SPACING[1] * scale

    layer = Image.new('RGBA', size, (0, 0, 0, 0))
    row, y = 0, -sy
    while y < size[1] + sy:
        offset = 0 if row % 2 == 0 else sx / 2
        x = -sx
        while x < size[0] + sx:
            layer.alpha_composite(icon, (int(x + offset - icon.width / 2),
                                         int(y - icon.height / 2)))
            x += sx
        y += sy
        row += 1

    clip = mask.point(lambda v: v * SCATTER_OPACITY // 255)
    layer.putalpha(ImageChops.multiply(layer.getchannel('A'), clip))
    return layer


def _stroke(pen, points, width):
    """Draw a closed outline by stamping a round brush along it.

    Not ImageDraw.line: a thick polyline is drawn as one quad per segment, and
    on the outside of a bend consecutive quads fan apart and leave a wedge of
    background showing. Its joint='curve' does not close those at the shallow
    turn angles a 512-point outline produces, so the coast ends up combed with
    radial slits. Stamping overlapping discs cannot gap, and gives the round
    join and cap the original has anyway.
    """
    step = max(1.0, width / 6.0)
    radius = width / 2.0
    n = len(points)
    seg = [math.dist(points[i], points[(i + 1) % n]) for i in range(n)]

    walked, i = 0.0, 0                  # `want` only grows, so walk once
    want = 0.0
    while i < n:
        if want > walked + seg[i]:
            walked += seg[i]
            i += 1
            continue
        t = (want - walked) / (seg[i] or 1.0)
        ax, ay = points[i]
        bx, by = points[(i + 1) % n]
        x, y = ax + (bx - ax) * t, ay + (by - ay) * t
        pen.ellipse((x - radius, y - radius, x + radius, y + radius), fill=OUTLINE)
        want += step


def _sea(size, scale, rng):
    """The parchment the island sits on, sprinkled with little wave glyphs."""
    px = size[0]
    img = Image.new('RGBA', size, SEA)
    draw = ImageDraw.Draw(img)
    step_x, step_y = WAVE_STEP[0] * px, WAVE_STEP[1] * px
    width = max(1, round(0.004 * px))
    row, y = 0, rng.uniform(-step_y, 0)
    while y < px + step_y:
        offset = (0 if row % 2 == 0 else step_x / 2) + rng.uniform(-0.02, 0.02) * px
        x = -step_x
        while x < px + step_x:
            span, amp = WAVE_SIZE * px, 0.012 * px
            pts = [(x + offset + span * t / 40,
                    y + math.sin(t / 40 * WAVE_HUMPS * math.tau) * amp)
                   for t in range(41)]
            draw.line(pts, fill=SEA_MARK, width=width, joint='curve')
            x += step_x
        y += step_y
        row += 1
    return img


def render_card(card, px=512, seed=None):
    rng = random.Random(seed)
    scale = px / REF_CARD
    img = _sea((px, px), scale, rng)
    q = px / 2
    pad = PADDING * scale

    # Two passes: every landmark first, then every symbol. The Illustrator
    # script did one quarter at a time, so a later quarter's shape could cover
    # an earlier quarter's symbols — the shapes are up to 1.7x the quarter and
    # do overlap. That matters here because the annotations would still call
    # those symbols visible, which is exactly the mislabelling this renderer
    # exists to avoid.
    placements, outlines = [], []
    for qi, name in enumerate(QUARTERS):
        symbols = [Symbols.of(x) for x in card['card']['quarters'].get(name, [])]
        n = len(symbols)
        qx, qy = (qi % 2) * q, (qi // 2) * q
        pw = ph = q - 2 * pad
        cx, cy = qx + q / 2, qy + q / 2
        if n:
            outlines.append(roughen(_shape_points(n, cx, cy, pw, ph, rng), rng))
        spots = {
            1: [(0.50, 0.50)],
            2: [(0.25, 0.25), (0.75, 0.75)],
            3: [(0.25, 0.25), (0.75, 0.25), (0.50, 0.75)],
            4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
        }.get(n, [])
        for symbol, (fx, fy) in zip(symbols, spots):
            placements.append((symbol, qx + pad + pw * fx, qy + pad + ph * fy))

    # One island, not four landmarks. Neighbouring quarters' shapes overlap, and
    # stroking each on its own leaves the seams showing as lines across the
    # middle of the island; the example sheet has a single unbroken coastline.
    # Stroking every outline first and only then filling them all hides each
    # seam under the next shape's fill, which unions them without needing any
    # polygon arithmetic. Drawn at SUPERSAMPLE and scaled down, because a hard
    # -edged 2.5%-of-card stroke aliases badly.
    if outlines:
        ss = SUPERSAMPLE
        big = Image.new('RGBA', (px * ss, px * ss), (0, 0, 0, 0))
        pen = ImageDraw.Draw(big)
        scaled = [[(x * ss, y * ss) for x, y in pts] for pts in outlines]
        for pts in scaled:      # stroke double width; the fill eats the inner half
            _stroke(pen, pts, max(1, round(2 * STROKE_WIDTH * px * ss)))
        for pts in scaled:
            pen.polygon(pts, fill=ISLAND)
        island = big.resize((px, px), Image.LANCZOS)
        # Clip the scatter to the fill, not to the island's alpha: the alpha
        # includes the outline and its soft downsampled edge, which puts trees
        # on the coast and a few pixels out to sea.
        inland = Image.new('L', (px * ss, px * ss), 0)
        ink = ImageDraw.Draw(inland)
        for pts in scaled:
            ink.polygon(pts, fill=255)
        img.alpha_composite(island)
        img.alpha_composite(_scatter(img.size, inland.resize((px, px), Image.LANCZOS),
                                     scale, rng))

    target = q * SYMBOL_SCALE
    boxes = []
    for symbol, px_, py_ in placements:
        art = symbol_art(symbol, int(target))
        r = target / max(art.size)
        if abs(r - 1) > 0.01:
            art = art.resize((max(1, int(art.width * r)), max(1, int(art.height * r))),
                             Image.LANCZOS)
        x0, y0 = int(px_ - art.width / 2), int(py_ - art.height / 2)
        img.alpha_composite(art, (x0, y0))
        boxes.append({'symbol': symbol.name,
                      'box': [x0, y0, x0 + art.width, y0 + art.height]})

    return img, boxes


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--deck', default='decks/grown_t20.json')
    p.add_argument('--out', default='cards')
    p.add_argument('--px', type=int, default=512)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    cards = json.load(open(a.deck))
    if a.limit:
        cards = cards[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    for i, card in enumerate(cards, 1):
        img, _ = render_card(card, a.px, seed=a.seed + i)
        img.save(os.path.join(a.out, f'card-{i:03d}.png'))
    print(f"  rendered {len(cards)} cards -> {a.out}/")


if __name__ == '__main__':
    main()
