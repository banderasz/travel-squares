#!/usr/bin/env python3
"""Render a card image from its JSON, the way the Illustrator script did.

Reimplements image_generator/create_card_bg_landmark.js so any generated deck
gets artwork. Until now only the old 20_25 deck had pictures, and the training
image generator pairs artwork with symbols *by index* — so pointing it at a new
deck silently produced images that disagreed with their own labels.

Per quarter it draws a "landmark" whose shape says how many symbols are there:

    1 symbol   small square        3 symbols  triangle, pointing down
    2 symbols  hexagon             4 symbols  large square

with every dimension jittered +-10% so no two cards look stamped out. Each
landmark is roughened and then held inside its own quarter, so the four of them
never touch each other or run off the card; inside, one of pine/palm/tree/grass
is scattered and clipped to the land. The card itself is parchment with wave
glyphs on the open sea. Symbols sit at fixed positions on top.

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
# Half the sea channel between two neighbouring islands, and the margin from
# the card's edge. The script's 8 was a starting inset for shapes that then
# grew past it; here nothing leaves the padded box, so it is the channel width
# and 8 makes it wider than it needs to be.
PADDING = 4.0
# Fractions of the padded quarter box, which is also the ceiling: a landmark is
# scaled down if roughen or jitter would take it past its own quarter. The
# script's {1: 0.6, 2: 1.0, 3: 1.7, 4: 2.0} were drawn against a deck holding
# at most one symbol per quarter and let neighbouring shapes overlap freely.
SHAPE_SCALE = {1: 0.45, 2: 0.95, 3: 1.00, 4: 0.95}
# Fraction of a quarter. Set by the three-symbol triangle, which is the tightest
# shape to place in: it has to hold two symbols side by side near its wide top
# and a third down where it has narrowed, all clear of a coast that wanders.
# The other three shapes have room to spare at this size.
SYMBOL_SCALE = 0.23
JITTER_PCT = 10
SCATTER_SPACING = (25 * 0.7 * 0.7, 22 * 0.7 * 0.7)
SCATTER_ICON = 15 * 0.45
SCATTER_OPACITY = 235
SCATTER_TRIES = 600                  # dart throws per island
SCATTER_MAX = 10                     # scenery pieces drawn on one island
# The example sheet's coast is 0.025 of the card wide, but its cards carry at
# most one symbol per quarter. Ours carry four, so a line that heavy crowds
# them; half of it reads the same at card size and leaves the artwork room.
STROKE_WIDTH = 0.013
SUPERSAMPLE = 2                      # the outline is thick; draw it big, scale down

# Roughen, as a spectrum rather than a size and a detail. Taking the FFT of
# the island outlines in example_image.ai gives amplitudes falling off as 1/k
# from about the 4th harmonic to the 16th, at 0.42/k of the shape's mean
# radius — so a handful of broad lobes with progressively finer structure laid
# over them, and nothing above ~16 bumps around the perimeter. Synthesising
# that directly beats trying to reach it by stacking passes, which is what the
# earlier size/detail pairs were groping towards.
#
# The measured figures are 0.42 over harmonics 4-16. These are dialled back to
# a calmer coast with the fine structure carried further out, chosen off a
# comparison sheet: the measured values are honest about the printed shapes but
# read as fussy once four symbols share a quarter.
ROUGHEN_AMPLITUDE = 0.22
ROUGHEN_HARMONICS = (3, 14)

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
    """The landmark outline for a quarter holding n symbols.

    Every shape is built around its own bounding box, centred on the quarter,
    so SHAPE_SCALE reads directly as "this fraction of the quarter" whichever
    shape it is. That was not true while the triangle was described by its
    circumradius: scale 1.7 there meant a shape almost 1.5 quarters wide.
    """
    k = SHAPE_SCALE[n]
    w, h = _jitter(pw * k, rng), _jitter(ph * k, rng)
    if n in (1, 4):
        return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
                (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]
    if n == 2:
        return [(cx - w / 2, cy), (cx - w / 2, cy - h / 2), (cx, cy - h / 2),
                (cx + w / 2, cy), (cx + w / 2, cy + h / 2), (cx, cy + h / 2)]
    # 3 -> equilateral triangle, flat top, apex at the bottom. The script builds
    # an upward polygon then rotates it 180. That orientation is what the symbol
    # layout needs: two symbols sit along the wide top edge and one near the
    # point, so an upward triangle leaves the top two hanging outside it.
    h *= math.sqrt(3) / 2                      # equilateral, so height follows width
    return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx, cy + h / 2)]


def _fit_in_box(points, box):
    """Shrink and slide an outline until it sits wholly inside `box`.

    A quarter's landmark has to stay in its own quarter: overlapping the next
    one reads as a single island spanning both, and running off the card looks
    like a printing error. Jitter and roughen both work outwards from the base
    shape, though, and roughen is random, so the size that fits cannot be known
    before the fact. Clamping afterwards is exact where a conservative base
    size would only be likely. It scales about the outline's own centre and
    never grows anything, so a shape already inside is left alone.
    """
    x0, y0, x1, y1 = box
    ax0 = min(p[0] for p in points)
    ax1 = max(p[0] for p in points)
    ay0 = min(p[1] for p in points)
    ay1 = max(p[1] for p in points)
    k = min(1.0, (x1 - x0) / max(ax1 - ax0, 1e-9), (y1 - y0) / max(ay1 - ay0, 1e-9))
    mx, my = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    points = [(mx + (x - mx) * k, my + (y - my) * k) for x, y in points]

    ax0, ax1 = mx + (ax0 - mx) * k, mx + (ax1 - mx) * k
    ay0, ay1 = my + (ay0 - my) * k, my + (ay1 - my) * k
    dx = max(0.0, x0 - ax0) - max(0.0, ax1 - x1)
    dy = max(0.0, y0 - ay0) - max(0.0, ay1 - y1)
    return [(x + dx, y + dy) for x, y in points]


def _scatter(size, mask, scale, rng, keepout=(), bbox=None):
    """Dot one island with one kind of scenery, avoiding the symbols.

    A grass island, a palm island, a pine island: the kind is drawn once here,
    so a single island never mixes them. Called per island rather than per
    card, which is what makes that possible -- the mask is that island alone.

    The Illustrator script tiles a pattern across the whole shape and lets the
    symbols land on top, which buries most of it — the visible scenery ends up
    being whatever happens to fall in the gaps. Here each position is tested
    first: it has to sit clear of every symbol's box and wholly on land, or it
    is dropped. Fewer get drawn, but the ones that do are all visible, so they
    can carry proper weight instead of being faded to near-nothing.
    """
    target = SCATTER_ICON * scale
    art = _load(SCATTER[rng.randrange(len(SCATTER))], int(max(target, 4)))
    ratio = target / max(art.size)
    icon = art.resize((max(1, int(art.width * ratio)),
                       max(1, int(art.height * ratio))), Image.LANCZOS)
    probe = mask.load()
    w, h = size
    bx0, by0, bx1, by1 = bbox if bbox else (0, 0, w, h)
    spacing = SCATTER_SPACING[0] * scale
    layer = Image.new('RGBA', size, (0, 0, 0, 0))

    # Dart-throwing rather than a grid. A card can carry sixteen symbols, and
    # the boxes reserved for them leave the island so broken up that a regular
    # grid lands almost every point on one and draws nothing at all. Trying
    # many random spots and keeping those that clear the symbols, the coast and
    # each other finds the gaps wherever they happen to be.
    taken = []
    for _ in range(SCATTER_TRIES):
        if len(taken) >= SCATTER_MAX:
            break
        cx, cy = rng.uniform(bx0, bx1), rng.uniform(by0, by1)
        box = (cx - icon.width / 2, cy - icon.height / 2,
               cx + icon.width / 2, cy + icon.height / 2)
        if not _on_land(probe, box, w, h):
            continue
        if _hits(box, keepout, target * 0.15) or _hits(box, taken, spacing * 0.5):
            continue
        taken.append(box)
        layer.alpha_composite(icon, (int(box[0]), int(box[1])))

    layer.putalpha(layer.getchannel('A').point(lambda v: v * SCATTER_OPACITY // 255))
    return layer


def _on_land(probe, box, w, h):
    """True when the whole icon sits inside the island, not over its coast."""
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return False
    return all(probe[int(x), int(y)] > 200
               for x in (x0, (x0 + x1) / 2, x1) for y in (y0, (y0 + y1) / 2, y1))


def _hits(box, boxes, margin):
    x0, y0, x1, y1 = box
    return any(x0 < bx1 + margin and bx0 - margin < x1 and
               y0 < by1 + margin and by0 - margin < y1
               for bx0, by0, bx1, by1 in boxes)


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
    # an earlier quarter's symbols. Landmarks are confined to their own quarter
    # now and cannot reach a neighbour's symbols, but the order still matters
    # for the scenery, which has to know where every symbol will land.
    placements, outlines = [], []
    for qi, name in enumerate(QUARTERS):
        symbols = [Symbols.of(x) for x in card['card']['quarters'].get(name, [])]
        n = len(symbols)
        qx, qy = (qi % 2) * q, (qi // 2) * q
        pw = ph = q - 2 * pad
        cx, cy = qx + q / 2, qy + q / 2
        if n:
            # Leave room for the stroke, which straddles the outline, so the
            # coastline itself stops short of the quarter's edge too.
            edge = STROKE_WIDTH * px / 2
            box = (qx + pad + edge, qy + pad + edge,
                   qx + q - pad - edge, qy + q - pad - edge)
            outlines.append(
                _fit_in_box(roughen(_shape_points(n, cx, cy, pw, ph, rng), rng), box))
        # Fractions of the padded quarter. Pulled well in from the quarter's
        # corners, which is where a roughened coast is furthest from the box it
        # is drawn in -- the corners get blunted before the roughen even starts.
        # These are the tightest grouping that still keeps two symbols from
        # touching, searched against the outlines this renderer actually
        # produces rather than against the ideal polygons.
        spots = {
            1: [(0.50, 0.50)],
            2: [(0.32, 0.32), (0.68, 0.68)],
            3: [(0.36, 0.28), (0.64, 0.28), (0.50, 0.56)],
            4: [(0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65)],
        }.get(n, [])
        for symbol, (fx, fy) in zip(symbols, spots):
            placements.append((symbol, qx + pad + pw * fx, qy + pad + ph * fy))

    # Stroke every outline first, then fill them all. The landmarks are in
    # separate quarters and no longer touch, so this is no longer hiding seams
    # between overlapping shapes, but it still saves stroking the inner half of
    # a line that the fill would only cover again. Drawn at SUPERSAMPLE and
    # scaled down, because a hard-edged stroke aliases badly.
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
        img.alpha_composite(island)

        # Every symbol is scaled to fit a target-by-target box, so reserving that
        # box keeps the scenery out from under artwork that has not been drawn
        # yet. Slightly generous for art that is not square, which is the right
        # way to be wrong here.
        half = q * SYMBOL_SCALE / 2
        keepout = [(sx - half, sy - half, sx + half, sy + half)
                   for _, sx, sy in placements]
        # One scatter pass per island, each picking its own kind of scenery, so
        # an island is all grass or all palm rather than a mixture. Each pass
        # gets that island's own mask -- the fill, not the drawn island's alpha,
        # which includes the outline and its soft downsampled edge and would put
        # trees on the coast and a few pixels out to sea.
        for pts in scaled:
            one = Image.new('L', (px * ss, px * ss), 0)
            ImageDraw.Draw(one).polygon(pts, fill=255)
            bbox = (min(x for x, _ in pts) / ss, min(y for _, y in pts) / ss,
                    max(x for x, _ in pts) / ss, max(y for _, y in pts) / ss)
            img.alpha_composite(_scatter(img.size, one.resize((px, px), Image.LANCZOS),
                                         scale, rng, keepout, bbox))

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
