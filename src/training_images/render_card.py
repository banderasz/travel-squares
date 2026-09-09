#!/usr/bin/env python3
"""Render a card image from its JSON, the way the Illustrator script did.

Reimplements image_generator/create_card_bg_landmark.js so any generated deck
gets artwork. Until now only the old 20_25 deck had pictures, and the training
image generator pairs artwork with symbols *by index* — so pointing it at a new
deck silently produced images that disagreed with their own labels.

Per quarter it draws a "landmark" whose shape says how many symbols are there:

    1 symbol   small square        3 symbols  triangle, pointing down
    2 symbols  hexagon             4 symbols  large square

filled light brown, stroked dark brown, with every dimension jittered +-10% so
no two cards look stamped out. The outline is then roughened — Illustrator's
Roughen effect, which walks the perimeter and displaces each point — and the
inside is scattered with one of pine/palm/tree/grass at half opacity, clipped
to the shape. Symbols sit at fixed positions on top.

    python -m src.training_images.render_card --deck decks/grown_t20.json --out cards/
"""
import argparse
import json
import math
import os
import random

from PIL import Image, ImageDraw

from src.symbols import Symbols

ASSETS = 'image_generator/symbols'
SCATTER = ['bg_1', 'bg_2', 'bg_3', 'bg_4']      # pine, palm, tree, grass
QUARTERS = ('top_left', 'top_right', 'bottom_left', 'bottom_right')

LIGHT_BROWN = (168, 118, 68, 255)    # CMYK 30/50/70/10
DARK_BROWN = (77, 45, 20, 255)       # CMYK 50/70/90/50

# All from the Illustrator script, expressed against its 135pt card.
REF_CARD = 135.0
PADDING = 8.0
SHAPE_SCALE = {1: 0.60, 2: 1.00, 3: 2.00, 4: 2.00}
SYMBOL_SCALE = 0.45 * 0.8            # tSize = qW*0.45, then symbolScale 0.8
JITTER_PCT = 10
SCATTER_SPACING = (25 * 0.7 * 1.8, 22 * 0.7 * 1.8)
SCATTER_ICON = 15 * 0.7
SCATTER_OPACITY = 128                # the script uses opacity 50 (of 100)
# Roughen, as (size, detail) passes applied one after another. Size is a
# fraction of the shape's bounding-box diagonal; detail is bumps per inch
# (72pt), so it sets the wavelength. Illustrator's own effect is a single
# pass, but one pass can only produce one wavelength: crank the size and you
# get a few huge lobes, crank the detail and you get an even fuzz. A coastline
# has both, so the coarse pass sets the silhouette and the fine pass crinkles
# the edge it leaves.
ROUGHEN_PASSES = ((0.075, 4.5), (0.022, 13.0))          # size, detail


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


def _resample(points, spacing):
    """Walk the closed outline and drop a point every `spacing` along it."""
    out = []
    for i in range(len(points)):
        ax, ay = points[i]
        bx, by = points[(i + 1) % len(points)]
        steps = max(1, round(math.hypot(bx - ax, by - ay) / spacing))
        for s in range(steps):
            t = s / steps
            out.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    return out


def _loop_noise(n_out, n_control, rng):
    """`n_out` smooth values in roughly [-1, 1], wrapping seamlessly.

    A few random control values with a Catmull-Rom spline read off between
    them, so neighbouring samples move *together*. This is the whole trick:
    displacing each point independently gives white noise, and white noise
    reverses direction at every point no matter how smoothly you interpolate
    the result — which is what made earlier versions look torn.
    """
    ctrl = [rng.uniform(-1.0, 1.0) for _ in range(n_control)]
    out = []
    for i in range(n_out):
        u = i / n_out * n_control
        k = int(u)
        t = u - k
        p0, p1 = ctrl[(k - 1) % n_control], ctrl[k % n_control]
        p2, p3 = ctrl[(k + 1) % n_control], ctrl[(k + 2) % n_control]
        t2, t3 = t * t, t * t * t
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return out


def _roughen_pass(points, size, wavelength, rng):
    """Push the outline in and out along its own normal, smoothly."""
    dense = _resample(points, max(wavelength / 10.0, 0.4))
    n = len(dense)
    perimeter = sum(math.dist(dense[i], dense[(i + 1) % n]) for i in range(n))
    noise = _loop_noise(n, max(3, round(perimeter / wavelength)), rng)

    out = []
    for i in range(n):
        (ax, ay), (bx, by) = dense[(i - 1) % n], dense[(i + 1) % n]
        tx, ty = bx - ax, by - ay
        length = math.hypot(tx, ty) or 1.0
        nx, ny = ty / length, -tx / length          # outward normal
        x, y = dense[i]
        out.append((x + nx * size * noise[i], y + ny * size * noise[i]))
    return out


def roughen(points, scale, rng, passes=ROUGHEN_PASSES):
    """Illustrator's Roughen, run at more than one wavelength.

    Each pass displaces the outline along its normal by smooth noise of one
    wavelength: the coarse pass shapes the silhouette, the fine one crinkles
    the edge it leaves. Size stays measured against the original bounding box
    so a later pass does not compound the displacement of an earlier one.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    for size_pct, detail_per_inch in passes:
        points = _roughen_pass(points, size_pct * diagonal,
                               (72.0 / detail_per_inch) * scale, rng)
    return points


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


def _scatter(size, outline, scale, rng):
    """Tile one environment symbol across the shape, clipped to it."""
    target = SCATTER_ICON * scale
    icon = _load(rng.choice(SCATTER), int(max(target, 4)))
    ratio = target / max(icon.size)
    icon = icon.resize((max(1, int(icon.width * ratio)), max(1, int(icon.height * ratio))),
                       Image.LANCZOS)
    sx, sy = SCATTER_SPACING[0] * scale, SCATTER_SPACING[1] * scale

    layer = Image.new('RGBA', size, (0, 0, 0, 0))
    xs = [p[0] for p in outline]
    ys = [p[1] for p in outline]
    row = 0
    y = min(ys) - sy
    while y < max(ys) + sy:
        offset = 0 if row % 2 == 0 else sx / 2
        x = min(xs) - sx
        while x < max(xs) + sx:
            layer.alpha_composite(icon, (int(x + offset - icon.width / 2),
                                         int(y - icon.height / 2)))
            x += sx
        y += sy
        row += 1

    mask = Image.new('L', size, 0)
    ImageDraw.Draw(mask).polygon(outline, fill=SCATTER_OPACITY)
    layer.putalpha(Image.composite(layer.getchannel('A').point(lambda v: v), mask,
                                   mask.point(lambda v: 255 if v else 0)))
    return layer


def render_card(card, px=512, seed=None):
    rng = random.Random(seed)
    scale = px / REF_CARD
    img = Image.new('RGBA', (px, px), (0, 0, 0, 0))
    q = px / 2
    pad = PADDING * scale

    # Two passes: every landmark first, then every symbol. The Illustrator
    # script did one quarter at a time, so a later quarter's shape could cover
    # an earlier quarter's symbols — the shapes are up to 1.7x the quarter and
    # do overlap. That matters here because the annotations would still call
    # those symbols visible, which is exactly the mislabelling this renderer
    # exists to avoid.
    placements = []
    for qi, name in enumerate(QUARTERS):
        symbols = [Symbols.of(x) for x in card['card']['quarters'].get(name, [])]
        n = len(symbols)
        qx, qy = (qi % 2) * q, (qi // 2) * q
        pw = ph = q - 2 * pad
        cx, cy = qx + q / 2, qy + q / 2
        if n:
            pts = roughen(_shape_points(n, cx, cy, pw, ph, rng), scale, rng)
            shape = Image.new('RGBA', img.size, (0, 0, 0, 0))
            ImageDraw.Draw(shape).polygon(pts, fill=LIGHT_BROWN, outline=DARK_BROWN,
                                          width=max(1, int(2 * scale)))
            img.alpha_composite(shape)
            img.alpha_composite(_scatter(img.size, pts, scale, rng))
        spots = {
            1: [(0.50, 0.50)],
            2: [(0.25, 0.25), (0.75, 0.75)],
            3: [(0.25, 0.25), (0.75, 0.25), (0.50, 0.75)],
            4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
        }.get(n, [])
        for symbol, (fx, fy) in zip(symbols, spots):
            placements.append((symbol, qx + pad + pw * fx, qy + pad + ph * fy))

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
