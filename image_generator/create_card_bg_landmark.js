/*
 * json2.js - Included for compatibility with older versions of Illustrator.
 */
if (typeof JSON !== 'object') {
    JSON = {};
}
(function() {
    'use strict';
    function f(n) { return n < 10 ? '0' + n : n; }
    if (typeof Date.prototype.toJSON !== 'function') {
        Date.prototype.toJSON = function() {
            return isFinite(this.valueOf()) ? this.getUTCFullYear() + '-' +
                f(this.getUTCMonth() + 1) + '-' + f(this.getUTCDate()) + 'T' + f(this.getUTCHours()) + ':' + f(this.getUTCMinutes()) + ':' + f(this.getUTCSeconds()) + 'Z' : null;
        };
        String.prototype.toJSON = Number.prototype.toJSON = Boolean.prototype.toJSON = function() { return this.valueOf(); };
    }
    var cx, escapable, gap, indent, meta, rep;
    function quote(string) {
        escapable.lastIndex = 0;
        return escapable.test(string) ? '"' + string.replace(escapable, function(a) {
            var c = meta[a];
            return typeof c === 'string' ? c : '\\u' + ('0000' + a.charCodeAt(0).toString(16)).slice(-4);
        }) + '"' : '"' + string + '"';
    }
    function str(key, holder) {
        var i, k, v, length, mind = gap, partial, value = holder[key];
        if (value && typeof value === 'object' && typeof value.toJSON === 'function') { value = value.toJSON(key); }
        if (typeof rep === 'function') { value = rep.call(holder, key, value); }
        switch (typeof value) {
            case 'string': return quote(value);
            case 'number': return isFinite(value) ? String(value) : 'null';
            case 'boolean':
            case 'null': return String(value);
            case 'object':
                if (!value) { return 'null'; }
                gap += indent;
                partial = [];
                if (Object.prototype.toString.apply(value) === '[object Array]') {
                    length = value.length;
                    for (i = 0; i < length; i += 1) { partial[i] = str(i, value) || 'null'; }
                    v = partial.length === 0 ? '[]' : gap ? '[\n' + gap + partial.join(',\n' + gap) + '\n' + mind + ']' : '[' + partial.join(',') + ']';
                    gap = mind; return v;
                }
                if (rep && typeof rep === 'object') {
                    length = rep.length;
                    for (i = 0; i < length; i += 1) {
                        if (typeof rep[i] === 'string') {
                            k = rep[i]; v = str(k, value);
                            if (v) { partial.push(quote(k) + (gap ? ': ' : ':') + v); }
                        }
                    }
                } else {
                    for (k in value) {
                        if (Object.prototype.hasOwnProperty.call(value, k)) {
                            v = str(k, value);
                            if (v) { partial.push(quote(k) + (gap ? ': ' : ':') + v); }
                        }
                    }
                }
                v = partial.length === 0 ? '{}' : gap ? '{\n' + gap + partial.join(',\n' + gap) + '\n' + mind + '}' : '{' + partial.join(',') + '}';
                gap = mind; return v;
        }
    }
    if (typeof JSON.stringify !== 'function') {
        escapable = /[\\\"\x00-\x1f\x7f-\x9f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u2028\u2060-\u206f\ufeff\ufff0-\uffff]/g;
        meta = { '\b': '\\b', '\t': '\\t', '\n': '\\n', '\f': '\\f', '\r': '\\r', '"': '\\"', '\\': '\\\\' };
        JSON.stringify = function(value, replacer, space) {
            var i; gap = ''; indent = '';
            if (typeof space === 'number') { for (i = 0; i < space; i += 0) { indent += ' '; } }
            else if (typeof space === 'string') { indent = space; }
            rep = replacer;
            return str('', { '': value });
        };
    }
    if (typeof JSON.parse !== 'function') {
        cx = /[\u0000\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g;
        JSON.parse = function(text, reviver) {
            var j;
            function walk(holder, key) {
                var k, v, value = holder[key];
                if (value && typeof value === 'object') {
                    for (k in value) {
                        if (Object.prototype.hasOwnProperty.call(value, k)) {
                            v = walk(value, k);
                            if (v !== undefined) { value[k] = v; } else { delete value[k]; }
                        }
                    }
                }
                return reviver.call(holder, key, value);
            }
            text = String(text);
            cx.lastIndex = 0;
            if (cx.test(text)) { text = text.replace(cx, function(a) { return '\\u' + ('0000' + a.charCodeAt(0).toString(16)).slice(-4); }); }
            if (/^[\],:{}\s]*$/.test(text.replace(/\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g, '@').replace(/"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, ']').replace(/(?:^|:|,)(?:\s*\[)+/g, ''))) {
                j = eval('(' + text + ')');
                return typeof reviver === 'function' ? walk({ '': j }, '') : j;
            }
            throw new SyntaxError('JSON.parse');
        };
    }
}());

if (!Array.isArray) {
    Array.isArray = function(arg) { return Object.prototype.toString.call(arg) === '[object Array]'; };
}

// Global Colors
var lightBrown = new CMYKColor();
lightBrown.cyan = 30; lightBrown.magenta = 50; lightBrown.yellow = 70; lightBrown.black = 10;
var darkBrown = new CMYKColor();
darkBrown.cyan = 50; darkBrown.magenta = 70; darkBrown.yellow = 90; darkBrown.black = 50;

/**
 * Fills the island with a diagonal pattern of a chosen environmental symbol.
 */
function fillWithTreePattern(doc, layer, baseShape) {
    var bounds = baseShape.visibleBounds;
    var left = bounds[0];
    var top = bounds[1];
    var right = bounds[2];
    var bottom = bounds[3];

    var possibleSymbols = ["Tree", "Pine", "Grass", "Palm"];
    var randomIndex = Math.floor(Math.random() * possibleSymbols.length);
    var symbolName = possibleSymbols[randomIndex];

    try {
        var asset = doc.symbols.getByName(symbolName);
        var group = layer.groupItems.add();
        
        // Pattern size and spacing adjusted to 70%
        var patternScaleFactor = 0.7;
        
        // Increased scarcity by doubling the default spacing
        var scarcityMultiplier = 1.8; 
        var spacingX = (25 * patternScaleFactor) * scarcityMultiplier; 
        var spacingY = (22 * patternScaleFactor) * scarcityMultiplier;
        var targetIconSize = 15 * patternScaleFactor; 

        for (var x = left - spacingX; x < right + spacingX; x += spacingX) {
            for (var y = bottom - spacingY; y < top + spacingY; y += spacingY) {
                var xOffset = (Math.floor((y - bottom) / spacingY) % 2 === 0) ? 0 : spacingX / 2;
                var inst = doc.symbolItems.add(asset);
                inst.move(group, ElementPlacement.PLACEATEND);
                
                var currentMax = Math.max(inst.width, inst.height);
                var scaleRatio = (targetIconSize / currentMax) * 100;
                inst.resize(scaleRatio, scaleRatio);
                inst.opacity = 50; 
                inst.position = [x + xOffset - (inst.width / 2), y + (inst.height / 2)];
            }
        }

        var mask = baseShape.duplicate(group, ElementPlacement.PLACEATBEGINNING);
        group.clipped = true;
        return group; 
    } catch (e) {
        return null;
    }
}

/**
 * Utility to get a jittered value.
 */
function jitter(val, maxPercent) {
    var amt = (Math.random() * 2 - 1) * (maxPercent / 100);
    return val * (1 + amt);
}

function drawQuarterSymbolShape(doc, layer, numSymbols, points, paddedWidth, paddedHeight, config) {
    var shape;
    var targetStyleName = "CardBG"; 
    
    var dimRandomness = 10; 
    // Reverted island base scale factor back to 1.0 (Original Size)
    var baseScaleFactor = 1.0;

    switch (numSymbols) {
        case 4:
            var rectW = jitter((points.top_right[0] - points.top_left[0]) * config.squareScale * baseScaleFactor, dimRandomness);
            var rectH = jitter((points.top_left[1] - points.bottom_left[1]) * config.squareScale * baseScaleFactor, dimRandomness);
            shape = layer.pathItems.rectangle(points.center[1] + rectH/2, points.center[0] - rectW/2, rectW, rectH);
            break;
        case 3:
            targetStyleName = "QBGT";
            var triRadius = jitter((paddedWidth * 0.5) * config.triangleScale * baseScaleFactor, dimRandomness);
            var triangleH = jitter((points.top_left[1] - points.bottom_left[1]) * config.triangleScale * baseScaleFactor, dimRandomness);
            shape = layer.pathItems.polygon(points.center[0], points.center[1]-triangleH/2.5, triRadius, 3);
            shape.rotate(180);
            break;
        case 2:
            shape = layer.pathItems.add();
            var w = jitter(paddedWidth * config.hexagonScale * baseScaleFactor, dimRandomness);
            var h = jitter(paddedHeight * config.hexagonScale * baseScaleFactor, dimRandomness);
            var cX = points.center[0];
            var cY = points.center[1];
            shape.setEntirePath([[cX-w/2,cY],[cX-w/2,cY+h/2],[cX,cY+h/2],[cX+w/2,cY],[cX+w/2,cY-h/2],[cX,cY-h/2]]);
            shape.closed = true;
            break;
        case 1:
            var sizeW = jitter(paddedWidth * config.singleSquareScale * baseScaleFactor, dimRandomness);
            var sizeH = jitter(paddedWidth * config.singleSquareScale * baseScaleFactor, dimRandomness);
            shape = layer.pathItems.rectangle(points.center[1] + sizeH/2, points.center[0] - sizeW/2, sizeW, sizeH);
            break;
        default: return null;
    }

    if (shape) {
        shape.filled = true;
        shape.fillColor = lightBrown;
        shape.stroked = true;
        shape.strokeColor = darkBrown;
        shape.strokeWidth = 2;
        try {
            var style = doc.graphicStyles.getByName(targetStyleName);
            style.applyTo(shape);
        } catch (e) {
            try { var fallback = doc.graphicStyles.getByName("CardBG"); fallback.applyTo(shape); } catch(err){}
        }
        fillWithTreePattern(doc, layer, shape);
    }
    return shape;
}

function placeSymbolsInQuarter(doc, layer, symbolsArray, bounds, targetWidth, targetHeight) {
    var config = { symbolScale: 0.8, squareScale: 2, triangleScale: 1.7, hexagonScale: 1.0, singleSquareScale: 0.6 };
    var padding = 8;
    var numSymbols = symbolsArray.length;
    
    var pX = bounds.x + padding, pY = bounds.y - padding, pW = bounds.width - (2*padding), pH = bounds.height - (2*padding);
    var pts = {
        center: [pX + pW*0.5, pY - pH*0.5],
        top_left: [pX + pW*0.25, pY - pH*0.25],
        top_right: [pX + pW*0.75, pY - pH*0.25],
        bottom_left: [pX + pW*0.25, pY - pH*0.75],
        bottom_right: [pX + pW*0.75, pY - pH*0.75],
        centerBottom: [pX + pW*0.5, pY - pH*0.75]
    };

    drawQuarterSymbolShape(doc, layer, numSymbols, pts, pW, pH, config);

    if (numSymbols === 0) return;

    var pos = [];
    if (numSymbols === 1) pos.push(pts.center);
    if (numSymbols === 2) { pos.push(pts.top_left); pos.push(pts.bottom_right); }
    if (numSymbols === 3) { pos.push(pts.top_left); pos.push(pts.top_right); pos.push(pts.centerBottom); }
    if (numSymbols === 4) { pos.push(pts.top_left); pos.push(pts.top_right); pos.push(pts.bottom_left); pos.push(pts.bottom_right); }

    var finalW = targetWidth * config.symbolScale;
    var finalH = targetHeight * config.symbolScale;

    for (var j = 0; j < numSymbols; j++) {
        try {
            var asset = doc.symbols.getByName(symbolsArray[j]);
            var inst = doc.symbolItems.add(asset);
            var scale = Math.min(finalW / inst.width, finalH / inst.height);
            inst.resize(scale * 100, scale * 100);
            inst.position = [pos[j][0] - (inst.width / 2), pos[j][1] + (inst.height / 2)];
        } catch (e) {}
    }
}

function createCards() {
    var jsonFile = File.openDialog("Select card JSON", "*.json");
    if (!jsonFile) return;
    jsonFile.open('r');
    var jsonStr = jsonFile.read();
    jsonFile.close();
    var configs;
    try { configs = JSON.parse(jsonStr); } catch (e) { alert("Invalid JSON"); return; }
    var cards = Array.isArray(configs) ? configs : [configs];
    var doc = app.activeDocument;
    var perRow = 4, hGap = 0, vGap = 0, col = 0, yOff = 0;

    for (var i = 0; i < cards.length; i++) {
        var data = cards[i];
        var w = data.card.dimensions.width, h = data.card.dimensions.height;
        var rect = [col * (w + hGap), yOff, col * (w + hGap) + w, yOff - h];
        var ab = (i === 0 && doc.artboards.length > 0) ? doc.artboards[0] : doc.artboards.add(rect);
        if (i === 0) ab.artboardRect = rect;
        var layer = doc.layers.add();
        layer.name = "Card " + (i + 1);
        var abL = ab.artboardRect[0], abT = ab.artboardRect[1], abW = ab.artboardRect[2] - abL, abH = Math.abs(abT - ab.artboardRect[3]);
        var qW = abW / 2, qH = abH / 2;
        var qs = {
            top_left: { x: abL, y: abT, width: qW, height: qH },
            top_right: { x: abL + qW, y: abT, width: qW, height: qH },
            bottom_left: { x: abL, y: abT - qH, width: qW, height: qH },
            bottom_right: { x: abL + qW, y: abT - qH, width: qW, height: qH }
        };
        var tSize = qW * 0.45;
        for (var qKey in data.card.quarters) {
            if (qs.hasOwnProperty(qKey)) {
                placeSymbolsInQuarter(doc, layer, data.card.quarters[qKey], qs[qKey], tSize, tSize);
            }
        }
        col++;
        if (col >= perRow) { col = 0; yOff -= (h + vGap); }
    }
    alert("Generation Complete");
}

createCards();