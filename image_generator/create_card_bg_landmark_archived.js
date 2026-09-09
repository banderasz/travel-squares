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
        escapable = /[\\\"\x00-\x1f\x7f-\x9f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g;
        meta = { '\b': '\\b', '\t': '\\t', '\n': '\\n', '\f': '\\f', '\r': '\\r', '"': '\\"', '\\': '\\\\' };
        JSON.stringify = function(value, replacer, space) {
            var i; gap = ''; indent = '';
            if (typeof space === 'number') { for (i = 0; i < space; i += 1) { indent += ' '; } }
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

// New Blue Color for Curves
var electricBlue = new CMYKColor();
electricBlue.cyan = 80; electricBlue.magenta = 20; electricBlue.yellow = 0; electricBlue.black = 0;

// Configuration for Curves
var curveOpacity = 50;   // Percentage 0-100
var curveWaviness = 0.4; // Waviness factor (default 0.4 represents the previous level)

/**
 * Generates a randomly curving line within a group masked by the background shape.
 * Updated to move in a general direction with adjustable waviness and opacity.
 * Modified handle logic to prevent loops and self-intersection.
 */
function drawMaskedCurvyLine(layer, baseShape) {
    var group = layer.groupItems.add();
    
    var line = group.pathItems.add();
    line.stroked = true;
    line.filled = false;
    line.strokeColor = electricBlue;
    line.strokeWidth = 2;
    line.strokeCap = StrokeCap.ROUNDENDCAP;
    line.opacity = curveOpacity; 

    var boundsRect = baseShape.visibleBounds;
    var left = boundsRect[0];
    var top = boundsRect[1];
    var right = boundsRect[2];
    var bottom = boundsRect[3];
    var width = right - left;
    var height = top - bottom;

    var isHorizontal = Math.random() > 0.5;
    var pointsCount = 6 + Math.floor(Math.random() * 3); 
    
    for (var i = 0; i < pointsCount; i++) {
        var newPoint = line.pathPoints.add();
        var progress = i / (pointsCount - 1); 
        
        var anchorX, anchorY;
        var jitter = (Math.random() * 0.1) - 0.05;
        var p = Math.max(0, Math.min(1, progress + jitter));

        if (isHorizontal) {
            anchorX = left + (p * width);
            anchorY = bottom + (Math.random() * height);
        } else {
            anchorX = left + (Math.random() * width);
            anchorY = bottom + (p * height);
        }
        
        newPoint.anchor = [anchorX, anchorY];
        
        // To avoid loops, we ensure the handle size never exceeds the distance between points
        // and that the handles are restricted primarily to the direction of flow.
        var baseHandleSize = Math.min(width, height) / (pointsCount / 2);
        var handleSize = baseHandleSize * (0.5 + Math.random() * 0.5);
        
        if (isHorizontal) {
            // Constrain yOffset to be smaller relative to handleSize to avoid high-curvature loops
            var yHandleOffset = (Math.random() * height * curveWaviness * 0.5) - (height * (curveWaviness * 0.25));
            // Keep handles mostly horizontal to ensure flow
            newPoint.leftDirection  = [anchorX - handleSize, anchorY - yHandleOffset];
            newPoint.rightDirection = [anchorX + handleSize, anchorY + yHandleOffset];
        } else {
            var xHandleOffset = (Math.random() * width * curveWaviness * 0.5) - (width * (curveWaviness * 0.25));
            // Keep handles mostly vertical to ensure flow
            newPoint.leftDirection  = [anchorX - xHandleOffset, anchorY - handleSize];
            newPoint.rightDirection = [anchorX + xHandleOffset, anchorY + handleSize];
        }
    }
    
    var maskShape = baseShape.duplicate(group, ElementPlacement.PLACEATBEGINNING);
    maskShape.filled = true;
    maskShape.stroked = false;
    group.clipped = true;
}

/**
 * Draws geometric shapes based on the number of symbols
 */
function drawQuarterSymbolShape(doc, layer, numSymbols, points, paddedWidth, paddedHeight, config) {
    var shape;
    var targetStyleName = "CardBG"; 

    switch (numSymbols) {
        case 4:
            var rectW = (points.top_right[0] - points.top_left[0]) * config.squareScale;
            var rectH = (points.top_left[1] - points.bottom_left[1]) * config.squareScale;
            shape = layer.pathItems.rectangle(points.center[1] + rectH/2, points.center[0] - rectW/2, rectW, rectH);
            break;
        case 3:
            targetStyleName = "QBGT";
            var triangleH = (points.top_left[1] - points.bottom_left[1]) * config.triangleScale;
            var triRadius = (paddedWidth * 0.5) * config.triangleScale;
            shape = layer.pathItems.polygon(points.center[0], points.center[1]-triangleH/2.5, triRadius, 3);
            shape.rotate(180);
            break;
        case 2:
            shape = layer.pathItems.add();
            var w = paddedWidth * config.hexagonScale;
            var h = paddedHeight * config.hexagonScale;
            var cX = points.center[0];
            var cY = points.center[1];
            shape.setEntirePath([
                [cX - w/2, cY],
                [cX - w/2, cY + h/2],
                [cX, cY + h/2],
                [cX + w/2, cY],
                [cX + w/2, cY - h/2],
                [cX, cY - h/2]
            ]);
            shape.closed = true;
            break;
        case 1:
            var size = paddedWidth * config.singleSquareScale;
            shape = layer.pathItems.rectangle(points.center[1] + size/2, points.center[0] - size/2, size, size);
            break;
        default:
            return null;
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
            if (targetStyleName !== "CardBG") {
                try {
                    var fallbackStyle = doc.graphicStyles.getByName("CardBG");
                    fallbackStyle.applyTo(shape);
                } catch (err) {}
            }
        }

        drawMaskedCurvyLine(layer, shape);
    }
    return shape;
}

/**
 * Places symbols in fixed positions per quarter.
 */
function placeSymbolsInQuarter(doc, layer, symbolsArray, bounds, targetWidth, targetHeight) {
    var config = {
        symbolScale: 0.80,
        squareScale: 2,
        triangleScale: 1.7,
        hexagonScale: 1.00,
        singleSquareScale: 0.60
    };

    var padding = 8;
    var reorderedSymbols = [];
    var startItems = [];
    var endItems = [];
    var middleItems = [];

    for (var i = 0; i < symbolsArray.length; i++) {
        var sName = symbolsArray[i].toLowerCase();
        if (sName.indexOf("arrow") === 0) {
            if (sName.indexOf("up") !== -1 || sName.indexOf("left") !== -1) {
                startItems.push(symbolsArray[i]);
            } else if (sName.indexOf("down") !== -1 || sName.indexOf("right") !== -1) {
                endItems.push(symbolsArray[i]);
            } else {
                middleItems.push(symbolsArray[i]);
            }
        } else {
            middleItems.push(symbolsArray[i]);
        }
    }
    
    reorderedSymbols = startItems.concat(middleItems).concat(endItems);

    var numSymbols = reorderedSymbols.length;
    if (numSymbols === 0) return;

    var pX = bounds.x + padding;
    var pY = bounds.y - padding;
    var pW = bounds.width - (2 * padding);
    var pH = bounds.height - (2 * padding);

    var pts = {
        center: [pX + pW * 0.50, pY - pH * 0.50],
        top_left: [pX + pW * 0.25, pY - pH * 0.25],
        top_right: [pX + pW * 0.75, pY - pH * 0.25],
        bottom_left: [pX + pW * 0.25, pY - pH * 0.75],
        bottom_right: [pX + pW * 0.75, pY - pH * 0.75],
        centerBottom: [pX + pW * 0.50, pY - pH * 0.75]
    };

    drawQuarterSymbolShape(doc, layer, numSymbols, pts, pW, pH, config);

    var pos = [];
    if (numSymbols === 1) pos.push(pts.center);
    if (numSymbols === 2) { pos.push(pts.top_left); pos.push(pts.bottom_right); }
    if (numSymbols === 3) { pos.push(pts.top_left); pos.push(pts.top_right); pos.push(pts.centerBottom); }
    if (numSymbols === 4) { pos.push(pts.top_left); pos.push(pts.top_right); pos.push(pts.bottom_left); pos.push(pts.bottom_right); }

    var finalWidth = targetWidth * config.symbolScale;
    var finalHeight = targetHeight * config.symbolScale;

    for (var j = 0; j < numSymbols; j++) {
        try {
            var asset = doc.symbols.getByName(reorderedSymbols[j]);
            var inst = doc.symbolItems.add(asset);
            var scale = Math.min(finalWidth / inst.width, finalHeight / inst.height);
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
    
    var perRow = 4;
    var hGap = 0;
    var vGap = 0;
    var col = 0;
    var yOff = 0;

    for (var i = 0; i < cards.length; i++) {
        var data = cards[i];
        var w = data.card.dimensions.width;
        var h = data.card.dimensions.height;

        var rect = [col * (w + hGap), yOff, col * (w + hGap) + w, yOff - h];
        var ab = (i === 0 && doc.artboards.length > 0) ? doc.artboards[0] : doc.artboards.add(rect);
        if (i === 0) ab.artboardRect = rect;

        var layer = doc.layers.add();
        layer.name = "Card " + (i + 1);
        
        var abL = ab.artboardRect[0];
        var abT = ab.artboardRect[1];
        var abW = ab.artboardRect[2] - ab.artboardRect[0];
        var abH = Math.abs(ab.artboardRect[1] - ab.artboardRect[3]);

        var qW = abW / 2;
        var qH = abH / 2;
        var qs = {
            top_left:      { x: abL, y: abT, width: qW, height: qH },
            top_right:     { x: abL + qW, y: abT, width: qW, height: qH },
            bottom_left:   { x: abL, y: abT - qH, width: qW, height: qH },
            bottom_right:  { x: abL + qW, y: abT - qH, width: qW, height: qH }
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