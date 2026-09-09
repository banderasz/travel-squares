/*
 * json2.js - Included for compatibility with older versions of Illustrator.
 */
if (typeof JSON !== 'object') {
    JSON = {};
}
(function() {
    'use strict';

    function f(n) {
        return n < 10 ? '0' + n : n;
    }
    if (typeof Date.prototype.toJSON !== 'function') {
        Date.prototype.toJSON = function() {
            return isFinite(this.valueOf()) ? this.getUTCFullYear() + '-' +
                f(this.getUTCMonth() + 1) + '-' + f(this.getUTCDate()) + 'T' + f(this.getUTCHours()) + ':' + f(this.getUTCMinutes()) + ':' + f(this.getUTCSeconds()) + 'Z' : null;
        };
        String.prototype.toJSON = Number.prototype.toJSON = Boolean.prototype.toJSON = function() {
            return this.valueOf();
        };
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
        var i, k, v, length, mind = gap,
            partial, value = holder[key];
        if (value && typeof value === 'object' && typeof value.toJSON === 'function') {
            value = value.toJSON(key);
        }
        if (typeof rep === 'function') {
            value = rep.call(holder, key, value);
        }
        switch (typeof value) {
            case 'string':
                return quote(value);
            case 'number':
                return isFinite(value) ? String(value) : 'null';
            case 'boolean':
            case 'null':
                return String(value);
            case 'object':
                if (!value) {
                    return 'null';
                }
                gap += indent;
                partial = [];
                if (Object.prototype.toString.apply(value) === '[object Array]') {
                    length = value.length;
                    for (i = 0; i < length; i += 1) {
                        partial[i] = str(i, value) || 'null';
                    }
                    v = partial.length === 0 ? '[]' : gap ? '[\n' + gap + partial.join(',\n' + gap) + '\n' + mind + ']' : '[' + partial.join(',') + ']';
                    gap = mind;
                    return v;
                }
                if (rep && typeof rep === 'object') {
                    length = rep.length;
                    for (i = 0; i < length; i += 1) {
                        if (typeof rep[i] === 'string') {
                            k = rep[i];
                            v = str(k, value);
                            if (v) {
                                partial.push(quote(k) + (gap ? ': ' : ':') + v);
                            }
                        }
                    }
                } else {
                    for (k in value) {
                        if (Object.prototype.hasOwnProperty.call(value, k)) {
                            v = str(k, value);
                            if (v) {
                                partial.push(quote(k) + (gap ? ': ' : ':') + v);
                            }
                        }
                    }
                }
                v = partial.length === 0 ? '{}' : gap ? '{\n' + gap + partial.join(',\n' + gap) + '\n' + mind + '}' : '{' + partial.join(',') + '}';
                gap = mind;
                return v;
        }
    }
    if (typeof JSON.stringify !== 'function') {
        escapable = /[\\\"\x00-\x1f\x7f-\x9f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g;
        meta = {
            '\b': '\\b',
            '\t': '\\t',
            '\n': '\\n',
            '\f': '\\f',
            '\r': '\\r',
            '"': '\\"',
            '\\': '\\\\'
        };
        JSON.stringify = function(value, replacer, space) {
            var i;
            gap = '';
            indent = '';
            if (typeof space === 'number') {
                for (i = 0; i < space; i += 1) {
                    indent += ' ';
                }
            } else if (typeof space === 'string') {
                indent = space;
            }
            rep = replacer;
            if (replacer && typeof replacer !== 'function' && (typeof replacer !== 'object' || typeof replacer.length !== 'number')) {
                throw new Error('JSON.stringify');
            }
            return str('', {
                '': value
            });
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
                            if (v !== undefined) {
                                value[k] = v;
                            } else {
                                delete value[k];
                            }
                        }
                    }
                }
                return reviver.call(holder, key, value);
            }
            text = String(text);
            cx.lastIndex = 0;
            if (cx.test(text)) {
                text = text.replace(cx, function(a) {
                    return '\\u' + ('0000' + a.charCodeAt(0).toString(16)).slice(-4);
                });
            }
            if (/^[\],:{}\s]*$/.test(text.replace(/\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g, '@').replace(/"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, ']').replace(/(?:^|:|,)(?:\s*\[)+/g, ''))) {
                j = eval('(' + text + ')');
                return typeof reviver === 'function' ? walk({
                    '': j
                }, '') : j;
            }
            throw new SyntaxError('JSON.parse');
        };
    }
}());

// Custom Array.isArray polyfill for older environments
if (!Array.isArray) {
    Array.isArray = function(arg) {
        return Object.prototype.toString.call(arg) === '[object Array]';
    };
}

// =================================================================================
// MAIN CARD CREATION SCRIPT
// =================================================================================
function createCards() {
    var jsonFile = File.openDialog("Select your card JSON file", "*.json");
    if (!jsonFile) {
        return;
    }

    jsonFile.open('r');
    var jsonString = jsonFile.read();
    jsonFile.close();
    try {
        var cardConfigs = JSON.parse(jsonString);
    } catch (e) {
        alert("Error: The selected file is not a valid JSON file.\n" + e);
        return;
    }

    var cardsToCreate = Array.isArray(cardConfigs) ? cardConfigs : [cardConfigs];

    var doc = app.activeDocument;
    
    // --- Grid Layout Parameters ---
    var artboardsPerRow = 6;
    var horizontalPadding = 0; // Padding between artboards horizontally
    var verticalPadding = 0;   // Padding between artboards vertically
    var currentRow = 0;
    var currentColumn = 0;
    var currentYOffset = 0; // Tracks the top-most Y coordinate for the current row

    // Ensure there's at least one artboard to start with if the document is empty
    if (doc.artboards.length === 0) {
        doc.artboards.add([0, 100, 100, 0]); // Add a small default artboard if none exists
        doc.artboards.setActiveArtboardIndex(0); // Set the newly created artboard as active
    } else {
        // If there are existing artboards, make sure the first one is active initially
        doc.artboards.setActiveArtboardIndex(0);
    }

    for (var cardIndex = 0; cardIndex < cardsToCreate.length; cardIndex++) {
        var cardData = cardsToCreate[cardIndex];

        var cardWidth = cardData.card.dimensions.width;
        var cardHeight = cardData.card.dimensions.height;

        var currentArtboard;

        // Calculate position for the current artboard
        var artboardLeft = currentColumn * (cardWidth + horizontalPadding);
        var artboardTop = currentYOffset; // Top of the current row

        // Adjust Y for Illustrator's inverted Y-axis (top-left is [X, Y], Y decreases downwards)
        var artboardRect = [artboardLeft, artboardTop + cardHeight, artboardLeft + cardWidth, artboardTop];

        if (cardIndex === 0) {
            // For the first card, use the first existing artboard and resize it
            currentArtboard = doc.artboards[0];
            currentArtboard.artboardRect = artboardRect;
        } else {
            // For subsequent cards, add a new artboard
            currentArtboard = doc.artboards.add(artboardRect);
        }

        try {
            doc.artboards.setActiveArtboardIndex(currentArtboard.index);
        } catch (e) {
            app.redraw();
        }

        // Update column and row for the next artboard
        currentColumn++;
        if (currentColumn >= artboardsPerRow) {
            currentColumn = 0; // Reset to first column
            currentRow++;       // Move to the next row
            currentYOffset -= (cardHeight + verticalPadding); // Adjust Y for the new row (Y decreases)
        }


        var cardLayer = doc.layers.add();
        cardLayer.name = "Generated Card " + (cardIndex + 1) + " - " + new Date().toLocaleTimeString();
        doc.activeLayer = cardLayer; // Make this the active layer for symbol placement

        // Use the actual artboard dimensions for drawing elements
        var actualArtboardLeft = currentArtboard.artboardRect[0];
        var actualArtboardTop = currentArtboard.artboardRect[1];
        var actualCardWidth = currentArtboard.artboardRect[2] - currentArtboard.artboardRect[0];
        var actualCardHeight = currentArtboard.artboardRect[1] - currentArtboard.artboardRect[3];

        var blackColor = new CMYKColor();
        blackColor.cyan = 0;
        blackColor.magenta = 0;
        blackColor.yellow = 0;
        blackColor.black = 100;

        // --- Add 3 pt black border ---
        var borderRect = cardLayer.pathItems.rectangle(actualArtboardTop, actualArtboardLeft, actualCardWidth, actualCardHeight);
        borderRect.stroked = true;
        borderRect.filled = false;
        borderRect.strokeColor = blackColor;
        borderRect.strokeWidth = 8;

        // --- Add 1 pt black horizontal and vertical crosslines ---
        var horizontalLine = cardLayer.pathItems.add();
        horizontalLine.setEntirePath([
            [actualArtboardLeft, actualArtboardTop - actualCardHeight / 2],
            [actualArtboardLeft + actualCardWidth, actualArtboardTop - actualCardHeight / 2]
        ]);
        horizontalLine.stroked = true;
        horizontalLine.filled = false;
        horizontalLine.strokeColor = blackColor;
        horizontalLine.strokeWidth = 4;

        var verticalLine = cardLayer.pathItems.add();
        verticalLine.setEntirePath([
            [actualArtboardLeft + actualCardWidth / 2, actualArtboardTop],
            [actualArtboardLeft + actualCardWidth / 2, actualArtboardTop - actualCardHeight]
        ]);
        verticalLine.stroked = true;
        verticalLine.filled = false;
        verticalLine.strokeColor = blackColor;
        verticalLine.strokeWidth = 4;

        var quarterWidth = actualCardWidth / 2;
        var quarterHeight = actualCardHeight / 2;

        var quarters = {
            top_left: {
                x: actualArtboardLeft,
                y: actualArtboardTop,
                width: quarterWidth,
                height: quarterHeight
            },
            top_right: {
                x: actualArtboardLeft + quarterWidth,
                y: actualArtboardTop,
                width: quarterWidth,
                height: quarterHeight
            },
            bottom_left: {
                x: actualArtboardLeft,
                y: actualArtboardTop - quarterHeight, // Y decreases downwards in Illustrator coordinates
                width: quarterWidth,
                height: quarterHeight
            },
            bottom_right: {
                x: actualArtboardLeft + quarterWidth,
                y: actualArtboardTop - quarterHeight,
                width: quarterWidth,
                height: quarterHeight
            }
        };

        // Calculate a global target size for symbols based on a 4-symbol arrangement
        var padding = 4;
        var globalTargetWidth = (quarterWidth - 3 * padding) / 2;
        var globalTargetHeight = (quarterHeight - 3 * padding) / 2;

        if (globalTargetWidth <= 0 || globalTargetHeight <= 0) {
            alert("Warning: Padding is too large for the quarter size for card " + (cardIndex + 1) + ". Symbols might not be placed correctly.");
        }

        for (var quarterName in cardData.card.quarters) {
            if (quarters.hasOwnProperty(quarterName)) {
                var symbolsToPlace = cardData.card.quarters[quarterName];
                var quarterBounds = quarters[quarterName];
                placeSymbolsInQuarter(doc, cardLayer, symbolsToPlace, quarterBounds, globalTargetWidth, globalTargetHeight);
            }
        }
    }

    alert("Cards created successfully!");
}

/**
 * Resizes and places symbols in a deterministic grid based on the count,
 * using a fixed symbol size.
 */
function placeSymbolsInQuarter(doc, layer, symbolsArray, bounds, fixedTargetWidth, fixedTargetHeight) {
    var padding = 4; // The space from the outer edge of the quarter
    var numSymbols = symbolsArray.length;

    if (numSymbols === 0) {
        return; // Do nothing if the quarter is empty
    }

    // --- 1. DEFINE ANCHOR POINTS FOR PLACEMENT ---
    // These points are relative to the *document's* top-left corner.
    var paddedX = bounds.x + padding;
    var paddedY = bounds.y - padding; // Illustrator Y-coordinates decrease downwards

    var paddedWidth = bounds.width - (2 * padding);
    var paddedHeight = bounds.height - (2 * padding);

    // Define the center-points for a 2x2 grid within the padded area
    var points = {
        center: [paddedX + paddedWidth * 0.50, paddedY - paddedHeight * 0.50],
        top_left: [paddedX + paddedWidth * 0.25, paddedY - paddedHeight * 0.25],
        top_right: [paddedX + paddedWidth * 0.75, paddedY - paddedHeight * 0.25],
        bottom_left: [paddedX + paddedWidth * 0.25, paddedY - paddedHeight * 0.75],
        bottom_right: [paddedX + paddedWidth * 0.75, paddedY - paddedHeight * 0.75],
        centerBottom: [paddedX + paddedWidth * 0.50, paddedY - paddedHeight * 0.75] // Special point for the 3-symbol case
    };

    // --- 2. DETERMINE POSITIONS BASED ON SYMBOL COUNT ---
    var positions = [];
    switch (numSymbols) {
        case 1:
            positions.push(points.center);
            break;
        case 2:
            positions.push(points.top_left);
            positions.push(points.bottom_right);
            break;
        case 3:
            positions.push(points.top_left);
            positions.push(points.top_right);
            positions.push(points.centerBottom);
            break;
        case 4:
            positions.push(points.top_left);
            positions.push(points.top_right);
            positions.push(points.bottom_left);
            positions.push(points.bottom_right);
            break;
        default: // Fallback for 5+ symbols: place randomly within the padded area
            for (var j = 0; j < numSymbols; j++) {
                var randomX = paddedX + Math.random() * paddedWidth;
                var randomY = paddedY - Math.random() * paddedHeight;
                positions.push([randomX, randomY]);
            }
            break;
    }

    // --- 3. CREATE, RESIZE, AND PLACE EACH SYMBOL ---
    for (var i = 0; i < numSymbols; i++) {
        try {
            var symbolAsset = doc.symbols.getByName(symbolsArray[i]);
            var symbolInstance = doc.symbolItems.add(symbolAsset);
            symbolInstance.layer = layer;

            var originalWidth = symbolInstance.width;
            var originalHeight = symbolInstance.height;

            // Use the fixed target size for scaling, which is derived from the 4-symbol case
            var scaleX = fixedTargetWidth / originalWidth;
            var scaleY = fixedTargetHeight / originalHeight;
            var scaleFactor = Math.min(scaleX, scaleY); // Use smaller scale to maintain aspect ratio

            symbolInstance.resize(scaleFactor * 100, scaleFactor * 100); // Resize takes percentage

            // Position the symbol. A symbol's `position` is its top-left corner.
            // We must offset it so its center lands on our calculated anchor point.
            var centerPoint = positions[i];
            var newX = centerPoint[0] - (symbolInstance.width / 2);
            var newY = centerPoint[1] + (symbolInstance.height / 2); // Add to Y because Y is inverted
            symbolInstance.position = [newX, newY];

        } catch (e) {
            // alert("Warning: Symbol '" + symbolsArray[i] + "' was not found in your Symbols panel. Please ensure it exists and try again.");
        }
    }
}

// Run the main function
createCards();