import os
import sys

SITENAME = "Travel Squares"
SITEURL = "http://localhost:8000"
AUTHOR_NAME = "András"
TIMEZONE = "Europe/Vienna"

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.symbols import Symbols
JINJA_GLOBALS = {symbol.name: symbol.display for symbol in Symbols}

print(JINJA_GLOBALS)

PATH = 'content'
ARTICLE_EXCLUDES = ['plotly_graphs']

PLUGINS = [
    'jinja2content', # For processing Jinja2 in Markdown
    'render_math',   # For LaTeX math
]