"""Replace the `collage of different snake games` placeholder cell with a
frozen code cell that displays the pre-generated snake_collage.png."""
import base64, json, sys, os

REPO = os.path.join(os.path.dirname(__file__), '..')
NB_PATH = os.path.join(REPO, 'chapters', 'prompting.ipynb')
IMG_PATH = os.path.join(REPO, 'images', 'prompting', 'snake_collage.png')
TARGET_ID = '2d13eeb3'

raw = open(NB_PATH, 'rb').read()
nb = json.loads(raw)

# Encode the PNG as base64
with open(IMG_PATH, 'rb') as f:
    png_b64 = base64.b64encode(f.read()).decode('ascii')

code_source = (
    "import matplotlib.pyplot as plt\n"
    "from PIL import Image\n"
    "img = Image.open('../images/prompting/snake_collage.png')\n"
    "fig, ax = plt.subplots(figsize=(10, 7))\n"
    "ax.imshow(img)\n"
    "ax.axis('off')\n"
    "plt.tight_layout(pad=0)\n"
    "plt.show()"
)

new_cell = {
    "id": TARGET_ID,
    "cell_type": "code",
    "execution_count": 1,
    "metadata": {"tags": ["freeze"]},
    "source": code_source,
    "outputs": [
        {
            "output_type": "display_data",
            "metadata": {},
            "data": {
                "image/png": png_b64,
                "text/plain": ["<Figure size 960x672 with 1 Axes>"]
            }
        }
    ]
}

cells = nb['cells']
for i, cell in enumerate(cells):
    if cell.get('id') == TARGET_ID:
        cells[i] = new_cell
        print(f"Replaced cell {TARGET_ID} at index {i}")
        break
else:
    sys.exit(f"Cell {TARGET_ID} not found")

# Preserve encoding style
ensure_ascii = raw.isascii()
serialized = json.dumps(nb, ensure_ascii=ensure_ascii, indent=1)
if raw.endswith(b'\n'):
    serialized += '\n'

with open(NB_PATH, 'w', encoding='utf-8') as f:
    f.write(serialized)
print("Done.")
