import os
import base64
from datetime import datetime
from PIL import Image
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from openai import AzureOpenAI

# === CONFIGURATION ===
REFERENCE_DIR = config.REFERENCE_DIR
TARGET_DIR = config.TARGET_DIR
OUTPUT_DIR = config.WITHOUT_SEGMENTATION_OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

AZURE_OPENAI_ENDPOINT = config.AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_KEY = config.AZURE_OPENAI_KEY
AZURE_GPT_DEPLOYMENT = config.AZURE_GPT_DEPLOYMENT
API_VERSION = config.API_VERSION

# === UTILS ===
def image_to_base64(image_path):
    with Image.open(image_path).convert("RGB") as img:
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

def compare_full_images(ref_path, target_path, out_path, client):
    messages = [{
        "role": "system",
        "content": (
            "You are a mechanical engineering drawing reviewer comparing individual views between a reference drawing and a target drawing.\n"
            "Each view is a drawing containing dimensional annotations and geometric shapes.\n"
            "- For each matched view pair, extract and compare annotations in the following order:\n"
            "  1. Vertical dimensions (e.g., height, left/right side measurements)\n"
            "  2. Horizontal dimensions (e.g., width, hole spacing, bottom/top edge distances)\n"
            "  3. Callouts and notes (e.g., 'Ø16 THRU, 4 PL', '2X M24x3.0 TAPPED HOLE', 'R25')\n"
            "- Do not guess, infer, or hallucinate any values that are not explicitly shown. If a label is partially obscured or unclear, mention that instead of assuming its value.\n"
            "- Pay particular attention to dimensional annotations such as numbers, especially those near the edges of the view which are often missed (e.g., small vertical dimensions).\n"
            "- For each missing annotation in the target view (present in the reference), extract its (x, y) coordinates in the reference view using normalized coordinates (top-left = (0.0, 0.0), bottom-right = (1.0, 1.0)).\n"
            "- For each extra annotation in the target view (present in the target but not in the reference), extract its label and its (x, y) coordinates in the target view using normalized coordinates (top-left = (0.0, 0.0), bottom-right = (1.0, 1.0)).\n"
            "- Suggest the same coordinates to reapply the missing label in the target view, unless obstructed.\n"
            "Then, list any unmatched reference or target views:\n"
            "- If a reference view is unmatched, treat it as potentially missing in the target.\n"
            "- If a target view is unmatched, treat it as potentially extra or misplaced.\n\n"
            "Structure the report as follows:\n"
            "1. Title: 'Drawing Comparison Report'\n"
            "2. For each matched view, include:\n"
            "   - 'Matched View #N:'\n"
            "   - A brief bullet list of differences or 'No differences observed.'\n"
            "3. Unmatched Views Section:\n"
            "   - 'Unmatched Reference Views:' with their images and note they were not found in the target.\n"
            "   - 'Unmatched Target Views:' with their images and note they were not present in the reference.\n"
            "4. View Count Heuristic:\n"
            "   - Comment on the visual complexity or approximate number of views if noticeable.\n"
            "5. Conclusion and Recommendation.\n"
            "6. At the end of your report, include a machine-readable summary block in the following format:\n"
            "Respond in plain text, no markdown formatting."
        )
    }]

    content_list = [
        {"type": "text", "text": (
            "Compare the following full engineering drawings. "
            "Only describe annotation and dimension differences that are clearly visible in each image. "
            "Do not guess or infer missing values. If text is unreadable, state that."
        )},
        {"type": "text", "text": "Reference Drawing:"},
        {"type": "image_url", "image_url": {"url": image_to_base64(ref_path)}},
        {"type": "text", "text": "Target Drawing:"},
        {"type": "image_url", "image_url": {"url": image_to_base64(target_path)}}
    ]

    reasoning_effort = "high"
    messages.append({"role": "user", "content": content_list})

    response = client.chat.completions.create(
        model=AZURE_GPT_DEPLOYMENT,
        messages=messages,
        reasoning_effort=reasoning_effort,
        top_p=1.0,
    )

    result = response.choices[0].message.content.strip()
    with open(out_path, "w") as f:
        f.write(result)
    print(f"✅ Saved: {out_path}")

# === MAIN LOOP ===
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

ref_images = sorted(f for f in os.listdir(REFERENCE_DIR) if f.endswith((".png", ".jpg", ".jpeg")))
target_images = sorted(f for f in os.listdir(TARGET_DIR) if f.endswith((".png", ".jpg", ".jpeg")))

# Assuming filenames are aligned
for ref, target in zip(ref_images, target_images):
    ref_path = os.path.join(REFERENCE_DIR, ref)
    target_path = os.path.join(TARGET_DIR, target)
    out_path = os.path.join(OUTPUT_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{target[:-4]}_without_segmentation.txt")
    compare_full_images(ref_path, target_path, out_path, client)
