import os
import json
import base64
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from transformers import CLIPProcessor, CLIPModel
from openai import AzureOpenAI
from ultralytics import YOLO
import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from shutil import copyfile
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# ==== Configuration ====
REFERENCE_DIR = config.REFERENCE_DIR
TARGET_DIR = config.TARGET_DIR
AZURE_OPENAI_ENDPOINT = config.AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_KEY = config.AZURE_OPENAI_KEY
AZURE_GPT_DEPLOYMENT = config.AZURE_GPT_DEPLOYMENT
API_VERSION = config.API_VERSION
GOLD_STANDARD_DIR = config.GOLD_STANDARD_DIR
OUTPUT_DIR = config.OUTPUT_DIR
os.makedirs(GOLD_STANDARD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO model for object detection
yolo_model = YOLO(config.YOLO_MODEL_PATH)

# ==== Load CLIP ====
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def load_font(size):
    preferred_font_path = "utils/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(preferred_font_path, size=size)
    except OSError:
        return ImageFont.load_default()

def clip_output_to_vector(output):
    if isinstance(output, torch.Tensor):
        tensor = output
    elif hasattr(output, "pooler_output") and output.pooler_output is not None:
        tensor = output.pooler_output
    elif isinstance(output, (tuple, list)) and output:
        tensor = output[0]
    else:
        raise TypeError(f"Unsupported CLIP output type: {type(output).__name__}")

    return tensor.squeeze().detach().cpu().numpy().astype(np.float32)

def encode_image_to_vector(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return clip_output_to_vector(embedding)

def image_to_base64_str(image, format='PNG'):
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"

def detect_and_crop_views(image_path, model, debug_dir=None, prefix=""):
    image = Image.open(image_path).convert("RGB")
    results = model.predict(image_path, conf=0.7, imgsz=640)[0]

    crops = []
    boxes = []
    base = os.path.splitext(os.path.basename(image_path))[0]

    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    font = load_font(size=40)

    for i, box in enumerate(results.boxes):
        xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        label_text = f"View {i+1} {conf:.2f}"

        crop = image.crop(tuple(xyxy))
        crops.append((i, xyxy, crop, float(box.conf[0])))
        boxes.append(xyxy)

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            crop.save(os.path.join(debug_dir, f"{prefix}{base}_crop_{i+1}_{config.AZURE_GPT_DEPLOYMENT}.png"))

            # Draw bounding box
            draw.rectangle(xyxy, outline="green", width=5)

            # Text and background box size
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Default label position above the box
            label_x = xyxy[0]
            label_y = xyxy[1] - text_height - 8

            # If label would be outside the image, move it inside the box
            if label_y < 0:
                label_y = xyxy[1] + 2

            # Draw background rectangle
            bg_rect = [
                (label_x, label_y),
                (label_x + text_width + 6, label_y + text_height + 6)
            ]
            draw.rectangle(bg_rect, fill="green")

            # Draw text
            draw.text((label_x + 3, label_y + 3), label_text, fill="white", font=font)

    if debug_dir:
        draw_image.save(os.path.join(debug_dir, f"{prefix}{base}_detection_result_{config.AZURE_GPT_DEPLOYMENT}.png"))

    return crops, boxes

def normalize(vec):
    return vec / np.linalg.norm(vec)

def encode_crop(img):
    img = img.resize((224, 224))  # CLIP expects 224x224
    inputs = clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return clip_output_to_vector(embedding)

def match_views_by_clip_hybrid(ref_crops, target_crops, clip_threshold=0.85, iou_threshold=0.25):
    matched = []
    unmatched_ref = []
    unmatched_target = set(range(len(target_crops)))

    ref_vecs = [normalize(encode_crop(r[2])) for r in ref_crops]
    target_vecs = [normalize(encode_crop(t[2])) for t in target_crops]

    # Try to match by CLIP first
    ref_to_target = [np.argmax([np.dot(rv, tv) for tv in target_vecs]) for rv in ref_vecs]
    target_to_ref = [np.argmax([np.dot(tv, rv) for rv in ref_vecs]) for tv in target_vecs]

    already_matched_targets = set()
    for ref_idx, target_idx in enumerate(ref_to_target):
        sim = np.dot(ref_vecs[ref_idx], target_vecs[target_idx])
        iou = compute_iou(ref_crops[ref_idx][1], target_crops[target_idx][1])
        if (target_to_ref[target_idx] == ref_idx) and (sim > clip_threshold or iou > iou_threshold):
            matched.append((ref_idx, ref_crops[ref_idx][2], target_crops[target_idx][2], target_idx))
            unmatched_target.discard(target_idx)
            already_matched_targets.add(target_idx)
        else:
            unmatched_ref.append(ref_idx)

    # Try to match any still-unmatched by IoU only
    for ref_idx in unmatched_ref[:]:
        for tgt_idx in list(unmatched_target):
            iou = compute_iou(ref_crops[ref_idx][1], target_crops[tgt_idx][1])
            if iou > iou_threshold:
                matched.append((ref_idx, ref_crops[ref_idx][2], target_crops[tgt_idx][2], tgt_idx))
                unmatched_target.discard(tgt_idx)
                unmatched_ref.remove(ref_idx)
                break

    unmatched_target = list(unmatched_target)
    return matched, unmatched_ref, unmatched_target

def match_views_by_clip(ref_crops, target_crops, threshold=0.85):
    matched = []
    unmatched_ref = []
    unmatched_target = set(range(len(target_crops)))

    # Precompute normalized CLIP embeddings
    ref_vecs = [normalize(encode_crop(r[2])) for r in ref_crops]
    target_vecs = [normalize(encode_crop(t[2])) for t in target_crops]

    # Forward: best target for each reference
    ref_to_target = [np.argmax([np.dot(rv, tv) for tv in target_vecs]) for rv in ref_vecs]

    # Backward: best reference for each target
    target_to_ref = [np.argmax([np.dot(tv, rv) for rv in ref_vecs]) for tv in target_vecs]

    for ref_idx, target_idx in enumerate(ref_to_target):
        if target_to_ref[target_idx] == ref_idx:
            sim = np.dot(ref_vecs[ref_idx], target_vecs[target_idx])
            if sim > threshold:
                matched.append((ref_idx, ref_crops[ref_idx][2], target_crops[target_idx][2], target_idx))
                unmatched_target.discard(target_idx)
            else:
                unmatched_ref.append(ref_idx)
        else:
            unmatched_ref.append(ref_idx)

    unmatched_target = list(unmatched_target)
    return matched, unmatched_ref, unmatched_target

def save_reference_metadata(image_path, model, output_json_path):
    image = Image.open(image_path).convert("RGB")
    embedding = encode_image_to_vector(image_path)  # for whole-image CLIP matching
    results = model.predict(image_path, conf=0.7, imgsz=640)[0]

    view_count = len(results.boxes)
    data = {
        "ref_image": os.path.basename(image_path),
        "clip_embedding": embedding.astype(np.float64).tolist(),
        "view_count": int(view_count),
        "view_boxes": []
    }

    w, h = image.size
    for box in results.boxes:
        x0, y0, x1, y1 = box.xyxy[0].cpu().numpy()
        norm_box = [
            float(x0 / w),
            float(y0 / h),
            float(x1 / w),
            float(y1 / h),
        ]
        data["view_boxes"].append(norm_box)

    tmp_output_json_path = f"{output_json_path}.tmp"
    with open(tmp_output_json_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_output_json_path, output_json_path)

    log(f"Saved reference metadata to {output_json_path}")

def load_all_reference_json_embeddings(gold_standard_dir):
    embeddings = []
    metadata = []

    for fname in sorted(os.listdir(gold_standard_dir)):
        if fname.endswith(".json"):
            path = os.path.join(gold_standard_dir, fname)
            with open(path, "r") as f:
                data = json.load(f)

            if "clip_embedding" not in data:
                print(f"[!] Skipping {fname}: no embedding found")
                continue

            embedding = np.array(data["clip_embedding"], dtype=np.float32)
            embeddings.append(embedding)
            metadata.append({
                "ref_image": data["ref_image"],
                "json_path": path,
                "clip_embedding": embedding
            })

    if not embeddings:
        raise ValueError("No embeddings loaded from reference JSONs.")

    return np.stack(embeddings), metadata

def match_target_to_reference(target_path, ref_embeddings, ref_metadata):
    target_vec = encode_image_to_vector(target_path)
    distances = np.linalg.norm(ref_embeddings - target_vec, axis=1)
    best_idx = np.argmin(distances)
    return ref_metadata[best_idx], distances[best_idx]

def compute_iou(boxA, boxB):
    # boxA, boxB: [x0, y0, x1, y1] in normalized or absolute coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def bbox_center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)

def center_dist(boxA, boxB):
    cA = bbox_center(boxA)
    cB = bbox_center(boxB)
    return np.sqrt((cA[0]-cB[0])**2 + (cA[1]-cB[1])**2)

def compare_views_with_gpt(matched_crops, unmatched_ref, unmatched_target, ref_crops, target_crops, expected_view_count, actual_target_count):
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

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
            "4. View Count Analysis:\n"
            f"  - Total reference views expected: {expected_view_count}\n"
            f"  - Total target views detected: {actual_target_count}\n"
            "   - Mismatch warnings may apply.\n"
            "5. Conclusion and Recommendation.\n"
            "6. At the end of your report, include a machine-readable summary block in the following format:\n"
            "STRUCTURED_SUMMARY_START\n"
            "{\n"
            "  \"matched_views\": [\n"
            "    {\n"
            "      \"view_index\": 1,\n"
            "      \"missing_annotations\": [\n"
            "        {\"label\": \"Ø140\", \"x\": 0.50, \"y\": 0.07},\n"
            "        {\"label\": \"100.124\", \"x\": 0.18, \"y\": 0.95}\n"
            "      ],\n"
            "      \"extra_annotations\": [\n"
            "        {\"label\": \"Ø100.146\", \"x\": 0.18, \"y\": 0.92},\n"
            "      ]\n"
            "    },\n"
            "    {\n"
            "      \"view_index\": 2,\n"
            "      \"missing_annotations\": []\n"
            "      \"extra_annotations\": []\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "STRUCTURED_SUMMARY_END\n"
            "This block is for machine parsing. Keep it in valid JSON format, enclosed between STRUCTURED_SUMMARY_START and STRUCTURED_SUMMARY_END.\n"
            "Respond in plain text, no markdown formatting."
        )
    }]

    content_list = [
        {"type": "text", "text": (
            "Compare the following matched drawing views:\n"
            "Important: Only describe annotations and dimension labels that are clearly visible in each image. "
            "Do not assume or guess any numbers or text not explicitly shown. If something is unreadable, state that."
        )}
    ]

    # Matched view crops
    for i, ref_img, target_img, target_idx in matched_crops:
        content_list += [
            {"type": "text", "text": f"Matched view #{i+1} (Reference) – please note any annotations and provide their positions in normalized (x, y) form."},
            {"type": "image_url", "image_url": {"url": image_to_base64_str(ref_img)}},
            {"type": "text", "text": f"Matched view #{i+1} (Target) – identify any missing annotations and suggest where to reapply them (use normalized (x, y) coordinates)."},
            {"type": "image_url", "image_url": {"url": image_to_base64_str(target_img)}},
        ]

    # Unmatched reference views
    if unmatched_ref:
        content_list.append({"type": "text", "text": "Unmatched reference views (possibly missing in target):"})
        for idx in unmatched_ref:
            img = ref_crops[idx][2]
            content_list.append({"type": "image_url", "image_url": {"url": image_to_base64_str(img)}})

    # Unmatched target views
    if unmatched_target:
        content_list.append({"type": "text", "text": "Unmatched target views (possibly extra or incorrect):"})
        for idx in unmatched_target:
            img = target_crops[idx][2]
            content_list.append({"type": "image_url", "image_url": {"url": image_to_base64_str(img)}})

    messages.append({"role": "user", "content": content_list})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(target_path))[0]
    
    if config.AZURE_GPT_DEPLOYMENT == "o4-mini" or config.AZURE_GPT_DEPLOYMENT == "gpt-5-mini" or config.AZURE_GPT_DEPLOYMENT == "gpt-5-chat":
        # For o4-mini, we can use reasoning_effort to control the depth of analysis
        # "low" for quick, "medium" for balanced, "high" for deep analysis
        reasoning_effort = "high"

        response = client.chat.completions.create(
            model=AZURE_GPT_DEPLOYMENT,
            messages=messages,
            reasoning_effort=reasoning_effort,
            top_p=1.0                               
        )
        report_path = os.path.join(OUTPUT_DIR, f"{timestamp}_{base}_{config.AZURE_GPT_DEPLOYMENT}_{reasoning_effort}.txt")

    else:
        response = client.chat.completions.create(
            model=AZURE_GPT_DEPLOYMENT,
            messages=messages,
            temperature=0.0,
            top_p=1.0
        )
        report_path = os.path.join(OUTPUT_DIR, f"{timestamp}_{base}_{config.AZURE_GPT_DEPLOYMENT}.txt")

    gpt_feedback = response.choices[0].message.content.strip()

    # print("\n========== View Comparison ==========")
    # print(gpt_feedback)

    with open(report_path, "w") as f:
        f.write(gpt_feedback)

    return report_path, gpt_feedback, timestamp, base

def extract_structured_annotations(gpt_text):
    pattern = r"STRUCTURED_SUMMARY_START(.*?)STRUCTURED_SUMMARY_END"
    match = re.search(pattern, gpt_text, re.DOTALL)
    if not match:
        return None

    json_text = match.group(1).strip()
    try:
        data = json.loads(json_text)
        return data
    except json.JSONDecodeError as e:
        print(f"[!] JSON parsing error: {e}")
        return None

def overlay_circle_with_label(image, norm_x, norm_y, label, radius=50, color="red", show_label=True):
    w, h = image.size
    x = int(norm_x * w)
    y = int(norm_y * h)

    # Create heatmap overlay
    heatmap_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    heat_pixels = heatmap_layer.load()

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            dist = np.sqrt(dx**2 + dy**2)
            if dist <= radius:
                fx = x + dx
                fy = y + dy
                if 0 <= fx < w and 0 <= fy < h:
                    t = 1 - dist / radius  # 1 at center, 0 at edge

                    if color == "red":
                        r = int(255 * t)
                        g = 0
                        b = 0
                    elif color == "orange":
                        r = int(255 * t)
                        g = 165
                        b = 0
                    a = int(255 * t * t)  # quadratic falloff

                    heat_pixels[fx, fy] = (r, g, b, a)

    # Composite heatmap onto original image
    image.paste(heatmap_layer, (0, 0), heatmap_layer)

    if show_label:
        # Draw label
        draw = ImageDraw.Draw(image)
        font = load_font(size=28)

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_pos = (x + radius + 4, y - text_h // 2)
        draw.text(label_pos, label, fill="red", font=font)

    return image

def overlay_annotations_on_full_image(
    full_img,
    view_boxes,
    structured_data,
    view_map,
    ref_view_boxes=None,
    unmatched_ref_idx=None,
    target_crops=None,
    outline="red",
    missing_outline="red"
):
    w_img, h_img = full_img.size

    font = load_font(size=28)

    draw_text = ImageDraw.Draw(full_img)
    radius = 50  # or whatever default

    # === Draw matched missing/extra annotations with heatmap effect ===
    for view in structured_data.get("matched_views", []):
        view_index = view["view_index"]
        target_idx = view_map.get(view_index)
        if target_idx is None or target_idx >= len(view_boxes):
            continue

        x0, y0, x1, y1 = view_boxes[target_idx]
        box_w, box_h = x1 - x0, y1 - y0

        # Overlay missing (red, with label)
        for ann in view.get("missing_annotations", []):
            norm_x, norm_y = ann["x"], ann["y"]
            abs_x = (x0 + norm_x * box_w) / w_img
            abs_y = (y0 + norm_y * box_h) / h_img
            overlay_circle_with_label(
                full_img, abs_x, abs_y, ann["label"], color="red", radius=50, show_label=True
            )

        # Overlay extra (orange, no label)
        for ann in view.get("extra_annotations", []):
            norm_x, norm_y = ann["x"], ann["y"]
            abs_x = (x0 + norm_x * box_w) / w_img
            abs_y = (y0 + norm_y * box_h) / h_img
            overlay_circle_with_label(
                full_img, abs_x, abs_y, ann.get("label", ""), color="orange", radius=100, show_label=False
            )

    # === Draw matched and unmatched target view boxes ===
    if target_crops:
        # Determine matched and unmatched indices
        matched_indices = set(view_map.values()) if view_map else set()
        cropidx_to_gptidx = {v: k for k, v in view_map.items()}

        for i, (idx, xyxy, _, conf) in enumerate(target_crops):
            x0, y0, x1, y1 = xyxy
            if i in cropidx_to_gptidx:
                # Matched: draw green and label as GPT's Matched View #N
                gpt_view_idx = cropidx_to_gptidx[i]
                draw_text.rectangle([x0, y0, x1, y1], outline="green", width=3)
                label = f"Matched View {gpt_view_idx} ({conf:.2f})"
                bg_color = "green"
            else:
                # Unmatched: draw orange
                draw_text.rectangle([x0, y0, x1, y1], outline="orange", width=3)
                label = f"Extra View {i+1} ({conf:.2f})"
                bg_color = "orange"

            text_bbox = draw_text.textbbox((0, 0), label, font=font)
            label_w = text_bbox[2] - text_bbox[0]
            label_h = text_bbox[3] - text_bbox[1]

            label_x = x0
            label_y = y0 - label_h - 6
            if label_y < 0:
                label_y = y0 + 2

            draw_text.rectangle(
                [label_x, label_y, label_x + label_w + 6, label_y + label_h + 6],
                fill=bg_color
            )
            draw_text.text((label_x + 3, label_y + 3), label, fill="white", font=font)

    # === Draw unmatched reference view boxes (from normalized coordinates) ===
    if ref_view_boxes and unmatched_ref_idx:
        for idx in unmatched_ref_idx:
            if idx >= len(ref_view_boxes):
                continue
            norm_box = ref_view_boxes[idx]
            x0 = int(norm_box[0] * w_img)
            y0 = int(norm_box[1] * h_img)
            x1 = int(norm_box[2] * w_img)
            y1 = int(norm_box[3] * h_img)

            draw_text.rectangle([x0, y0, x1, y1], outline=missing_outline, width=5)
            label = f"Missing View {idx+1}"
            text_bbox = draw_text.textbbox((0, 0), label, font=font)
            draw_text.rectangle(
                [x0, y0 - text_bbox[3] - 8, x0 + text_bbox[2] + 6, y0],
                fill=missing_outline
            )
            draw_text.text((x0 + 3, y0 - text_bbox[3] - 4), label, fill="white", font=font)

    return full_img

# ==== Pipeline Entry ====
# Rebuild missing or outdated gold standard JSONs
for fname in sorted(os.listdir(REFERENCE_DIR)):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        ref_path = os.path.join(REFERENCE_DIR, fname)
        json_name = os.path.splitext(fname)[0] + ".json"
        json_path = os.path.join(GOLD_STANDARD_DIR, json_name)

        regenerate = True
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    regenerate = "clip_embedding" not in data or "view_count" not in data
            except Exception:
                pass

        if regenerate:
            log(f"[🔄] Generating gold standard for: {fname}")
            save_reference_metadata(ref_path, yolo_model, json_path)
        else:
            log(f"[✅] Found complete gold standard for: {fname}")

ref_embeddings, ref_metadata = load_all_reference_json_embeddings(GOLD_STANDARD_DIR)

for fname in sorted(os.listdir(TARGET_DIR)):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        target_path = os.path.join(TARGET_DIR, fname)
        log(f"\n==== Processing {fname} ====")

        match_info, score = match_target_to_reference(target_path, ref_embeddings, ref_metadata)
        log(f"Best match for {fname} is {match_info['ref_image']} with distance score {score:.4f}")
        
        # If distance is too large, treat this as a new reference image
        if score > 5.5: # threshold based on empirical testing
            log(f"[⚠️] {fname} does not match any reference closely (score={score:.4f}). Saving as new reference.")

            # 1. Save gold standard JSON
            new_json_name = os.path.splitext(fname)[0] + ".json"
            new_json_path = os.path.join(GOLD_STANDARD_DIR, new_json_name)
            save_reference_metadata(target_path, yolo_model, new_json_path)

            # 2. Copy image into REFERENCE_DIR
            new_ref_path = os.path.join(REFERENCE_DIR, fname)
            if not os.path.exists(new_ref_path):
                copyfile(target_path, new_ref_path)
                log(f"[🖼️] Copied unmatched image to reference directory: {new_ref_path}")

            # 3. Refresh reference memory and skip this iteration
            ref_embeddings, ref_metadata = load_all_reference_json_embeddings(GOLD_STANDARD_DIR)
            log(f"[✅] New reference embedding saved for {fname}. Skipping further comparison.\n")
            continue

        ref_json_path = match_info["json_path"]
        ref_path = os.path.join(REFERENCE_DIR, match_info["ref_image"])

        with open(ref_json_path, "r") as f:
            ref_data = json.load(f)

        ref_crops, _ = detect_and_crop_views(ref_path, yolo_model, debug_dir=os.path.join(OUTPUT_DIR, "debug_crops"), prefix="ref_")
        target_crops, target_boxes = detect_and_crop_views(target_path, yolo_model, debug_dir=os.path.join(OUTPUT_DIR, "debug_crops"), prefix="target_")

        expected_view_count = ref_data["view_count"]
        actual_target_count = len(target_crops)

        matched, unmatched_ref_idx, unmatched_target_idx = match_views_by_clip_hybrid(ref_crops, target_crops)

        full_target_img = Image.open(target_path).convert("RGB")

        suppressed_unmatched_ref_idx = []
        suppression_log = []

        for idx in unmatched_ref_idx:
            ref_box = ref_data["view_boxes"][idx]  # normalized [x0, y0, x1, y1]
            ref_abs = [
                int(ref_box[0] * full_target_img.width),
                int(ref_box[1] * full_target_img.height),
                int(ref_box[2] * full_target_img.width),
                int(ref_box[3] * full_target_img.height),
            ]
            found_similar = False
            for _, tgt_xyxy, _, conf in target_crops:
                iou = compute_iou(ref_abs, tgt_xyxy)
                dist = center_dist(ref_abs, tgt_xyxy)
                if iou > 0.25 or dist < 100:
                    found_similar = True
                    suppression_log.append(
                        f"Suppressing unmatched reference view {idx+1}: overlaps/near target view (IoU={iou:.2f}, dist={dist:.1f})."
                    )
                    break
            if not found_similar:
                suppressed_unmatched_ref_idx.append(idx)

        if suppression_log:
            for line in suppression_log:
                log(line)
        else:
            log("No unmatched reference views suppressed.")

        matched_view_map = {}  # maps GPT's view_index (1-based) → actual target crop index
        for ref_idx, _, _, target_idx in matched:
            matched_view_map[ref_idx + 1] = target_idx

        log(f"Matched view indices: {[m[0]+1 for m in matched]}")
        log(f"Suppressed unmatched reference indices: {[i+1 for i in suppressed_unmatched_ref_idx]}")
        log(f"Unmatched target indices: {[i+1 for i in unmatched_target_idx]}")

        # Generate GPT-based comparison report
        report_path, gpt_output, timestamp, base = compare_views_with_gpt(
            matched,
            suppressed_unmatched_ref_idx,
            [i for i in unmatched_target_idx],
            ref_crops,
            target_crops,
            ref_data["view_count"],
            len(target_crops)
        )

        summary = extract_structured_annotations(gpt_output)
        
        if summary and "matched_views" in summary:
            for view in summary["matched_views"]:
                view_num = view["view_index"]
                target_idx = matched_view_map.get(view_num)
                if target_idx is None or target_idx >= len(target_crops):
                    log(f"[!] Skipping view {view_num}: target index not found.")
                    continue

                target_img = target_crops[target_idx][2].copy()

                for ann in view.get("missing_annotations", []):
                    overlay_circle_with_label(target_img, ann["x"], ann["y"], ann["label"], color="red", radius=50, show_label=True)
                for ann in view.get("extra_annotations", []):  # Suppose your GPT outputs this
                    overlay_circle_with_label(target_img, ann["x"], ann["y"], ann["label"], color="orange", radius=100, show_label=False)

                highlight_name = f"{timestamp}_{base}_highlighted_view_{view_num}_{config.AZURE_GPT_DEPLOYMENT}.png"
                target_img.save(os.path.join(OUTPUT_DIR, highlight_name))

            log(f"Overlayed annotations for {len(summary['matched_views'])} views.")

            ref_view_boxes = ref_data.get("view_boxes", [])
            highlighted_full = overlay_annotations_on_full_image(
                full_img=full_target_img,
                view_boxes=target_boxes,
                structured_data=summary,
                view_map=matched_view_map,
                ref_view_boxes=ref_view_boxes,
                unmatched_ref_idx=suppressed_unmatched_ref_idx,
                target_crops=target_crops
            )
            full_output_path = os.path.join(OUTPUT_DIR, f"{timestamp}_{base}_FULL_HIGHLIGHTED_{config.AZURE_GPT_DEPLOYMENT}.png")
            highlighted_full.save(full_output_path)
            log(f"Saved full-image highlight to {full_output_path}")

        else:
            log("No structured annotations found in GPT output.")
