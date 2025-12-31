import clip
import torch
from PIL import Image
from pathlib import Path
import pandas as pd
from datetime import datetime
from collections import defaultdict


DOWNLOAD_FOLDER = Path(input("Enter the path for photos folder: ").strip().strip('"'))
ENGAGEMENT_CSV = Path(input("Enter the path for CSV file: ").strip().strip('"'))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}



CATEGORY_PROMPTS = {
    "casual": [
        "a photo of a person wearing a casual t-shirt and jeans",
        "a casual everyday outfit worn outdoors",
        "a relaxed casual clothing style"
    ],
    "formal": [
        "a photo of a person wearing formal clothing",
        "a professional or business outfit",
        "a formal suit or elegant dress"
    ],
    "streetwear": [
        "a streetwear fashion outfit",
        "urban fashion with hoodie or sneakers",
        "modern street style clothing"
    ],
    "sportswear": [
        "a person wearing sportswear or gym clothes",
        "athletic clothing for exercise",
        "activewear worn during training"
    ],
    "traditional": [
        "traditional south asian clothing",
        "ethnic traditional outfit",
        "cultural ceremonial dress"
    ],
    "party_revealing": [
        "a party outfit or night out fashion",
        "a revealing or glamorous dress",
        "fashion worn for parties or nightlife"
    ]
}

FINE_PROMPTS = {
    "casual": [
        "t-shirt and jeans",
        "polo shirt and chinos",
        "casual button-down shirt",
        "hoodie and joggers"
    ],
    "formal": [
        "business suit with tie",
        "formal evening gown",
        "blazer and dress pants",
        "office wear formal outfit"
    ],
    "streetwear": [
        "oversized hoodie streetwear",
        "bomber jacket outfit",
        "urban fashion with sneakers"
    ],
    "sportswear": [
        "gym tank top and shorts",
        "running outfit",
        "athletic leggings and sports bra"
    ],
    "traditional": [
        "kurta pajama traditional wear",
        "saree traditional dress",
        "lehenga choli festive wear"
    ],
    "party_revealing": [
        "cocktail dress",
        "mini dress party wear",
        "bodycon dress night out"
    ]
}


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_filename_timestamp(path: Path):
    try:
        base = path.stem
        if "_" in base and base.rsplit("_", 1)[-1].isdigit():
            base = base.rsplit("_", 1)[0]
        base = base.replace("_UTC", "")
        return datetime.strptime(base, "%Y-%m-%d_%H-%M-%S")
    except:
        return None


def group_images(images):
    groups = defaultdict(list)
    for img in images:
        ts = parse_filename_timestamp(img)
        if ts:
            key = ts.strftime("%Y-%m-%d_%H-%M-%S")
            groups[key].append(img)
    return groups


def encode_text(model, device, prompts):
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
    return feats


def encode_images(model, preprocess, device, img_paths):
    feats = []
    for p in img_paths:
        img = preprocess(Image.open(p)).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model.encode_image(img)
            f /= f.norm(dim=-1, keepdim=True)
            feats.append(f)
    return torch.mean(torch.stack(feats), dim=0)


def best_match(image_feat, text_feats, labels):
    logits = (image_feat @ text_feats.T)[0].cpu().numpy()
    idx = logits.argmax()
    return labels[idx], float(logits[idx])




def analyze():
    device = get_device()
    print(f"\nUsing device: {device}")

    print("Loading CLIP...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Encode category prompts
    category_labels = list(CATEGORY_PROMPTS.keys())
    category_text = [p for plist in CATEGORY_PROMPTS.values() for p in plist]
    category_map = []
    for k, plist in CATEGORY_PROMPTS.items():
        category_map += [k] * len(plist)

    category_feats = encode_text(model, device, category_text)

    # Load CSV
    df = pd.read_csv(ENGAGEMENT_CSV)
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df["likes"] = df.get("likes", 0).clip(lower=0)
    df["comments"] = df.get("comments", 0).clip(lower=0)
    df["engagement"] = df["likes"] + 4 * df["comments"]

    # Load images
    images = [p for p in DOWNLOAD_FOLDER.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    groups = group_images(images)

    results = []

    print(f"\nAnalyzing {len(groups)} posts...\n")

    for ts, imgs in groups.items():
        img_feat = encode_images(model, preprocess, device, imgs)

        # Stage 1: category
        cat, cat_score = best_match(img_feat, category_feats, category_map)

        # Stage 2: fine label
        fine_prompts = FINE_PROMPTS[cat]
        fine_feats = encode_text(model, device, fine_prompts)
        fine_label, fine_score = best_match(img_feat, fine_feats, fine_prompts)

        # Match CSV row
        img_dt = datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S")
        df["diff"] = abs((df["date_parsed"] - img_dt).dt.total_seconds())
        row = df.loc[df["diff"].idxmin()]

        if row["diff"] > 86400:
            continue

        results.append({
            "shortcode": row["shortcode"],
            "category": cat,
            "outfit": fine_label,
            "confidence": round(fine_score, 3),
            "likes": int(row["likes"]),
            "comments": int(row["comments"]),
            "engagement": float(row["engagement"]),
            "num_images": len(imgs),
            "date": row["date"]
        })

    

    if not results:
        print("No results found.")
        return

    df_final = pd.DataFrame(results)

    # SORTING on the basis of engagement 
    df_final = df_final.sort_values(
        by=["engagement", "confidence"],
        ascending=[False, False]
    )

    output = "fashion_engagement_analysis_v2.csv"
    df_final.to_csv(output, index=False)

    print("\n" + "=" * 70)
    print(f"Saved results to {output}")
    print("=" * 70)

    print("\nTOP 10 POSTS (HIGH → LOW ENGAGEMENT):\n")
    print(df_final.head(10)[
        ["category", "outfit", "confidence", "likes", "comments", "engagement"]
    ].to_string(index=False))




if __name__ == "__main__":
    analyze()
