import clip
import torch
from PIL import Image
from pathlib import Path
import pandas as pd
from datetime import datetime
from collections import defaultdict
import instaloader

DOWNLOAD_FOLDER = Path(input("Enter the path for photos folder: ").strip().strip('"'))
ENGAGEMENT_CSV = Path(input("Enter the path for CSV file: ").strip().strip('"'))


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

CLOTHING_TYPES = [
    "t-shirt and jeans casual wear",
    "polo shirt and jeans",
    "casual shirt and jeans",
    "shorts and t-shirt summer outfit",
    "hoodie and joggers streetwear",
    "sweater and pants winter outfit",
    "button-up shirt and trousers",
    "formal shirt and dress pants",
    "business suit formal wear",
    "traditional ethnic wear",
    "kurta pajama traditional outfit",
    "saree traditional wear",
    "lehenga choli festive wear",
    "salwar kameez ethnic outfit",
    "athletic wear sportswear",
    "gym workout clothes",
    "cocktail dress party attire",
    "evening gown formal dress",
    "mini short dress",
    "winter coat heavy jacket",
]

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_filename_timestamp(filename):
   
    try:
        base = filename.stem        
        if '_' in base:
            parts = base.rsplit('_', 1)
            if parts[-1].isdigit():
                base = parts[0]  
        
        # Remove _UTC suffix
        base = base.replace('_UTC', '')
        
        # Parse: "2023-12-16_02-01-15"
        dt = datetime.strptime(base, "%Y-%m-%d_%H-%M-%S")
        return dt
    except Exception as e:
        print(f" Could not parse timestamp from {filename}: {e}")
        return None


def group_images_by_timestamp(images):
    groups = defaultdict(list)
    
    for img in images:
        timestamp = parse_filename_timestamp(img)
        if timestamp:
            # Use timestamp as key (without seconds for grouping)
            key = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
            groups[key].append(img)
    
    return groups


def match_images_to_csv(image_groups, df_eng):
    matches = {}
    
    df_eng['date_parsed'] = pd.to_datetime(df_eng['date'], errors='coerce')
    
    print("\n Matching images to posts...")
    print(f"   - Image groups: {len(image_groups)}")
    print(f"   - CSV posts: {len(df_eng)}")
    
    for timestamp_str, img_list in image_groups.items():
        try:
            img_dt = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
        except:
            continue
        
        df_eng['time_diff'] = abs((df_eng['date_parsed'] - img_dt).dt.total_seconds())
        closest_idx = df_eng['time_diff'].idxmin()
        
       
        if df_eng.loc[closest_idx, 'time_diff'] < 86400:
            shortcode = df_eng.loc[closest_idx, 'shortcode']
            
            
            if shortcode in matches:
                existing_dt = matches[shortcode]['timestamp']
                existing_diff = abs((existing_dt - img_dt).total_seconds())
                new_diff = df_eng.loc[closest_idx, 'time_diff']
                
                if new_diff < existing_diff:
                    matches[shortcode] = {
                        'images': img_list,
                        'timestamp': img_dt
                    }
            else:
                matches[shortcode] = {
                    'images': img_list,
                    'timestamp': img_dt
                }
    
    print(f"    Matched {len(matches)} posts with images")
    return matches




def analyze():
    device = get_device()
    print(f" Using device: {device}")

    print("\n Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    
    text_tokens = clip.tokenize(CLOTHING_TYPES).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    
    print(f"\n Loading engagement CSV...")
    df_eng = pd.read_csv(ENGAGEMENT_CSV)
    
    
    if "likes" not in df_eng.columns:
        df_eng["likes"] = 0
    if "comments" not in df_eng.columns:
        df_eng["comments"] = 0
    
    df_eng["likes"] = df_eng["likes"].clip(lower=0)
    df_eng["engagement_score"] = df_eng["likes"] + 4 * df_eng["comments"]
    
    print(f"    Loaded {len(df_eng)} posts")

    
    folder = Path(DOWNLOAD_FOLDER)
    images = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    print(f"\n Found {len(images)} images")

    
    image_groups = group_images_by_timestamp(images)
    
    
    matches = match_images_to_csv(image_groups, df_eng)
    
    if not matches:
        print("\n No matches found between images and CSV!")
        print(" Try checking if dates in CSV match image timestamps")
        return

    
    results = {}
    
    print(f"\n Analyzing {len(matches)} posts...\n")
    
    for idx, (shortcode, data) in enumerate(matches.items(), 1):
        images_list = data['images']
        
        print(f"[{idx}/{len(matches)}] Processing {shortcode} ({len(images_list)} images)")
        
        img_path = images_list[0]
        
        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

            with torch.no_grad():
                img_feat = model.encode_image(image)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                scores = (img_feat @ text_features.T).softmax(dim=-1)[0].cpu().numpy()

            idx_max = scores.argmax()
            cloth = CLOTHING_TYPES[idx_max]
            confidence = float(scores[idx_max])
            
            row = df_eng[df_eng['shortcode'] == shortcode].iloc[0]
            
            results[shortcode] = {
                "shortcode": shortcode,
                "clothing_type": cloth,
                "confidence": confidence,
                "likes": int(row["likes"]),
                "comments": int(row["comments"]),
                "engagement_score": float(row["engagement_score"]),
                "num_images": len(images_list),
                "date": row["date"] if "date" in df_eng.columns else None
            }
            
            print(f" {cloth} (confidence: {confidence:.1%})")
            print(f" Engagement: {row['likes']} likes, {row['comments']} comments\n")
            
        except Exception as e:
            print(f" Error: {e}\n")
            continue

    
    if results:
        df_final = pd.DataFrame(results.values())
        df_final = df_final.sort_values("engagement_score", ascending=False)
        
        output_file = "fashion_engagement_analysis.csv"
        df_final.to_csv(output_file, index=False)
        
        
        print(f" Successfully analyzed {len(results)} posts!")
        print(f" Saved to: {output_file}")
        
        
        print(f"\n TOP 5 PERFORMING OUTFITS:\n")
        top5 = df_final[["clothing_type", "confidence", "likes", "comments", "engagement_score"]].head()
        print(top5.to_string(index=False))
        
        print(f"\n ENGAGEMENT BY CLOTHING TYPE:\n")
        clothing_stats = df_final.groupby('clothing_type').agg({
            'engagement_score': ['mean', 'count'],
            'likes': 'mean'
        }).round(2)
        print(clothing_stats)
        
    else:
        print("\n No results generated")




if __name__ == "__main__":
    analyze()










