# Fashion-Engagement-Intelligence-System

This repository contains a suite of Python scripts designed to automate the collection and analysis of Instagram fashion data. The system utilizes Computer Vision (OpenAI's CLIP model) and web scraping (Instaloader) to identify clothing styles and correlate them with audience engagement metrics.

## System Overview

The project is divided into three primary components:

1. **Data Acquisition:** Downloads media and metadata from Instagram profiles.
2. **Standard Analysis:** Basic image classification using single-stage CLIP matching.
3. **Advanced Analysis (v2):** High-accuracy classification using hierarchical prompting and multi-image feature averaging.

---
# Instagram Fashion Analytics Pipeline

1. **Download**: Fetch Instagram posts with engagement data using Instaloader
2. **Classify**: Use CLIP AI model to identify clothing categories (6 types, 20+ variations)
3. **Analyze**: Match outfits with engagement metrics and rank by performance
4. **Insight**: Discover which fashion styles resonate most with your audience

Built for content creators, fashion marketers, and data enthusiasts.

## Technical Components

### 1. Data Download (datadownload.py)

This script serves as the entry point for data collection. It interfaces with Instagram to retrieve public post data without requiring API credentials in most cases.

* **Functionality:**
* Fetches a user-defined number of recent posts.
* Downloads images and videos to a structured local directory.
* Generates a CSV file containing shortcodes, like counts, comment counts, timestamps, and captions.


* **Requirements:** `instaloader`

### 2. Basic Analyzer (analyzer.py)

The basic analyzer provides a rapid assessment of clothing types using a flat list of 20 fashion categories.

* **Functionality:**
* Uses OpenAI's CLIP (ViT-B/32) to match images against text strings.
* Groups images by timestamp to handle multi-image posts.
* Calculates an engagement score based on a weighted formula: `Likes + (4 * Comments)`.


* **Output:** A CSV report ranked by engagement score and confidence.

### 3. Advanced Analyzer (analyzerv2.py)

This version is the recommended tool for research-grade data. It introduces several logic layers to improve classification precision.

* **Key Features:**
* **Hierarchical Classification:** First identifies the broad category (Casual, Formal, Streetwear, etc.) before narrowing down to specific outfit labels. This reduces "false positive" matches.
* **Multi-Image Averaging:** For carousel posts, the script extracts visual features from every image and calculates the mean feature vector. This ensures the classification represents the entire post rather than just a single slide.
* **Temporal Matching:** Implements a nearest-neighbor algorithm to link local image files to CSV metadata rows based on timestamps, allowing for a 24-hour tolerance window.
* **Sentence-Based Prompting:** Uses descriptive sentences rather than keywords to better align with the CLIP model's training data.



---

## Installation

### Prerequisites

* Python 3.8 or higher
* CUDA-enabled GPU (optional but recommended for faster analysis)

### Required Libraries

Install the dependencies using pip:

```bash
pip install torch torchvision torchaudio
pip install git+https://github.com/openai/CLIP.git
pip install pandas pillow instaloader

```

---

## Usage Instructions

### Step 1: Scrape Data

Run the downloader and enter the target Instagram handle.

```bash
python datadownload.py

```

### Step 2: Analyze Data

Run the advanced analyzer for the best results. Provide the path to the folder and the CSV generated in Step 1.

```bash
python analyzerv2.py

```

---

## Data Output Structure

The final output `fashion_engagement_analysis_v2.csv` includes the following fields:

| Field | Description |
| --- | --- |
| shortcode | Unique Instagram post identifier |
| category | Broad fashion style (e.g., Streetwear) |
| outfit | Specific clothing item identified |
| confidence | Probability score of the AI's prediction |
| engagement | The calculated score (Likes + 4*Comments) |
| num_images | Number of images processed for that post |

---

## Technical Specifications

* **AI Model:** CLIP (Contrastive Language-Image Pre-training)
* **Architecture:** ViT-B/32
* **Framework:** PyTorch
* **Data Processing:** Pandas
* **Image Handling:** Pillow (PIL)
