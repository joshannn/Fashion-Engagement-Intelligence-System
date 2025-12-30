import instaloader
import os
import csv
from datetime import datetime
from itertools import islice
def download_instagram():
    # asking username
    username = input("Enter the Instagram username to download: ").strip()
    max_posts = int(input("How many recent posts to download? ").strip())


    # save in folder where script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    download_dir = os.path.join(current_dir, username)
    os.makedirs(download_dir, exist_ok=True)

    # downloading data from insta loader
    L = instaloader.Instaloader(
        dirname_pattern=download_dir,
        filename_pattern="{date_utc}",
        save_metadata=True,
        download_comments=False,
        compress_json=False
    )

    print(f" Downloading posts from @{username} into {download_dir}")

    # creating csv file
    csv_file_path = os.path.join(download_dir, f"{username}_posts.csv")
    csv_columns = ["shortcode", "likes", "comments", "date", "caption", "is_video", "typename"]

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        
        profile = instaloader.Profile.from_username(L.context, username)
        for post in islice(profile.get_posts(), max_posts):
            L.download_post(post, target=username)

            # writing metadata to CSV
            writer.writerow({
                "shortcode": post.shortcode,
                "likes": post.likes,
                "comments": post.comments,
                "date": post.date_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "caption": post.caption or "",
                "is_video": post.is_video,
                "typename": post.typename
            })

    print(f" Download completed! Metadata saved to {csv_file_path}")

if __name__ == "__main__":
    download_instagram()
