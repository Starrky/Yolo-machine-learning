"""
reddit_image_scraper.py

arguments:
    --subs: Comma-separated list of subreddits to scrape
    --subs-file: Path to a file containing one subreddit per line
    --limit: Maximum number of posts to scrape per subreddit
    --sort: Sorting method for posts (hot, new, top, rising, controversial)
    --time_filter: Time filter for posts (all, day, hour, month, week, year)
    --out: Output directory for downloaded media
    --skip-gifs: Skip downloading GIFs
e.g
python reddit_image_scraper.py --subs "cats,birds,dogs," --sort hot --limit 500


- Downloads images, GIFs, MP4s (Reddit, Redgifs via async downloader, Gfycat, v.redd.it)
- Avoids duplicate downloads
- Prints summary counts of images and videos/gifs
- Prints debug info for every Redgifs link found
"""

from __future__ import annotations
import os, sys, hashlib, shutil, argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
import requests
from tqdm import tqdm
import asyncio

from redgifs_downloader import download_redgifs_from_list

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
VIDEO_EXTS = {'.mp4', '.gif'}
DEFAULT_USER_AGENT = 'reddit-image-scraper/0.1 by (https://github.com/Starrky)'

# ----------------------
# Utilities
# ----------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Reddit media scraper')
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument('--subs')
    group.add_argument('--subs-file')
    p.add_argument('--limit', type=int, default=500)
    p.add_argument('--sort', choices=['hot','new','top','rising','controversial'], default='hot')
    p.add_argument('--time_filter', choices=['all','day','hour','month','week','year'], default='all')
    p.add_argument('--out', default='reddit_images')
    p.add_argument('--skip-gifs', action='store_true')
    args, _ = p.parse_known_args(argv)
    return args

def sanitize_filename(s: str) -> str:
    keep = " ._-()[]{}"
    return ''.join(c for c in s if c.isalnum() or c in keep).strip()[:200]

def download_file(session: requests.Session, url: str, dest_path: Path) -> Dict[str, Any]:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        return {'status': 'skipped', 'path': str(dest_path)}
    try:
        headers = {'User-Agent': session.headers.get('User-Agent'), 'Accept': '*/*'}
        with session.get(url, stream=True, timeout=30, headers=headers, allow_redirects=True) as r:
            r.raise_for_status()
            tmp = dest_path.with_suffix(dest_path.suffix + '.part')
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(8192):
                    if chunk: f.write(chunk)
            tmp.rename(dest_path)
            return {'status': 'ok', 'path': str(dest_path)}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

# ----------------------
# Reddit source
# ----------------------

class RedditSource:
    def __init__(self, user_agent: str):
        self.user_agent = user_agent

    def fetch_subreddit(self, subreddit: str, sort: str='hot', limit: int=500, time_filter: str='all'):
        after=None
        fetched=0
        while fetched<limit:
            r = requests.get(f'https://www.reddit.com/r/{subreddit}/{sort}.json',
                             params={'limit':500,'after':after},
                             headers={'User-Agent':self.user_agent})
            r.raise_for_status()
            data=r.json().get('data',{})
            children=data.get('children',[])
            if not children: break
            for c in children:
                yield c['data']
            after=data.get('after')
            if not after: break
            fetched+=len(children)

# ----------------------
# Media extraction
# ----------------------

def extract_media_from_post(post: Dict[str, Any]) -> List[Dict[str,str]]:
    result = []
    url = post.get('url')

    # Reddit hosted video (v.redd.it)
    if post.get('secure_media') and post['secure_media'].get('reddit_video'):
        reddit_video = post['secure_media']['reddit_video']
        fallback_url = reddit_video.get('fallback_url')
        if fallback_url: result.append({'url': fallback_url, 'type': 'video'})

    # Redgifs
    elif url and 'redgifs.com' in url:
        #print(f"[DEBUG] Found Redgifs link: {url}")
        result.append({'url': url, 'type': 'video'})

    # Gfycat
    elif url and 'gfycat.com' in url:
        id_part = url.rstrip('/').split('/')[-1]
        url = f'https://giant.gfycat.com/{id_part}.mp4'
        result.append({'url': url, 'type': 'video'})

    # Direct video/image
    elif url and any(url.lower().endswith(e) for e in IMAGE_EXTS | VIDEO_EXTS):
        typ = 'video' if url.lower().endswith(tuple(VIDEO_EXTS)) else 'image'
        result.append({'url': url, 'type': typ})

    # gallery
    if post.get('is_gallery') and post.get('media_metadata'):
        for m in post['media_metadata'].values():
            u = m.get('s',{}).get('u','').replace('&amp;','&')
            if u: result.append({'url': u, 'type': 'image'})

    return result

# ----------------------
# Processing
# ----------------------

def process_subreddit(sub: str, source: RedditSource, out_dir: Path, args):
    session = requests.Session()
    session.headers.update({'User-Agent': source.user_agent})
    meta_dir = out_dir / 'metadata'
    meta_dir.mkdir(parents=True, exist_ok=True)
    seen_urls = set()
    image_count = 0
    video_count = 0

    to_download = []
    for post in source.fetch_subreddit(sub, sort=args.sort, limit=args.limit, time_filter=args.time_filter):
        medias = extract_media_from_post(post)
        for i, m in enumerate(medias):
            u, typ = m['url'], m['type']
            if not u or u in seen_urls: continue
            seen_urls.add(u)
            if args.skip_gifs and u.lower().endswith('.gif'): continue
            h = hashlib.sha1(u.encode()).hexdigest()[:12]
            title_safe = sanitize_filename(post.get('title') or post.get('id') or '')
            ext = Path(u.split('?',1)[0]).suffix
            if not ext and typ=='video': ext = '.mp4'
            elif not ext and typ=='image': ext = '.jpg'
            fname = f"{sub}_{post.get('id')}_{i}_{h}{ext}"
            dest = out_dir / sub / fname
            to_download.append( (u, dest, typ) )

    # Separate Redgifs for async download
    redgifs_to_download = [(u, dest) for u, dest, typ in to_download if 'redgifs.com' in u]
    to_download = [(u, dest, typ) for u, dest, typ in to_download if 'redgifs.com' not in u]

    # Download images/videos (non-Redgifs)
    for u, dest, typ in tqdm(to_download, desc=f'Downloading {sub}'):
        res = download_file(session, u, dest)
        if res['status']=='ok':
            if typ=='image': image_count+=1
            elif typ=='video': video_count+=1

    # Download Redgifs asynchronously
    if redgifs_to_download:
        asyncio.run(download_redgifs_from_list(redgifs_to_download, output_dir=str(out_dir / sub)))
        video_count += len(redgifs_to_download)

    print(f"Downloaded {image_count} images, {video_count} videos/gifs from {sub}")

    if meta_dir.exists(): shutil.rmtree(meta_dir)

# ----------------------
# Main
# ----------------------

def main(argv: Optional[List[str]]=None) -> int:
    args = parse_args(argv)
    subs=[]

    if args.subs: subs=[s.strip() for s in args.subs.split(',') if s.strip()]
    elif args.subs_file: subs= [line.strip() for line in Path(args.subs_file).read_text().splitlines() if line.strip()]
    if not subs: print('No subreddits provided'); return 2

    out_dir=Path(args.out); out_dir.mkdir(parents=True,exist_ok=True)
    source=RedditSource(user_agent='reddit-image-scraper')

    for sub in subs:
        process_subreddit(sub, source, out_dir, args)

    return 0

if __name__=='__main__': sys.exit(main())
