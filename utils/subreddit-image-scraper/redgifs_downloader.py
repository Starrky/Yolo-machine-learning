# redgifs_downloader.py
import os
import asyncio
import aiohttp
from aiohttp import ClientTimeout

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36"
}
TOKEN_URL = "https://api.redgifs.com/v2/auth/temporary"


async def download_redgifs_from_list(urls, output_dir="downloads", concurrency=5):
    os.makedirs(output_dir, exist_ok=True)
    TOKEN = None

    async def get_token(session):
        nonlocal TOKEN
        async with session.get(TOKEN_URL, headers=HEADERS) as resp:
            resp.raise_for_status()
            data = await resp.json()
            TOKEN = data.get("token")
            print(f"🔑 Got token: {TOKEN[:8]}...")

    async def get_video_url(session, slug):
        nonlocal TOKEN
        if not TOKEN:
            await get_token(session)

        api_url = f"https://api.redgifs.com/v2/gifs/{slug}"
        headers = {**HEADERS, "Authorization": f"Bearer {TOKEN}"}

        try:
            async with session.get(api_url, headers=headers) as resp:
                if resp.status == 401:  # token expired
                    await get_token(session)
                    headers["Authorization"] = f"Bearer {TOKEN}"
                    async with session.get(api_url, headers=headers) as resp_retry:
                        resp_retry.raise_for_status()
                        data = await resp_retry.json()
                else:
                    resp.raise_for_status()
                    data = await resp.json()

            video_url = (
                data.get("gif", {}).get("urls", {}).get("hd") or
                data.get("gif", {}).get("urls", {}).get("sd")
            )
            return video_url
        except Exception as e:
            print(f"⚠️ Failed to get API for {slug}: {e}")
            return None

    async def download_video(session, url, output_path):
        try:
            async with session.get(url, headers=HEADERS) as resp:
                resp.raise_for_status()
                with open(output_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(8192):
                        f.write(chunk)
            print(f"⬇️ Downloaded {output_path}")
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")

    sem = asyncio.Semaphore(concurrency)

    async def process_slug(session, slug, output_path):
        async with sem:
            if os.path.exists(output_path):
                print(f"✔️ Skipping {slug}, already exists")
                return

            video_url = await get_video_url(session, slug)
            if video_url:
                await download_video(session, video_url, output_path)
            else:
                print(f"❌ Could not get video for {slug}")

    async with aiohttp.ClientSession(timeout=ClientTimeout(total=60)) as session:
        tasks = []
        for url, dest in urls:
            slug = url.split("/")[-1].split(";")[0].split("#")[0]
            tasks.append(process_slug(session, slug, dest))

        await asyncio.gather(*tasks)
