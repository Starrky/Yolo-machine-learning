import asyncio
import json
import os
import shutil
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from urllib.parse import urlparse, urlencode
from playwright.async_api import async_playwright
import concurrent.futures
import multiprocessing
import hashlib
from pyaml_env import parse_config, BaseConfig

config = BaseConfig(parse_config('configs/project_config.yaml'))

image_query = config.google_scraper.image_query

# Get number of CPU cores for optimal concurrency
MAX_CONCURRENT_DOWNLOADS = min(multiprocessing.cpu_count() * 4, 50)  # Cap at 50 to avoid overwhelming servers
MAX_CONNECTIONS = min(multiprocessing.cpu_count() * 2, 20)  # Connection pool size


# Function to extract the domain from a URL
def extract_domain(url):
    """
    Extract the domain from the given URL.
    If the domain starts with 'www.', it removes it.

    Args:
        url (str): The URL to extract the domain from.

    Returns:
        str: The extracted domain.
    """
    domain = urlparse(url).netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


# Add these globals for tracking duplicates
downloaded_urls = set()
downloaded_hashes = set()

def get_image_hash(content):
    """Generate hash of image content to detect duplicates"""
    return hashlib.md5(content).hexdigest()


# Enhanced download function with duplicate detection
async def download_image_concurrent(session, img_url, file_path, semaphore, retries=3):
    """
    Download an image concurrently with duplicate detection
    """
    # Skip if URL already downloaded
    if img_url in downloaded_urls:
        print(f"⏭️  Skipping duplicate URL: {os.path.basename(file_path)}")
        return False
        
    async with semaphore:  # Limit concurrent downloads
        attempt = 0
        while attempt < retries:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://www.google.com/'
                }
                
                async with session.get(img_url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        # Check for duplicate content
                        content_hash = get_image_hash(content)
                        if content_hash in downloaded_hashes:
                            print(f"⏭️  Skipping duplicate content: {os.path.basename(file_path)}")
                            downloaded_urls.add(img_url)  # Still mark URL as processed
                            return False
                        
                        # Write file using asyncio to avoid blocking
                        await asyncio.get_event_loop().run_in_executor(
                            None, 
                            lambda: write_file_sync(file_path, content)
                        )
                        
                        # Mark as downloaded
                        downloaded_urls.add(img_url)
                        downloaded_hashes.add(content_hash)
                        
                        print(f"✓ Downloaded: {os.path.basename(file_path)}")
                        return True
                    else:
                        print(f"✗ Failed download {img_url}. Status: {response.status}")
            except Exception as e:
                print(f"✗ Error downloading {img_url}: {e}")
            
            attempt += 1
            if attempt < retries:
                await asyncio.sleep(min(2**attempt, 5))
        
        print(f"✗ Failed to download after {retries} attempts: {os.path.basename(file_path)}")
        return False

def write_file_sync(file_path, content):
    """Synchronous file writing to be run in executor"""
    with open(file_path, "wb") as f:
        f.write(content)

# Enhanced pagination and scrolling functions
async def load_more_images_with_pagination(page, target_images):
    """
    Load more images by scrolling and handling pagination
    """
    print(f"🔄 Loading images with pagination (target: {target_images})...")
    
    images_found = 0
    scroll_attempts = 0
    max_scroll_attempts = 20  # Prevent infinite scrolling
    
    while images_found < target_images and scroll_attempts < max_scroll_attempts:
        # Scroll down to load more images
        previous_height = await page.evaluate("document.body.scrollHeight")
        
        # Scroll to bottom
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(2)  # Wait for new content to load
        
        # Check if "Show more results" button exists and click it
        try:
            show_more_selectors = [
                'input[value="Show more results"]',
                'div[jsname="BKxS1e"]',  # Google's "Show more results" div
                'div[role="button"]:has-text("Show more results")',
                'span:has-text("Show more results")'
            ]
            
            for selector in show_more_selectors:
                try:
                    show_more_button = await page.wait_for_selector(selector, timeout=2000)
                    if show_more_button:
                        print("📄 Found 'Show more results' button, clicking...")
                        await show_more_button.click()
                        await asyncio.sleep(3)  # Wait for new page to load
                        break
                except:
                    continue
        except:
            pass
        
        # Check for new height (new content loaded)
        new_height = await page.evaluate("document.body.scrollHeight")
        
        # Count current images
        current_images = await count_available_images(page)
        
        print(f"📊 Scroll attempt {scroll_attempts + 1}: Found {current_images} images")
        
        if current_images > images_found:
            images_found = current_images
            scroll_attempts = 0  # Reset counter when we find new images
        else:
            scroll_attempts += 1
            
        # If no new content after scrolling, try to find and click "Load more" type buttons
        if new_height == previous_height:
            try:
                load_more_selectors = [
                    'div[jsaction*="trigger.EGZ4hc"]',  # Google's load more trigger
                    'div[role="button"][jsaction]',
                    'span[jsaction*="click"]'
                ]
                
                for selector in load_more_selectors:
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        try:
                            text = await element.inner_text()
                            if any(keyword in text.lower() for keyword in ['more', 'load', 'show']):
                                print(f"🔄 Clicking load more element: {text[:50]}...")
                                await element.click()
                                await asyncio.sleep(3)
                                break
                        except:
                            continue
            except:
                pass
    
    print(f"✅ Finished loading. Found {images_found} total images after {scroll_attempts} scroll attempts")
    return images_found

async def count_available_images(page):
    """Count currently available images on the page"""
    image_selectors = [
        'div[data-attrid="images universal"]',
        'div[jsname="N760b"]',
        'div[data-ved] img',
        'div[role="listitem"]',
        'div[data-tbnid]'
    ]
    
    for selector in image_selectors:
        try:
            elements = await page.query_selector_all(selector)
            if elements:
                return len(elements)
        except:
            continue
    
    return 0

# Replace the existing extract_all_image_urls function with this paginated version
async def extract_all_image_urls_paginated(page, max_images):
    """
    Extract image URLs with pagination support and duplicate prevention
    """
    print(f"🎯 Extracting up to {max_images} unique image URLs...")
    
    # First, load enough images through pagination
    await load_more_images_with_pagination(page, max_images * 2)  # Load extra to account for duplicates
    
    # Find all image elements
    image_selectors = [
        'div[data-attrid="images universal"]',
        'div[jsname="N760b"]', 
        'div[data-ved] img',
        'h3[class*="ob5Hkd"] ~ div img',
        'div[role="listitem"]',
        'div[data-tbnid]'
    ]
    
    image_elements = []
    for selector in image_selectors:
        try:
            elements = await page.query_selector_all(selector)
            if elements:
                # Take more elements than needed to account for duplicates/failures
                image_elements = elements[:max_images * 3] if max_images else elements
                print(f"🔍 Found {len(image_elements)} image elements using selector: {selector}")
                break
        except:
            continue
    
    if not image_elements:
        return []
    
    # Extract URLs in batches with duplicate checking
    unique_image_urls = []
    seen_urls = set()
    batch_size = 10
    
    for i in range(0, len(image_elements), batch_size):
        if len(unique_image_urls) >= max_images:
            break
            
        batch = image_elements[i:i + batch_size]
        batch_tasks = []
        
        for idx, element in enumerate(batch):
            global_idx = i + idx + 1
            task = extract_single_image_url(page, element, global_idx)
            batch_tasks.append(task)
        
        # Process batch concurrently
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        for result in batch_results:
            if isinstance(result, dict) and result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_image_urls.append(result)
                
                if len(unique_image_urls) >= max_images:
                    break
    
    print(f"✅ Extracted {len(unique_image_urls)} unique image URLs")
    return unique_image_urls

async def extract_single_image_url(page, image_element, idx):
    """Extract single image URL with metadata"""
    try:
        # Click on the image to get full view
        await image_element.click()
        await asyncio.sleep(0.5)  # Reduced wait time
        
        # Try multiple selectors for the full-size image
        full_image_selectors = [
            "img.sFlh5c.FyHeAf.iPVvYb[jsaction]",
            "img[jsname='kn3ccd']",
            "img[alt]:not([alt=''])",
            "div[data-tbnid] img"
        ]
        
        img_tag = None
        for img_selector in full_image_selectors:
            try:
                img_tag = await page.wait_for_selector(img_selector, timeout=2000)
                if img_tag:
                    break
            except:
                continue
        
        if not img_tag:
            return None
        
        # Get the image URL
        img_url = await img_tag.get_attribute("src")
        if not img_url or img_url.startswith("data:"):
            img_url = await img_tag.get_attribute("data-src") or await img_tag.get_attribute("data-original")
        
        if not img_url or img_url.startswith("data:"):
            return None
        
        # Get metadata
        try:
            source_url_element = await page.query_selector('div[jsname="figiqf"] a[class*="YsLeY"]', timeout=1000)
            source_url = await source_url_element.get_attribute("href") if source_url_element else "N/A"
        except:
            source_url = "N/A"
        
        image_description = await img_tag.get_attribute("alt") or f"Image {idx}"
        source_name = extract_domain(source_url) if source_url != "N/A" else "N/A"
        
        file_extension = os.path.splitext(urlparse(img_url).path)[1] or ".png"
        
        return {
            "idx": idx,
            "url": img_url,
            "extension": file_extension,
            "description": image_description,
            "source_url": source_url,
            "source_name": source_name
        }
        
    except Exception as e:
        print(f"Error extracting image {idx}: {e}")
        return None


# Enhanced main function that guarantees max_images unique downloads
async def scrape_google_images(search_query=image_query, max_images=None, timeout_duration=30):
    """
    High-performance scrape with guaranteed unique image count
    """
    # Reset global tracking sets
    global downloaded_urls, downloaded_hashes
    downloaded_urls = set()
    downloaded_hashes = set()

    if not max_images:
        max_images = 50  # Default

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--no-first-run',
                '--disable-default-apps',
                '--disable-popup-blocking'
            ]
        )

        page = await browser.new_page()
        await page.set_viewport_size({"width": 1920, "height": 1080})

        await page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Build search URL
        query_params = urlencode({
            "q": search_query,
            "tbm": "isch",
            "safe": "off",
            "filter": "0",
            "tbs": "isz:m",
            "ijn": "0"
        })
        search_url = f"https://www.google.com/search?{query_params}"

        print(f"🚀 Navigating to: {search_url}")
        await page.goto(search_url)

        # Handle cookie consent
        try:
            accept_selectors = [
                'button[id="L2AGLb"]',
                'button:has-text("Accept all")',
                'button[jsname="b3VHJd"]'
            ]

            for selector in accept_selectors:
                try:
                    accept_button = await page.wait_for_selector(selector, timeout=3000)
                    if accept_button:
                        await accept_button.click()
                        print("✅ Accepted cookies")
                        await asyncio.sleep(2)
                        break
                except:
                    continue
        except:
            pass

        # Set up directories
        download_folder = f"downloaded_images/{search_query}/"
        json_file_path = "google_images_data.json"

        if os.path.exists(download_folder):
            user_input = input(f"Folder '{download_folder}' exists. Delete it? (yes/no): ")
            if user_input.lower() == "yes":
                shutil.rmtree(download_folder)
            else:
                import time
                timestamp = int(time.time())
                archive_folder = f"downloaded_images/{search_query}_archive_{timestamp}"
                shutil.move(download_folder, archive_folder)

        os.makedirs(download_folder, exist_ok=True)

        print(f"🎯 Target: {max_images} unique images")

        # Create session for downloads
        connector = TCPConnector(
            limit=MAX_CONNECTIONS,
            limit_per_host=10,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        timeout = ClientTimeout(total=timeout_duration, connect=10)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

        successful_downloads = 0
        final_image_data = []

        async with ClientSession(connector=connector, timeout=timeout) as session:
            # Keep downloading until we get max_images unique images
            while successful_downloads < max_images:
                remaining_needed = max_images - successful_downloads
                print(f"\n🔄 Need {remaining_needed} more unique images. Extracting URLs...")

                # Extract URLs (get extra to account for duplicates and failures)
                extraction_target = min(remaining_needed * 5, 100)  # Get 5x what we need, capped at 100
                image_data_list = await extract_all_image_urls_paginated(page, extraction_target)

                if not image_data_list:
                    print("❌ No more images found!")
                    break

                print(f"🚀 Attempting to download {len(image_data_list)} URLs...")

                # Prepare download tasks for this batch
                download_tasks = []
                for img_data in image_data_list:
                    if img_data and img_data['url'] not in downloaded_urls:
                        file_path = os.path.join(download_folder,
                                                 f"image_{successful_downloads + len(download_tasks) + 1}{img_data['extension']}")
                        task = download_image_concurrent_with_count(session, img_data, file_path, semaphore)
                        download_tasks.append(task)

                if not download_tasks:
                    print("⚠️  All URLs already processed, loading more...")
                    continue

                # Execute downloads concurrently
                batch_results = await asyncio.gather(*download_tasks, return_exceptions=True)

                # Count successful downloads in this batch
                batch_success_count = 0
                for result in batch_results:
                    if isinstance(result, dict) and result.get('success'):
                        batch_success_count += 1
                        final_image_data.append(result['metadata'])

                        if successful_downloads + batch_success_count >= max_images:
                            break

                successful_downloads += batch_success_count
                print(f"✅ Batch completed: {batch_success_count} new unique images")
                print(f"📊 Progress: {successful_downloads}/{max_images} unique images downloaded")

                if batch_success_count == 0:
                    print("⚠️  No new unique images in this batch, loading more...")
                    # Force load more content
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(3)

        # Save metadata
        with open(json_file_path, "w") as json_file:
            json.dump(final_image_data, json_file, indent=4)

        print(f"\n🎉 Mission Accomplished!")
        print(f"✅ Successfully downloaded exactly {successful_downloads} unique images")
        print(f"📊 Skipped {len(downloaded_urls) - successful_downloads} duplicates/failures")
        print(f"💾 Used {MAX_CONCURRENT_DOWNLOADS} concurrent downloads on {multiprocessing.cpu_count()} CPU cores")

        await browser.close()


# Enhanced download function that returns metadata on success
async def download_image_concurrent_with_count(session, img_data, file_path, semaphore, retries=3):
    """
    Download an image and return success status with metadata
    """
    img_url = img_data['url']

    # Skip if URL already downloaded
    if img_url in downloaded_urls:
        return {'success': False, 'reason': 'duplicate_url'}

    async with semaphore:
        attempt = 0
        while attempt < retries:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://www.google.com/'
                }

                async with session.get(img_url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.read()

                        # Check for duplicate content
                        content_hash = get_image_hash(content)
                        if content_hash in downloaded_hashes:
                            downloaded_urls.add(img_url)  # Mark URL as processed
                            return {'success': False, 'reason': 'duplicate_content'}

                        # Write file
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: write_file_sync(file_path, content)
                        )

                        # Mark as downloaded
                        downloaded_urls.add(img_url)
                        downloaded_hashes.add(content_hash)

                        print(f"✓ Downloaded: {os.path.basename(file_path)}")

                        # Return success with metadata
                        return {
                            'success': True,
                            'metadata': {
                                "image_description": img_data['description'],
                                "source_url": img_data['source_url'],
                                "source_name": img_data['source_name'],
                                "image_file": file_path.replace(" ", "")
                            }
                        }
                    else:
                        print(f"✗ HTTP {response.status}: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"✗ Error: {os.path.basename(file_path)} - {str(e)[:50]}")

            attempt += 1
            if attempt < retries:
                await asyncio.sleep(min(2 ** attempt, 5))

        return {'success': False, 'reason': 'download_failed'}


# Run with pagination support
if __name__ == "__main__":
    asyncio.run(scrape_google_images(search_query=image_query, max_images=100, timeout_duration=45))