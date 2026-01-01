import os
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

base_url = "https://www.wcoomd.org/-/media/wco/public/global/pdf/topics/nomenclature/instruments-and-tools/hs-nomenclature-2022/2022/"

# Mapping of Chapters to Sections
ranges = [
    (1, 5, 1), (6, 14, 2), (15, 15, 3), (16, 24, 4), (25, 27, 5),
    (28, 38, 6), (39, 40, 7), (41, 43, 8), (44, 46, 9), (47, 49, 10),
    (50, 63, 11), (64, 67, 12), (68, 70, 13), (71, 71, 14), (72, 83, 15),
    (84, 85, 16), (86, 89, 17), (90, 92, 18), (93, 93, 19), (94, 96, 20), (97, 97, 21)
]

# Thread-safe counter
lock = threading.Lock()
completed_count = 0

def generate_links():
    links = []
    
    # Section notes PDFs (Section number + "00")
    section_notes = [1, 2, 4, 6, 7, 11, 15, 16, 17]
    for section in section_notes:
        filename = f"{section:02d}00_2022e.pdf"
        links.append(base_url + filename)
    
    # Chapter PDFs
    for start, end, section in ranges:
        for chapter in range(start, end + 1):
            filename = f"{section:02d}{chapter:02d}_2022e.pdf"
            links.append(base_url + filename)
    
    return links

def download_single_pdf(url, output_path, total):
    global completed_count
    filename = url.split("/")[-1]
    filepath = output_path / filename
    
    if filepath.exists():
        with lock:
            completed_count += 1
            print(f"[{completed_count}/{total}] Skipping {filename} (already exists)")
        return filename, "skipped"
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(filepath, "wb") as f:
            f.write(response.content)
        
        with lock:
            completed_count += 1
            print(f"[{completed_count}/{total}] Downloaded {filename}")
        return filename, "success"
    except requests.exceptions.RequestException as e:
        with lock:
            completed_count += 1
            print(f"[{completed_count}/{total}] Error {filename}: {e}")
        return filename, f"error: {e}"

def download_pdfs(output_dir="hs6", max_workers=10):
    global completed_count
    completed_count = 0
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    links = generate_links()
    total = len(links)
    
    print(f"Downloading {total} PDFs to '{output_dir}/' using {max_workers} threads...\n")
    
    results = {"success": 0, "skipped": 0, "error": 0}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single_pdf, url, output_path, total): url for url in links}
        
        for future in as_completed(futures):
            filename, status = future.result()
            if status == "success":
                results["success"] += 1
            elif status == "skipped":
                results["skipped"] += 1
            else:
                results["error"] += 1
    
    print(f"\nDone! Results: {results['success']} downloaded, {results['skipped']} skipped, {results['error']} errors")
    print(f"PDFs saved to '{output_dir}/'")

if __name__ == "__main__":
    download_pdfs()
