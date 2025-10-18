#!/usr/bin/env python3
"""
Download FinanceBench PDF documents.

This script downloads all the PDF files referenced in the FinanceBench dataset
to enable full functionality of the benchmark.
"""

import json
import os
import requests
from urllib.parse import urlparse
from pathlib import Path
import time
from tqdm import tqdm

def extract_pdf_links(jsonl_file):
    """Extract unique PDF links from FinanceBench JSONL file."""
    pdf_links = set()

    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            doc_link = data.get('doc_link', '')
            if doc_link and doc_link.endswith('.pdf'):
                pdf_links.add(doc_link)

    return sorted(pdf_links)

def download_pdf(url, output_dir, filename=None):
    """Download a single PDF file."""
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename from URL if not provided
        if not filename:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename.endswith('.pdf'):
                filename = f"{filename}.pdf"

        output_path = output_dir / filename

        # Skip if file already exists
        if output_path.exists():
            print(f"  ⏭️  {filename} already exists, skipping")
            return True

        # Download the file
        print(f"  📥 Downloading {filename}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                f.write(response.content)

        print(f"  ✅ Downloaded {filename} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True

    except Exception as e:
        print(f"  ❌ Error downloading {url}: {e}")
        return False

def main():
    """Main function to download all FinanceBench PDFs."""
    print("📄 FinanceBench PDF Download")
    print("=" * 50)

    # Paths
    jsonl_file = Path("data/financebench/financebench_open_source.jsonl")
    pdf_dir = Path("data/financebench_pdfs")

    # Check if JSONL file exists
    if not jsonl_file.exists():
        print(f"❌ FinanceBench data file not found: {jsonl_file}")
        print("💡 Make sure you've downloaded the FinanceBench dataset first")
        return False

    # Extract PDF links
    print("🔍 Extracting PDF links from dataset...")
    pdf_links = extract_pdf_links(jsonl_file)

    print(f"📊 Found {len(pdf_links)} unique PDF documents")

    if not pdf_links:
        print("❌ No PDF links found in the dataset")
        return False

    # Show some examples
    print("\n📋 Sample PDF links:")
    for i, link in enumerate(pdf_links[:3]):
        print(f"  {i+1}. {link}")
    if len(pdf_links) > 3:
        print(f"  ... and {len(pdf_links) - 3} more")

    # Confirm download
    response = input(f"\n🚨 About to download {len(pdf_links)} PDF files. This may take a while and use significant disk space. Continue? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Download cancelled")
        return False

    # Download PDFs
    print("\n📥 Starting PDF downloads...")
    print("=" * 50)

    success_count = 0
    failed_links = []

    for i, url in enumerate(pdf_links):
        print(f"\n[{i+1}/{len(pdf_links)}] Processing {url}")

        # Extract filename from URL
        filename = None
        try:
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)
        except:
            pass

        success = download_pdf(url, pdf_dir, filename)

        if success:
            success_count += 1
        else:
            failed_links.append(url)

        # Add a small delay to be respectful to servers
        if i < len(pdf_links) - 1:
            time.sleep(1)

    # Summary
    print("\n" + "=" * 50)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"Total PDFs: {len(pdf_links)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_links)}")

    if failed_links:
        print("\n❌ Failed downloads:")
        for link in failed_links[:5]:  # Show first 5 failed links
            print(f"  - {link}")
        if len(failed_links) > 5:
            print(f"  ... and {len(failed_links) - 5} more")

    if success_count > 0:
        total_size = sum(f.stat().st_size for f in pdf_dir.glob("*.pdf"))
        print(f"\n📁 PDFs saved to: {pdf_dir}")
        print(f"💾 Total size: {total_size / 1024 / 1024 / 1024:.2f} GB")

        print("\n✅ FinanceBench PDFs downloaded successfully!")
        print("💡 You can now run evaluations with full document context")
        return True
    else:
        print("\n❌ No PDFs were successfully downloaded")
        return False

if __name__ == "__main__":
    main()
