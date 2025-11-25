#!/usr/bin/env python3
"""
Standalone script to test deadline extraction from PDF files
"""
import sys
import os
from pathlib import Path
import json

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.utils.pdf_parser import parse_pdf, parse_text_document
from app.utils.llm_service import extract_deadlines_from_text
from app.config import settings
from openai import OpenAI

# Initialize OpenAI client for checking
client = OpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None

print("=" * 80)
print("📅 DEADLINE EXTRACTION TEST")
print("=" * 80)
print()

# Check if file path is provided
if len(sys.argv) < 2:
    # If no argument, look for the most recent file in uploads folder
    uploads_dir = Path(__file__).parent.parent.parent / "uploads"
    if uploads_dir.exists():
        pdf_files = sorted(uploads_dir.glob("*.pdf"), key=lambda x: x.stat().st_mtime, reverse=True)
        if pdf_files:
            file_path = pdf_files[0]
            print(f"📄 Using most recent PDF: {file_path.name}\n")
        else:
            print("❌ No PDF files found in uploads folder")
            print("Usage: python test_deadline_extraction.py <path_to_pdf>")
            sys.exit(1)
    else:
        print("❌ Uploads folder not found")
        print("Usage: python test_deadline_extraction.py <path_to_pdf>")
        sys.exit(1)
else:
    file_path = Path(sys.argv[1])

# Check if file exists
if not file_path.exists():
    print(f"❌ File not found: {file_path}")
    sys.exit(1)

# Parse the file
print(f"🔍 Step 1: Parsing PDF...")
try:
    with open(file_path, "rb") as f:
        content = f.read()
    
    if file_path.suffix.lower() == ".pdf":
        text = parse_pdf(content)
    else:
        text = parse_text_document(content, file_path.suffix.lower())
    
    print(f"✅ Parsed {len(text)} characters")
    print(f"📝 First 200 chars: {text[:200]}...\n")
except Exception as e:
    print(f"❌ Error parsing file: {e}")
    sys.exit(1)

# Extract deadlines
print(f"🤖 Step 2: Extracting deadlines using LLM...")
api_key_valid = settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.startswith('sk-')
print(f"OpenAI API Key configured: {'Yes ✅' if api_key_valid else 'No ❌ (will use keyword extraction)'}")
if api_key_valid:
    print(f"   Using API key: {settings.OPENAI_API_KEY[:15]}...")
print()

try:
    deadlines = extract_deadlines_from_text(text, context="syllabus")
    
    print("=" * 80)
    print(f"📋 FOUND {len(deadlines)} DEADLINES")
    print("=" * 80)
    print()
    
    for i, deadline in enumerate(deadlines, 1):
        print(f"Deadline #{i}:")
        print(f"  📌 Title: {deadline.get('title', 'N/A')}")
        print(f"  📅 Date: {deadline.get('date', 'N/A')}")
        print(f"  🏷️  Type: {deadline.get('type', 'N/A')}")
        print(f"  📝 Description: {deadline.get('description', 'N/A')[:100]}{'...' if len(deadline.get('description', '')) > 100 else ''}")
        print(f"  ⏱️  Estimated Hours: {deadline.get('estimated_hours', 'N/A')}")
        print()
    
    print("=" * 80)
    print("✅ Deadline extraction complete!")
    print("=" * 80)
    
    # Option to save to JSON
    print()
    save_option = input("💾 Save results to JSON file? (y/n): ").lower().strip()
    if save_option == 'y':
        output_file = Path(__file__).parent.parent.parent / "uploads" / f"{file_path.stem}_deadlines.json"
        with open(output_file, 'w') as f:
            json.dump(deadlines, f, indent=2)
        print(f"✅ Saved to: {output_file}")

except Exception as e:
    print(f"❌ Error extracting deadlines: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
