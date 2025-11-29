"""
Standalone script to test deadline extraction from PDF files
Uses snippet-based approach for reliability across different syllabi.

New mental model:
- hard_deadline tasks for anything due / graded.
- class_session events per class date, each with a list of readings.
"""
import sys
import os
from pathlib import Path
import json
import re
from typing import List, Dict, Optional, Tuple

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

# Change to backend directory for .env loading
os.chdir(backend_dir)

from app.utils.pdf_parser import parse_pdf, parse_text_document
from app.config import settings
from openai import OpenAI

# Initialize OpenAI client
api_key_valid = settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.startswith("sk-")
client = OpenAI(api_key=settings.OPENAI_API_KEY) if api_key_valid else None

# --- Trigger lists for post-filtering ---
HARD_DEADLINE_TRIGGERS = [
    "due",
    "submit",
    "submission",
    "hand in",
    "exam",
    "test",
    "assessment",
    "quiz",
    "final",
    "midterm",
    "paper",
    "project",
    "assignment",
]

READING_TRIGGERS = [
    "read ",
    "reading",
    "readings",
    "chapter",
    "chap.",
    "pp.",
    "required reading",
    "recommended reading",
    "read before class",
]

# Date regex:
#  - numeric: 6/9, 06/09, 6.9.22, 13/10/2023   (NO hyphen to avoid '1-5' chapter junk)
#  - text:    Sept 11, Sep 11, September 11
DATE_REGEX = re.compile(
    r"\b("
    r"(\d{1,2})[/.](\d{1,2})(?:[/.](\d{2,4}))?"  # numeric dd/mm(/yyyy)?
    r"|"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}"  # short month names + day
    r"|"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}"  # long names + day
    r")\b",
    re.IGNORECASE,
)


def get_date_snippets(text: str, before: int = 1, after: int = 3) -> List[str]:
    """
    Extract coarse snippets of text around lines that contain at least one date.
    Each snippet usually contains one or more dates plus local context.
    """
    lines = text.splitlines()
    snippets: List[str] = []
    seen_lines = set()

    for i, line in enumerate(lines):
        if DATE_REGEX.search(line) and i not in seen_lines:
            start = max(0, i - before)
            end = min(len(lines), i + 1 + after)
            snippet = "\n".join(lines[start:end]).strip()
            if snippet:
                snippets.append(snippet)
                # Mark these lines as seen to avoid duplicates
                for j in range(start, end):
                    seen_lines.add(j)

    return snippets


def extract_date_strings(snippet: str) -> List[str]:
    """
    Extract valid date-like strings (day + month) from a snippet
    and ignore things like academic years or chapter ranges.
    """
    matches = DATE_REGEX.findall(snippet)
    date_strings: List[str] = []

    for match in matches:
        full_match = match[0].strip()

        # 1) Numeric formats: dd/mm(/yyyy) or dd.mm(/yyyy)
        m_num = re.match(r"^(\d{1,2})[/.](\d{1,2})(?:[/.](\d{2,4}))?$", full_match)
        if m_num:
            # Reject things like "1/2", "2/2" that are almost always part/section numbers
            if re.match(r"^[1-9]/[1-9]$", full_match):
                continue

            day, month = int(m_num.group(1)), int(m_num.group(2))
            if 1 <= day <= 31 and 1 <= month <= 12:
                date_strings.append(full_match)
            continue

        # 2) Month-name formats: e.g. "Sept 11", "September 29"
        m_mon = re.match(
            r"(?i)^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}$",
            full_match,
        )
        if m_mon:
            date_strings.append(full_match)
            continue

        m_mon_long = re.match(
            r"(?i)^(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}$",
            full_match,
        )
        if m_mon_long:
            date_strings.append(full_match)
            continue

    # Deduplicate
    return list(set(date_strings))


def analyze_snippet(snippet: str, assessment_context: Optional[str] = None) -> Optional[List[Dict]]:
    """
    Call the LLM on a snippet (may contain multiple dates + context)
    and get an array of date-level items.

    If assessment_context is provided, the LLM will be aware of graded
    components and can properly classify them as hard deadlines rather
    than preparatory reading.
    """
    if not client:
        return None

    date_strings = extract_date_strings(snippet)
    if not date_strings:
        return None

    unique_dates = sorted(set(date_strings))
    date_hint = ", ".join(unique_dates)

    # Add assessment context if provided
    assessment_info = ""
    if assessment_context:
        assessment_info = f"""

GRADED ASSESSMENT COMPONENTS:
The following are graded assignments/exams from the TESTS / grading section:
{assessment_context}

If the snippet mentions any of these graded components, classify them as
'hard_deadline' with type 'assignment', 'exam', 'project' or 'assessment' as appropriate,
NOT as preparatory reading.
"""

    prompt = f"""
You are processing a short excerpt from a university syllabus. The text is
shown below. It contains one or more explicit date strings.

Your job is to identify ONLY the concrete student tasks and class sessions
in this snippet, GROUPED BY DATE STRING.

The allowed date strings for this snippet are:
{date_hint}{assessment_info}

For each date, you may output:
- zero or one "class_session" object (with readings attached), and/or
- one or more "hard_deadline" objects (for things that are due, exams, etc.),
or nothing if that date is irrelevant ("ignore").

### IMPORTANT RULES

1. Class sessions
   - A "class_session" represents a class meeting on that date.
   - Use it when the text clearly describes a session (e.g., a topic list,
     guest speaker, or schedule entry).
   - Attach all readings for that class under "prep_tasks" and
     "mandatory_tasks" arrays. Do NOT create separate top-level objects
     just for individual readings.
   - There should usually be at most ONE "class_session" object per date.
     If the snippet has multiple mentions, merge them into one object.

2. What counts as a HARD DEADLINE
   - Use "hard_deadline" when something is due or must be submitted, or is a
     clearly graded assessment on that date. Typical trigger words:
       "due", "submit", "submission", "hand in", "exam", "test",
       "assessment", "quiz", "final", "midterm", "paper", "project",
       "assignment".
   - If the date is only used for a class meeting, a review session, or
     "Course and Grading Structure" WITHOUT any of those verbs, do NOT
     invent a deadline. Either use "class_session" or "ignore".
   - If there are MULTIPLE assignments/assessments/projects mentioned for
     the same date, create a separate hard_deadline entry for EACH distinct
     deliverable (e.g., “Second Research Project Review (Final)” and
     “Third Research Project: Interim Report” should be two items).

3. What counts as READING TASKS (inside class_session)
   - Only create reading tasks (inside a class_session) if they are under
     headings like:
       "Readings", "Readings for Discussion", "Read before class",
       "Required Reading", "Recommended Reading".
   - Also include recommended readings explicitly marked as such (e.g.,
     "not required but recommended") and you may label them with type
     "reading_optional".
   - Do NOT treat lecture "Topics" as readings. Do NOT create readings from
     bullets under "Topics", "Timing", etc., unless the text explicitly says
     "Read", "Reading", "Chapter", "Ch.", "Chap." or clearly names a
     book/article or chapter/section as required/recommended reading.

4. In-Class Assessments
   - If the snippet mentions "In-Class Assessment" or "in-class skills
     assessment", you SHOULD create at least one graded "hard_deadline"
     of type "assessment" on that date for each such occurrence, even if
     you also represent the session as "class_session".
     For example, title could be:
       "In-Class Assessment: Valuation of Property Types (1)".

5. Avoid generic umbrella deadlines
   - Do NOT create vague tasks like "Submit primary research assignments"
     unless the snippet explicitly says that ALL of them are due on that
     exact date. Prefer the specific names used by the syllabus
     ("Special Assignment #1", "First Research Project", etc.).

6. Multi-date phrases
   - When a sentence mentions multiple dates (e.g., a class on one date
     and something "due" on a different date), assign the deadline to the
     date that appears in the same "due/submit" phrase (e.g., "due Oct. 3"
     must be assigned to "Oct. 3", NOT to the earlier class date).

7. Do NOT hallucinate
   - Use ONLY the date strings above; do not invent new dates.
   - Do not create tasks that cannot be clearly justified from the text.

### OUTPUT FORMAT

Return a JSON ARRAY. Each element corresponds to ONE interpretation
for a single date string and has:

{{
  "kind": "class_session" | "hard_deadline" | "ignore",
  "date_string": "<one of: {date_hint}>",

  "session_title": "optional, for class_session (e.g. session topic line)",

  "prep_tasks": [
    {{"title": "...", "type": "reading_preparatory" | "reading_optional" | "reading_mandatory"}}
  ],

  "mandatory_tasks": [
    {{"title": "...", "type": "reading_mandatory" | "reading_optional"}}
  ],

  "hard_deadlines": [
    {{
      "title": "...",
      "type": "assignment" | "exam" | "project" | "assessment" | "administrative",
      "description": "max 120 chars",
      "assessment_name": "optional, exact name from the graded components list if this matches one"
    }}
  ]
}}

Notes:
- For "hard_deadline" objects, you usually leave "session_title",
  "prep_tasks", and "mandatory_tasks" empty.
- For "class_session" objects, you usually leave "hard_deadlines" empty
  and only use the reading arrays.

If nothing useful for a given date, you may omit that date entirely or set
"kind": "ignore" and empty lists.

Syllabus snippet:
\"\"\"{snippet}\"\"\""""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You strictly extract structured tasks from syllabus snippets "
                        "without hallucinating new dates. Always return a JSON array."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()

        try:
            arr = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if not m:
                return None
            arr = json.loads(m.group(0))

        if not isinstance(arr, list):
            arr = [arr]

        return arr

    except Exception as e:
        print(f"⚠️  Error analyzing snippet: {e}")
        return None


def extract_inclass_assessment_tasks(text: str) -> List[Dict]:
    """
    Deterministically extract in-class assessments from the raw syllabus text.
    We track the most recent date seen and assign that date to any
    'In-Class Assessment:' line that follows.
    """
    lines = text.splitlines()
    current_date: Optional[str] = None
    tasks: List[Dict] = []

    for line in lines:
        # Update current_date when we see any date in the line
        m_date = DATE_REGEX.search(line)
        if m_date:
            current_date = m_date.group(0).strip()

        # Look for in-class assessments
        if "In-Class Assessment" in line or "in-class skills assessment" in line:
            if not current_date:
                continue

            # Try to extract title after the colon
            m_assess = re.search(r"In-Class Assessment:\s*(.+)", line, re.IGNORECASE)
            if m_assess:
                title = m_assess.group(1).strip()
            else:
                title = "In-Class Assessment"

            tasks.append(
                {
                    "date": current_date,
                    "title": title,
                    "description": "In-class graded assessment.",
                    "type": "assessment",
                }
            )

    return tasks


def deduplicate_tasks(tasks: List[Dict]) -> List[Dict]:
    """
    Deduplicate tasks by (type, date, normalized title).
    Keeps the first occurrence for each key.
    """
    seen: Dict[Tuple[str, str, str], Dict] = {}
    result: List[Dict] = []

    for t in tasks:
        t_type = t.get("type", "")
        t_date = t.get("date", "")
        t_title_norm = t.get("title", "").strip().lower()

        key = (t_type, t_date, t_title_norm)
        if key in seen:
            continue

        seen[key] = t
        result.append(t)

    return result


def extract_all_tasks_from_syllabus(
    text: str,
    show_snippets: bool = False,
    assessment_components: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Main pipeline: extract snippets, analyze each, flatten into:
      - hard_deadline tasks
      - class_session events with readings

    Returns a single flat list with both types.
    """

    # 1. Extract coarse big snippets that contain dates
    big_snippets = get_date_snippets(text)
    snippets: List[str] = big_snippets

    print(f"📍 Found {len(big_snippets)} snippets containing dates\n")

    # Build assessment context string if provided
    assessment_context = None
    if assessment_components:
        lines = []
        for comp in assessment_components:
            name = comp.get("name", "Unknown")
            comp_type = comp.get("type", "assignment")
            weight = comp.get("weight_percent", 0)
            lines.append(f"- {name} ({comp_type}, {weight}% of grade)")
        assessment_context = "\n".join(lines)
        print(f"📋 Using {len(assessment_components)} assessment components for context\n")

    # Optional debugging: show snippets
    if show_snippets:
        print("=" * 80)
        print("EXTRACTED SNIPPETS:")
        print("=" * 80)
        for i, snippet in enumerate(snippets, 1):
            print(f"\n--- Snippet #{i} ---")
            print(snippet)
            print("-" * 40)
        print("\n")

    # 3. Analyze each snippet with the LLM, passing assessment context
    all_items: List[Dict] = []
    snippet_items_pairs: List[Tuple[str, List[Dict]]] = []

    for i, snippet in enumerate(snippets, 1):
        print(f"Analyzing snippet {i}/{len(snippets)}...", end="\r")
        result = analyze_snippet(snippet, assessment_context)
        if result:
            snippet_items_pairs.append((snippet, result))
            all_items.extend(result)

    print(f"\n✅ Analyzed {len(snippets)} snippets, gathered {len(all_items)} date-items\n")

    # 4. Flatten into:
    #    - hard_deadline tasks
    #    - class_session events (each with readings)
    tasks: List[Dict] = []
    class_sessions: Dict[Tuple[str, str], Dict] = {}  # (date, title) -> session

    for snippet_text, items in snippet_items_pairs:
        lower_snippet = snippet_text.lower()
        has_reading_trigger = any(rt in lower_snippet for rt in READING_TRIGGERS)
        has_hard_trigger = any(ht in lower_snippet for ht in HARD_DEADLINE_TRIGGERS)

        for item in items:
            if not isinstance(item, dict):
                print(f"⚠️  Skipping malformed item of type {type(item)}")
                continue

            kind = item.get("kind")
            date_string = item.get("date_string")

            if not date_string or kind == "ignore":
                continue

            # --- HARD DEADLINES ---
            if kind == "hard_deadline":
                if not has_hard_trigger:
                    continue

                for t in item.get("hard_deadlines", []):
                    title = t.get("title", "").strip()
                    if not title:
                        continue

                    task = {
                        "date": date_string,
                        "title": title,
                        "description": t.get("description", ""),
                        "type": t.get("type", "deadline"),
                    }
                    assessment_name = t.get("assessment_name")
                    if assessment_name:
                        task["assessment_name"] = assessment_name
                    tasks.append(task)

            # --- CLASS SESSIONS ---
            elif kind == "class_session":
                session_title = (item.get("session_title") or "").strip()
                if not session_title:
                    session_title = "Class Session"

                key = (date_string, session_title)

                if key not in class_sessions:
                    class_sessions[key] = {
                        "date": date_string,
                        "title": session_title,
                        "type": "class_session",
                        "readings": [],  # list of {title, role, reading_type}
                    }

                session = class_sessions[key]

                if has_reading_trigger:
                    # Prep tasks
                    for t in item.get("prep_tasks", []):
                        title = t.get("title", "").strip()
                        if not title:
                            continue
                        session["readings"].append(
                            {
                                "title": title,
                                "role": "prep",
                                "reading_type": t.get("type", "reading_preparatory"),
                            }
                        )

                    # Mandatory / optional tasks
                    for t in item.get("mandatory_tasks", []):
                        title = t.get("title", "").strip()
                        if not title:
                            continue
                        session["readings"].append(
                            {
                                "title": title,
                                "role": "mandatory",
                                "reading_type": t.get("type", "reading_mandatory"),
                            }
                        )
                # If no reading trigger: keep the class_session with empty readings

    # 5. Add deterministic in-class assessments as hard_deadline tasks
    inclass_tasks = extract_inclass_assessment_tasks(text)
    tasks.extend(inclass_tasks)

    # 6. Deduplicate hard_deadline-like tasks
    tasks = deduplicate_tasks(tasks)

    # 7. Merge class_sessions into final list
    all_items_flat: List[Dict] = []
    all_items_flat.extend(tasks)
    all_items_flat.extend(class_sessions.values())

    return all_items_flat


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("📅 DEADLINE EXTRACTION TEST (Snippet-Based)")
    print("=" * 80)
    print()

    # Check if file path is provided
    if len(sys.argv) < 2:
        # If no argument, look for the most recent file in uploads folder
        uploads_dir = Path(__file__).parent.parent.parent / "uploads"
        if uploads_dir.exists():
            pdf_files = sorted(
                uploads_dir.glob("*.pdf"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
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
    print("🔍 Step 1: Parsing PDF...")
    try:
        with open(file_path, "rb") as f:
            content = f.read()

        if file_path.suffix.lower() == ".pdf":
            text = parse_pdf(content)
        else:
            text = parse_text_document(content, file_path.suffix.lower())

        print(f"✅ Parsed {len(text)} characters\n")
    except Exception as e:
        print(f"❌ Error parsing file: {e}")
        sys.exit(1)

    # Extract assessment components first (optional but recommended)
    print("🤖 Step 2a: Extracting assessment components...")
    assessment_components = None
    try:
        from app.utils.test_assessment_parser import extract_assessment_components

        assessment_components = extract_assessment_components(text)
        if assessment_components:
            print(f"   Found {len(assessment_components)} graded components:")
            for comp in assessment_components:
                print(f"      - {comp['name']} ({comp['weight_percent']}%)")
        print()
    except Exception as e:
        print(f"   ⚠️  Could not extract assessments: {e}")
        print("   Continuing without assessment context...\n")

    # Extract deadlines + class sessions
    print("🤖 Step 2b: Extracting deadlines and class sessions using snippet-based approach...")
    print(f"OpenAI API Key configured: {'Yes ✅' if api_key_valid else 'No ❌'}")
    if api_key_valid:
        print(f"   Using API key: {settings.OPENAI_API_KEY[:15]}...")
    print()

    if not client:
        print("❌ Cannot proceed without OpenAI API key")
        sys.exit(1)

    # Ask if user wants to see snippets
    show_snippets_input = input(
        "🔍 Show extracted snippets for debugging? (y/n): "
    ).lower().strip()
    show_snippets = show_snippets_input == "y"
    print()

    try:
        items = extract_all_tasks_from_syllabus(
            text,
            show_snippets=show_snippets,
            assessment_components=assessment_components,
        )

        print("=" * 80)
        print(f"📋 EXTRACTED {len(items)} ITEMS (deadlines + class sessions)")
        print("=" * 80)
        print()

        if not items:
            print("⚠️  No items found.")
        else:
            for i, item in enumerate(items, 1):
                print(f"Item #{i}:")
                print(f"  📅 Date:        {item.get('date', 'N/A')}")
                print(f"  🏷️  Type:        {item.get('type', 'N/A')}")
                print(f"  📌 Title:       {item.get('title', 'N/A')}")
                if item.get("type") == "class_session":
                    readings = item.get("readings", [])
                    if readings:
                        print("  📚 Readings:")
                        for r in readings:
                            print(
                                f"    - [{r.get('role')}] {r.get('title')} "
                                f"({r.get('reading_type')})"
                            )
                else:
                    print(f"  📝 Description: {item.get('description', 'N/A')}")
                    if "assessment_name" in item:
                        print(f"  🎯 Assessment:  {item['assessment_name']}")
                print()

        print("=" * 80)
        print(f"✅ Extraction complete! Found {len(items)} items")
        print("=" * 80)

        # Option to save to JSON
        if items:
            print()
            save_option = input(
                "💾 Save results to JSON file? (y/n): "
            ).lower().strip()
            if save_option == "y":
                output_file = (
                    Path(__file__).parent.parent.parent
                    / "uploads"
                    / f"{file_path.stem}_tasks.json"
                )
                with open(output_file, "w") as f:
                    json.dump(items, f, indent=2)
                print(f"✅ Saved to: {output_file}")

    except Exception as e:
        print(f"❌ Error extracting items: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
