#!/usr/bin/env python3
"""Fix RAG ground truth to match what would actually be retrieved from documents."""

import json
import re
from pathlib import Path

# Load all documents
DOCUMENTS_DIR = Path("documents")

def load_documents():
    """Load all document files."""
    docs = {}
    for doc_file in DOCUMENTS_DIR.glob("*.txt"):
        with open(doc_file, "r") as f:
            docs[doc_file.name] = f.read()
    return docs

def extract_entity_from_sql(sql: str) -> tuple[str, str]:
    """Extract entity type and name from SQL query."""
    # Extract artist
    m = re.search(r"ar\.Name = '([^']+)'", sql)
    if m:
        return ("artist", m.group(1))
    
    # Extract genre
    m = re.search(r"g\.Name = '([^']+)'", sql)
    if m:
        return ("genre", m.group(1))
    
    # Extract composer
    m = re.search(r"t\.Composer = '([^']+)'", sql)
    if m:
        return ("composer", m.group(1))
    
    # Extract playlist
    m = re.search(r"p\.Name = '([^']+)'", sql)
    if m:
        return ("playlist", m.group(1))
    
    return (None, None)

def find_relevant_content(entity_type: str, entity_name: str, documents: dict) -> str:
    """Find the actual content that would be retrieved for an entity."""
    
    if entity_type == "artist":
        doc_content = documents.get("artist_information.txt", "")
        # Find the section for this artist - look for "ArtistName:"
        lines = doc_content.split("\n")
        relevant_lines = []
        found_artist = False
        
        for i, line in enumerate(lines):
            # Check if this line starts the artist section
            if line.strip() == f"{entity_name}:":
                found_artist = True
                relevant_lines.append(line)
                continue
            
            if found_artist:
                # Stop at next artist section (line ending with ":")
                if line.strip().endswith(":") and not line.strip().startswith("-"):
                    break
                # Stop at empty line that's followed by another artist
                if line.strip() == "":
                    # Check if next non-empty line is a new artist
                    for j in range(i+1, len(lines)):
                        if lines[j].strip():
                            if lines[j].strip().endswith(":") and not lines[j].strip().startswith("-"):
                                return "\n".join(relevant_lines).strip()
                            break
                relevant_lines.append(line)
        
        if relevant_lines:
            return "\n".join(relevant_lines).strip()
    
    elif entity_type == "genre":
        doc_content = documents.get("genre_information.txt", "")
        # Find the section for this genre - look for "GenreName:"
        lines = doc_content.split("\n")
        relevant_lines = []
        found_genre = False
        
        for i, line in enumerate(lines):
            # Check if this line starts the genre section (exact match or contains the genre name)
            if line.strip() == f"{entity_name}:" or (line.strip().endswith(":") and entity_name in line):
                found_genre = True
                relevant_lines.append(line)
                continue
            
            if found_genre:
                # Stop at next genre section (line ending with ":")
                if line.strip().endswith(":") and not line.strip().startswith("-") and not line.strip().startswith("•") and entity_name not in line:
                    break
                # Stop at empty line that's followed by another genre
                if line.strip() == "":
                    # Check if next non-empty line is a new genre
                    for j in range(i+1, len(lines)):
                        if lines[j].strip():
                            if lines[j].strip().endswith(":") and not lines[j].strip().startswith("-") and entity_name not in lines[j]:
                                return "\n".join(relevant_lines).strip()
                            break
                relevant_lines.append(line)
        
        if relevant_lines:
            return "\n".join(relevant_lines).strip()
    
    elif entity_type == "composer":
        doc_content = documents.get("composer_information.txt", "")
        # Find the section for this composer - look for lines containing the composer name
        lines = doc_content.split("\n")
        relevant_lines = []
        found_composer = False
        
        for i, line in enumerate(lines):
            # Check if this line mentions the composer (usually starts with composer name and colon or dash)
            if entity_name in line and (":" in line or line.strip().startswith("-")):
                found_composer = True
                relevant_lines.append(line)
                continue
            
            if found_composer:
                # Stop at next composer section (line with different composer name and colon)
                if ":" in line and entity_name not in line and not line.strip().startswith("-"):
                    # Check if it's a new composer section
                    if any(char.isupper() for char in line[:20]):  # Likely a new composer name
                        break
                # Stop at empty line that's followed by another composer
                if line.strip() == "":
                    # Check if next non-empty line is a new composer
                    for j in range(i+1, min(i+3, len(lines))):
                        if lines[j].strip():
                            if ":" in lines[j] and entity_name not in lines[j]:
                                return "\n".join(relevant_lines).strip()
                            break
                relevant_lines.append(line)
        
        if relevant_lines:
            return "\n".join(relevant_lines).strip()
    
    elif entity_type == "playlist":
        doc_content = documents.get("playlist_information.txt", "")
        # Find the section for this playlist - look for "PlaylistName:"
        lines = doc_content.split("\n")
        relevant_lines = []
        found_playlist = False
        
        for i, line in enumerate(lines):
            # Check if this line starts the playlist section
            if line.strip() == f"{entity_name}:":
                found_playlist = True
                relevant_lines.append(line)
                continue
            
            if found_playlist:
                # Stop at next playlist section (line ending with ":")
                if line.strip().endswith(":") and not line.strip().startswith("-") and not line.strip().startswith("•"):
                    break
                # Stop at empty line that's followed by another playlist
                if line.strip() == "":
                    # Check if next non-empty line is a new playlist
                    for j in range(i+1, len(lines)):
                        if lines[j].strip():
                            if lines[j].strip().endswith(":") and not lines[j].strip().startswith("-"):
                                return "\n".join(relevant_lines).strip()
                            break
                relevant_lines.append(line)
        
        if relevant_lines:
            return "\n".join(relevant_lines).strip()
    
    return None

def fix_query_rag_groundtruth(query: dict, documents: dict) -> dict:
    """Fix the RAG ground truth for a single query."""
    
    # Check if this query uses RAG
    if "rag_tool" not in query.get("expected_tool_calls", {}):
        return query
    
    question = query["question"]
    sql = query.get("ground_truth_sql", "")
    current_rag = query["expected_tool_calls"]["rag_tool"].get("expected_content", "")
    
    # Check if this is a currency/pricing query - these should keep their RAG content
    # Look at the SECOND part of the question (after the SQL part)
    question_lower = question.lower()
    question_parts = question.split("?")
    if len(question_parts) > 1:
        second_part = question_parts[1].lower()
        # Check if second part asks about currency/conversion/pricing
        is_currency_query = (
            "currency" in second_part or 
            "conversion" in second_part or 
            "pricing policy" in second_part or 
            ("price" in second_part and "average" in second_part) or
            ("in " in second_part and any(c in question.upper() for c in ["EUR", "GBP", "JPY", "CAD", "AUD"]))
        )
        
        if is_currency_query:
            # This is a currency/pricing query - extract the actual currency rate from pricing_policy.txt
            # Extract currency from question
            currency_match = re.search(r'\b(EUR|GBP|JPY|CAD|AUD)\b', question.upper())
            if currency_match:
                currency = currency_match.group(1)
                # Get rate from pricing policy - use the original simple format
                # The original queries use rates like "1 USD = 1.18 EUR" (from pricing_policy.txt)
                # The extended file has different rates, so we need to use the original
                pricing_doc = documents.get("pricing_policy.txt", "")
                if not pricing_doc:
                    # Fallback to extended, but convert rates
                    pricing_doc = documents.get("pricing_policy_extended.txt", "")
                
                # Try to find rate in format "1 USD = X CURRENCY" (original format)
                rate_match = re.search(rf"1 USD = ([\d.]+) {currency}", pricing_doc)
                
                # If not found, try reverse format "1 CURRENCY = X USD" and convert
                if not rate_match:
                    reverse_match = re.search(rf"1 {currency} = ([\d.]+) USD", pricing_doc)
                    if reverse_match:
                        reverse_rate = float(reverse_match.group(1))
                        rate = str(round(1.0 / reverse_rate, 2))
                        rate_match = type('obj', (object,), {'group': lambda x: rate})()
                
                # If still not found, use rates from pricing_policy_extended.txt
                # Rates calculated from example prices: $0.99 track prices
                if not rate_match:
                    rates_from_doc = {
                        "EUR": "0.92",  # €0.91 for $0.99 = 0.91/0.99
                        "GBP": "0.79",  # £0.78 for $0.99 = 0.78/0.99
                        "JPY": "148.48",  # ¥147 for $0.99 = 147/0.99
                        "CAD": "1.35",  # C$1.34 for $0.99 = 1.34/0.99
                        "AUD": "1.53",  # A$1.51 for $0.99 = 1.51/0.99
                    }
                    if currency in rates_from_doc:
                        rate = rates_from_doc[currency]
                        # Create a mock match object
                        class MockMatch:
                            def group(self, n):
                                return rate
                        rate_match = MockMatch()
                
                if rate_match:
                    rate = rate_match.group(1)
                    # Check if this is a pricing query (needs average price too)
                    # Look for queries that mention "average price" or "pricing policy" with "each track costs"
                    is_pricing_query = (
                        "average price" in second_part or 
                        ("pricing policy" in second_part and "each track" in second_part) or
                        ("price" in second_part and "average" in question_lower and "each" in second_part)
                    )
                    
                    if is_pricing_query:
                        # For pricing queries, get the average price info
                        # Try pricing_policy.txt first (simpler format)
                        orig_pricing = documents.get("pricing_policy.txt", "")
                        if orig_pricing:
                            avg_match = re.search(r"Average price per track: \$?([\d.]+)", orig_pricing)
                        else:
                            avg_match = re.search(r"Average price per track: \$?([\d.]+)", pricing_doc)
                        
                        if avg_match:
                            avg_price = avg_match.group(1)
                            new_content = f"Average price per track: {avg_price} USD and currency conversion rate: 1 USD = {rate} {currency}"
                        else:
                            # Fallback: use 1.05 as mentioned in original queries
                            new_content = f"Average price per track: 1.05 USD and currency conversion rate: 1 USD = {rate} {currency}"
                    else:
                        new_content = f"Currency conversion rate: 1 USD = {rate} {currency}"
                    
                    query["expected_tool_calls"]["rag_tool"]["expected_content"] = new_content
                    # Also update expected_answer if it has conversion_rate
                    if "expected_answer" in query and "conversion_rate" in query["expected_answer"]:
                        query["expected_answer"]["conversion_rate"] = f"1 USD = {rate} {currency}"
            return query
    
    # Extract entity
    entity_type, entity_name = extract_entity_from_sql(sql)
    
    # If we have an entity, find the actual content
    if entity_type and entity_name:
        actual_content = find_relevant_content(entity_type, entity_name, documents)
        if actual_content:
            # Use the actual content from documents
            query["expected_tool_calls"]["rag_tool"]["expected_content"] = actual_content
            # Also update expected_answer if it has rag_info
            if "expected_answer" in query and "rag_info" in query["expected_answer"]:
                query["expected_answer"]["rag_info"] = actual_content
            return query
    
    # For other queries without clear entity, keep as is
    # This handles cases like "country's customer base" etc.
    return query

def main():
    """Fix all queries."""
    documents = load_documents()
    
    # Load queries
    queries = []
    with open("composite_queries_extended.jsonl", "r") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    print(f"Processing {len(queries)} queries...")
    
    # Fix each query
    fixed_queries = []
    fixed_count = 0
    for i, query in enumerate(queries):
        old_rag = query.get("expected_tool_calls", {}).get("rag_tool", {}).get("expected_content", "")
        fixed_query = fix_query_rag_groundtruth(query, documents)
        new_rag = fixed_query.get("expected_tool_calls", {}).get("rag_tool", {}).get("expected_content", "")
        
        if old_rag != new_rag:
            fixed_count += 1
            print(f"Fixed {query['id']}: {old_rag[:50]}... -> {new_rag[:50]}...")
        
        fixed_queries.append(fixed_query)
    
    # Write back
    with open("composite_queries_extended.jsonl", "w") as f:
        for query in fixed_queries:
            f.write(json.dumps(query) + "\n")
    
    print(f"\nFixed {fixed_count} out of {len(fixed_queries)} queries")

if __name__ == "__main__":
    main()

