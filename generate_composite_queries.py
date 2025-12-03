#!/usr/bin/env python3
"""Generate 50 composite queries from the bottom 50 SQL queries."""

import json
import sqlite3
from typing import Dict, List, Any, Tuple
from agent.tools_api import (
    convert_from_usd, calculate_total_value, calculate_estimated_revenue,
    format_duration_hours, calculate_percentage
)

# Database connection
DB_PATH = "Chinook.db"

# Query data: (id, question, sql, expected_answer)
QUERIES = [
    (3300, "What is the average number of tracks per album for Body Count?", 
     "SELECT ROUND(AVG(track_count), 2) FROM (SELECT COUNT(*) as track_count FROM Album al JOIN Track t ON al.AlbumId = t.AlbumId JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'Body Count' GROUP BY al.AlbumId)", 
     "17.0"),
    (3301, "What is the total revenue from tracks composed by Steven Tyler, Joe Perry, Taylor Rhodes?", 
     "SELECT ROUND(SUM(il.UnitPrice * il.Quantity), 2) FROM InvoiceLine il JOIN Track t ON il.TrackId = t.TrackId WHERE t.Composer = 'Steven Tyler, Joe Perry, Taylor Rhodes'", 
     "0.99"),
    (3302, "What is the average number of tracks per album for Caetano Veloso?", 
     "SELECT ROUND(AVG(track_count), 2) FROM (SELECT COUNT(*) as track_count FROM Album al JOIN Track t ON al.AlbumId = t.AlbumId JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'Caetano Veloso' GROUP BY al.AlbumId)", 
     "10.5"),
    (3303, "What is the average number of tracks per album for Marcos Valle?", 
     "SELECT ROUND(AVG(track_count), 2) FROM (SELECT COUNT(*) as track_count FROM Album al JOIN Track t ON al.AlbumId = t.AlbumId JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'Marcos Valle' GROUP BY al.AlbumId)", 
     "17.0"),
    (3304, "What is the total revenue from tracks composed by Steven Tyler, Joe Perry, Jim Vallance?", 
     "SELECT ROUND(SUM(il.UnitPrice * il.Quantity), 2) FROM InvoiceLine il JOIN Track t ON il.TrackId = t.TrackId WHERE t.Composer = 'Steven Tyler, Joe Perry, Jim Vallance'", 
     "None"),
    (3305, "What is the average number of tracks per album for Antônio Carlos Jobim?", 
     "SELECT ROUND(AVG(track_count), 2) FROM (SELECT COUNT(*) as track_count FROM Album al JOIN Track t ON al.AlbumId = t.AlbumId JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'Antônio Carlos Jobim' GROUP BY al.AlbumId)", 
     "15.5"),
    (3306, "How many tracks were composed by Steven Tyler, Jim Vallance?", 
     "SELECT COUNT(*) FROM Track WHERE Composer = 'Steven Tyler, Jim Vallance'", 
     "2"),
    (3307, "What is the total quantity of Science Fiction tracks sold in 2012?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Science Fiction' AND strftime('%Y', i.InvoiceDate) = '2012'", 
     "1"),
    (3308, "How many tracks were composed by AC/DC?", 
     "SELECT COUNT(*) FROM Track WHERE Composer = 'AC/DC'", 
     "8"),
    (3309, "What is the total length in minutes of all tracks in albums by Aerosmith?", 
     "SELECT ROUND(SUM(t.Milliseconds) / 60000.0, 2) FROM Track t JOIN Album al ON t.AlbumId = al.AlbumId JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'Aerosmith'", 
     "73.53"),
    (3310, "What is the total revenue from MPEG audio file tracks in 2010?", 
     "SELECT ROUND(SUM(il.UnitPrice * il.Quantity), 2) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN MediaType mt ON t.MediaTypeId = mt.MediaTypeId WHERE mt.Name = 'MPEG audio file' AND strftime('%Y', i.InvoiceDate) = '2010'", 
     "373.23"),
    (3311, "What is the total length in minutes of all tracks in albums by Milton Nascimento & Bebeto?", 
     "SELECT ROUND(SUM(t.Milliseconds) / 60000.0, 2) FROM Track t JOIN Album al ON t.AlbumId = al.AlbumId JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'Milton Nascimento & Bebeto'", 
     "None"),
    (3312, "How many tracks were composed by Steven Tyler, Richie Supa?", 
     "SELECT COUNT(*) FROM Track WHERE Composer = 'Steven Tyler, Richie Supa'", 
     "1"),
    (3313, "What is the total length in minutes of all tracks in albums by BackBeat?", 
     "SELECT ROUND(SUM(t.Milliseconds) / 60000.0, 2) FROM Track t JOIN Album al ON t.AlbumId = al.AlbumId JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'BackBeat'", 
     "26.93"),
    (3314, "What is the total revenue from tracks composed by Steven Tyler, Joe Perry, Jack Blades, Tommy Shaw?", 
     "SELECT ROUND(SUM(il.UnitPrice * il.Quantity), 2) FROM InvoiceLine il JOIN Track t ON il.TrackId = t.TrackId WHERE t.Composer = 'Steven Tyler, Joe Perry, Jack Blades, Tommy Shaw'", 
     "None"),
    (3315, "What is the total quantity of Bossa Nova tracks sold in 2009?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Bossa Nova' AND strftime('%Y', i.InvoiceDate) = '2009'", 
     "1"),
    (3316, "What is the total quantity of R&B/Soul tracks sold in 2009?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'R&B/Soul' AND strftime('%Y', i.InvoiceDate) = '2009'", 
     "8"),
    (3317, "What is the total quantity of Soundtrack tracks sold in 2011?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Soundtrack' AND strftime('%Y', i.InvoiceDate) = '2011'", 
     "5"),
    (3318, "What is the total length in minutes of all tracks in albums by Gilberto Gil?", 
     "SELECT ROUND(SUM(t.Milliseconds) / 60000.0, 2) FROM Track t JOIN Album al ON t.AlbumId = al.AlbumId JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'Gilberto Gil'", 
     "128.65"),
    (3319, "What is the total quantity of Comedy tracks sold in 2012?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Comedy' AND strftime('%Y', i.InvoiceDate) = '2012'", 
     "6"),
    (3320, "What is the total length in minutes of all tracks in albums by Black Label Society?", 
     "SELECT ROUND(SUM(t.Milliseconds) / 60000.0, 2) FROM Track t JOIN Album al ON t.AlbumId = al.AlbumId JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'Black Label Society'", 
     "91.79"),
    (3321, "How many tracks were composed by Steven Tyler, Joe Perry, Desmond Child?", 
     "SELECT COUNT(*) FROM Track WHERE Composer = 'Steven Tyler, Joe Perry, Desmond Child'", 
     "3"),
    (3322, "How many tracks were composed by Bert Russell/Phil Medley?", 
     "SELECT COUNT(*) FROM Track WHERE Composer = 'Bert Russell/Phil Medley'", 
     "1"),
    (3323, "What is the total quantity of World tracks sold in 2010?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'World' AND strftime('%Y', i.InvoiceDate) = '2010'", 
     "3"),
    (3324, "What is the total quantity of Drama tracks sold in 2010?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Drama' AND strftime('%Y', i.InvoiceDate) = '2010'", 
     "9"),
    (3325, "How many tracks were composed by Apocalyptica?", 
     "SELECT COUNT(*) FROM Track WHERE Composer = 'Apocalyptica'", 
     "8"),
    (3326, "What is the total quantity of Sci Fi & Fantasy tracks sold in 2009?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Sci Fi & Fantasy' AND strftime('%Y', i.InvoiceDate) = '2009'", 
     "None"),
    (3327, "What is the total quantity of Alternative & Punk tracks sold in 2011?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Alternative & Punk' AND strftime('%Y', i.InvoiceDate) = '2011'", 
     "46"),
    (3328, "What is the total quantity of Pop tracks sold in 2013?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Pop' AND strftime('%Y', i.InvoiceDate) = '2013'", 
     "None"),
    (3329, "What is the total length in minutes of all tracks in albums by Black Sabbath?", 
     "SELECT ROUND(SUM(t.Milliseconds) / 60000.0, 2) FROM Track t JOIN Album al ON t.AlbumId = al.AlbumId JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'Black Sabbath'", 
     "81.61"),
    (3330, "What is the total quantity of Drama tracks sold in 2012?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Drama' AND strftime('%Y', i.InvoiceDate) = '2012'", 
     "9"),
    (3331, "What is the total quantity of Hip Hop/Rap tracks sold in 2013?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Hip Hop/Rap' AND strftime('%Y', i.InvoiceDate) = '2013'", 
     "4"),
    (3332, "How many tracks were composed by Steven Tyler, Joe Perry, Jack Blades, Tommy Shaw?", 
     "SELECT COUNT(*) FROM Track WHERE Composer = 'Steven Tyler, Joe Perry, Jack Blades, Tommy Shaw'", 
     "1"),
    (3333, "What is the rank of Ireland by total revenue?", 
     "SELECT rank FROM (SELECT c.Country, RANK() OVER (ORDER BY SUM(i.Total) DESC) as rank FROM Invoice i JOIN Customer c ON i.CustomerId = c.CustomerId WHERE c.Country IS NOT NULL GROUP BY c.Country) WHERE Country = 'Ireland'", 
     "11"),
    (3334, "What is the total quantity of Drama tracks sold in 2013?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Drama' AND strftime('%Y', i.InvoiceDate) = '2013'", 
     "5"),
    (3335, "What is the average number of tracks per album for BackBeat?", 
     "SELECT ROUND(AVG(track_count), 2) FROM (SELECT COUNT(*) as track_count FROM Album al JOIN Track t ON al.AlbumId = t.AlbumId JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'BackBeat' GROUP BY al.AlbumId)", 
     "12.0"),
    (3336, "What is the total revenue from tracks composed by Audioslave/Chris Cornell?", 
     "SELECT ROUND(SUM(il.UnitPrice * il.Quantity), 2) FROM InvoiceLine il JOIN Track t ON il.TrackId = t.TrackId WHERE t.Composer = 'Audioslave/Chris Cornell'", 
     "5.94"),
    (3337, "What is the total length in minutes of all tracks in albums by Led Zeppelin?", 
     "SELECT ROUND(SUM(t.Milliseconds) / 60000.0, 2) FROM Track t JOIN Album al ON t.AlbumId = al.AlbumId JOIN Artist ar ON al.ArtistId = ar.ArtistId WHERE ar.Name = 'Led Zeppelin'", 
     "668.69"),
    (3338, "What is the total length in minutes of the Grunge playlist?", 
     "SELECT ROUND(SUM(t.Milliseconds) / 60000.0, 2) FROM Playlist p JOIN PlaylistTrack pt ON p.PlaylistId = pt.PlaylistId JOIN Track t ON pt.TrackId = t.TrackId WHERE p.Name = 'Grunge'", 
     "68.7"),
    (3339, "How many tracks were composed by F. Baltes, S. Kaufman, U. Dirkscneider & W. Hoffman?", 
     "SELECT COUNT(*) FROM Track WHERE Composer = 'F. Baltes, S. Kaufman, U. Dirkscneider & W. Hoffman'", 
     "1"),
    (3340, "What is the total quantity of Soundtrack tracks sold in 2010?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Soundtrack' AND strftime('%Y', i.InvoiceDate) = '2010'", 
     "4"),
    (3341, "What is the rank of Spain by total revenue?", 
     "SELECT rank FROM (SELECT c.Country, RANK() OVER (ORDER BY SUM(i.Total) DESC) as rank FROM Invoice i JOIN Customer c ON i.CustomerId = c.CustomerId WHERE c.Country IS NOT NULL GROUP BY c.Country) WHERE Country = 'Spain'", 
     "23"),
    (3342, "What is the total quantity of Rock tracks sold in 2011?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Rock' AND strftime('%Y', i.InvoiceDate) = '2011'", 
     "158"),
    (3343, "What is the total quantity of Science Fiction tracks sold in 2011?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Science Fiction' AND strftime('%Y', i.InvoiceDate) = '2011'", 
     "2"),
    (3344, "What is the total length in minutes of the Classical 101 - The Basics playlist?", 
     "SELECT ROUND(SUM(t.Milliseconds) / 60000.0, 2) FROM Playlist p JOIN PlaylistTrack pt ON p.PlaylistId = pt.PlaylistId JOIN Track t ON pt.TrackId = t.TrackId WHERE p.Name = 'Classical 101 - The Basics'", 
     "124.0"),
    (3345, "What is the total quantity of Rock And Roll tracks sold in 2009?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Rock And Roll' AND strftime('%Y', i.InvoiceDate) = '2009'", 
     "1"),
    (3346, "What is the total quantity of Latin tracks sold in 2013?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Latin' AND strftime('%Y', i.InvoiceDate) = '2013'", 
     "80"),
    (3347, "What is the total quantity of Heavy Metal tracks sold in 2009?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Heavy Metal' AND strftime('%Y', i.InvoiceDate) = '2009'", 
     "4"),
    (3348, "What is the total quantity of R&B/Soul tracks sold in 2012?", 
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Invoice i ON il.InvoiceId = i.InvoiceId JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'R&B/Soul' AND strftime('%Y', i.InvoiceDate) = '2012'", 
     "10"),
    (3349, "What is the total revenue generated from Easy Listening tracks?", 
     "SELECT ROUND(SUM(il.UnitPrice * il.Quantity), 2) FROM InvoiceLine il JOIN Track t ON il.TrackId = t.TrackId JOIN Genre g ON t.GenreId = g.GenreId WHERE g.Name = 'Easy Listening'", 
     "9.9"),
]


def execute_sql(sql: str) -> Any:
    """Execute SQL and return result."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        conn.close()


def parse_answer(answer: str) -> float:
    """Parse answer string to float, handling 'None'."""
    if answer == "None" or answer is None:
        return None
    try:
        return float(answer)
    except:
        return None


def extract_entity_from_sql(sql: str) -> tuple[str, str]:
    """Extract entity type and name from SQL query."""
    import re
    
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


def get_rag_content_for_entity(entity_type: str, entity_name: str) -> str:
    """Get appropriate RAG content for an entity."""
    if entity_type == "artist":
        if entity_name in ["Body Count", "Aerosmith", "Led Zeppelin", "Black Sabbath", "BackBeat", 
                          "Black Label Society", "Caetano Veloso", "Marcos Valle", "Antônio Carlos Jobim",
                          "Gilberto Gil", "Milton Nascimento & Bebeto"]:
            return f"Artist information: {entity_name} is featured in the store's catalog. The store maintains detailed information about all artists including discography, genre classifications, and customer preferences."
        return "The store maintains detailed information about all artists in our catalog, including discography, genre classifications, and customer preferences."
    
    elif entity_type == "genre":
        genre_info = {
            "Science Fiction": "Science Fiction genre features soundtracks and music inspired by science fiction themes. Includes electronic and ambient tracks. Popular among fans of sci-fi movies and games.",
            "Alternative & Punk": "Alternative & Punk is one of the most popular genres in the store. Features independent and alternative rock bands. Strong sales, especially among younger demographics.",
            "Heavy Metal": "Heavy Metal features hard-hitting metal tracks. Popular among metal enthusiasts. Features both classic and modern metal bands.",
            "Rock And Roll": "Rock And Roll is classic rock and roll from the 50s and 60s. Foundational genre for rock music. Popular among classic rock fans.",
            "Bossa Nova": "Bossa Nova is a Brazilian music genre with smooth, sophisticated sound. Popular among jazz and world music fans.",
            "Comedy": "Comedy genre includes comedy albums and spoken word tracks. Includes stand-up comedy and comedy music. Tracks are typically longer than music tracks.",
            "Drama": "Drama genre features soundtracks and music from dramatic productions. Includes orchestral and cinematic music. Popular among fans of film and theater.",
            "Soundtrack": "Soundtrack genre includes music from movies, TV shows, and games. Very popular genre. Includes both original scores and featured songs.",
            "World": "World music genre features music from around the globe. Diverse collection of international music. Popular among world music enthusiasts.",
            "Hip Hop/Rap": "Hip Hop/Rap is an urban music genre. Features both classic and modern hip hop. Strong following among younger demographics.",
            "Easy Listening": "Easy Listening features relaxing, ambient music. Popular for background music. Features smooth jazz, instrumental, and soft rock.",
            "Rock": "Rock is the most popular genre with hundreds of tracks. Rock and Pop genres typically have the highest sales volumes.",
            "Latin": "Latin music features music from Latin American artists. Latin music has seen significant growth in recent years, especially in 2013.",
            "Pop": "Pop features popular music and top 40 hits. Rock and Pop genres typically have the highest sales volumes.",
            "R&B/Soul": "R&B/Soul features rhythm and blues and soul music. Popular among fans of urban and contemporary music."
        }
        return genre_info.get(entity_name, f"Genre information: {entity_name} is available in the store's catalog. The store maintains a balanced catalog across all genres.")
    
    elif entity_type == "composer":
        composer_info = {
            "Steven Tyler, Joe Perry": "Steven Tyler and Joe Perry are the primary songwriting team for Aerosmith. Known for creating many of the band's biggest hits. Their compositions span multiple decades.",
            "Steven Tyler, Joe Perry, Taylor Rhodes": "Extended songwriting collaboration that created several Aerosmith tracks. Known for their rock anthems and ballads.",
            "Steven Tyler, Joe Perry, Jim Vallance": "Songwriting collaboration that created tracks blending rock with pop sensibilities. Popular among fans of 80s and 90s rock.",
            "Steven Tyler, Joe Perry, Jack Blades, Tommy Shaw": "Cross-band collaboration combining talents from Aerosmith and other rock bands. Created unique rock compositions.",
            "Steven Tyler, Jim Vallance": "Songwriting partnership known for creating memorable rock tracks. Their compositions are featured in multiple albums.",
            "Audioslave/Chris Cornell": "Compositions by Chris Cornell, lead singer of Audioslave. Known for powerful vocals and hard rock sound. Tracks are popular among alternative and hard rock fans.",
            "AC/DC": "The band members themselves compose most of their tracks. Known for their distinctive hard rock sound. Their compositions are instantly recognizable."
        }
        return composer_info.get(entity_name, f"Composer information: {entity_name} is credited on tracks in the store's catalog. The store maintains accurate information about songwriters and composers.")
    
    elif entity_type == "playlist":
        playlist_info = {
            "Grunge": "Grunge playlist features tracks from the grunge movement of the 1990s. Includes artists like Nirvana, Soundgarden, Alice in Chains, and Pearl Jam. Typically contains 10-20 tracks. Total duration approximately 60-70 minutes.",
            "Classical 101 - The Basics": "Classical 101 - The Basics is a curated introduction to classical music. Features essential classical compositions. Designed for newcomers to classical music. Contains approximately 20-30 tracks. Total duration around 2 hours."
        }
        return playlist_info.get(entity_name, f"Playlist information: {entity_name} is a curated playlist in the store. Playlists are organized by genre, mood, era, and theme.")
    
    return "The store maintains comprehensive information about all artists, albums, tracks, genres, and playlists in our catalog."


def generate_composite_query(query_id: int, base_question: str, sql: str, expected_answer: str) -> Dict[str, Any]:
    """Generate a composite query with augmentations."""
    
    sql_result = execute_sql(sql)
    expected_value = parse_answer(expected_answer)
    
    # Extract entity from SQL for RAG queries
    entity_type, entity_name = extract_entity_from_sql(sql)
    
    # Skip queries with None results for now (we can handle them differently)
    if sql_result is None or expected_value is None:
        # For None results, add RAG info about the entity if available
        if entity_type and entity_name:
            rag_content = get_rag_content_for_entity(entity_type, entity_name)
            entity_question = {
                "artist": "What information does the store have about this artist?",
                "genre": "What information does the store have about this genre?",
                "composer": "What information does the store have about this composer?",
                "playlist": "What information does the store have about this playlist?"
            }.get(entity_type, "What information does the store have about this?")
            
            return {
                "id": f"comp_{query_id}",
                "question": f"{base_question} {entity_question}",
                "requires_tools": ["sql_tool", "rag_tool"],
                "base_query_id": query_id,
                "notes": f"Query + {entity_type} info from RAG",
                "expected_answer": {"result": expected_value, "rag_info": rag_content},
                "ground_truth_sql": sql,
                "expected_tool_calls": {
                    "sql_tool": {"expected_result_value": expected_value},
                    "rag_tool": {"expected_content": rag_content},
                    "expected_plan": {
                        "steps": [
                            {"id": 1, "description": "Use sql_tool to get required information", "action_type": "tool_call", "tool": "sql_tool"},
                            {"id": 2, "description": "Use rag_tool to get required information", "action_type": "tool_call", "tool": "rag_tool"},
                            {"id": 3, "description": "Generate final answer", "action_type": "answer", "tool": None}
                        ]
                    }
                }
            }
        else:
            # For rank queries or other queries without clear entity
            if "rank" in base_question.lower():
                return {
                    "id": f"comp_{query_id}",
                    "question": f"{base_question} What information does the store have about this country's customer base?",
                    "requires_tools": ["sql_tool", "rag_tool"],
                    "base_query_id": query_id,
                    "notes": "Rank query + company info from RAG",
                    "expected_answer": {"rank": expected_value, "rag_info": "Global customer base, customers from over 20 countries"},
                    "ground_truth_sql": sql,
                    "expected_tool_calls": {
                        "sql_tool": {"expected_result_value": expected_value},
                        "rag_tool": {"expected_content": "Global customer base, customers from over 20 countries"},
                        "expected_plan": {
                            "steps": [
                                {"id": 1, "description": "Use sql_tool to get required information", "action_type": "tool_call", "tool": "sql_tool"},
                                {"id": 2, "description": "Use rag_tool to get required information", "action_type": "tool_call", "tool": "rag_tool"},
                                {"id": 3, "description": "Generate final answer", "action_type": "answer", "tool": None}
                            ]
                        }
                    }
                }
            # Default for None results without entity
            return {
                "id": f"comp_{query_id}",
                "question": f"{base_question} What information does the store have about this?",
                "requires_tools": ["sql_tool", "rag_tool"],
                "base_query_id": query_id,
                "notes": "Query + store info from RAG",
                "expected_answer": {"result": expected_value, "rag_info": "Digital Music Retailer, Per-track and per-album sales, Global customer base"},
                "ground_truth_sql": sql,
                "expected_tool_calls": {
                    "sql_tool": {"expected_result_value": expected_value},
                    "rag_tool": {"expected_content": "Digital Music Retailer, Per-track and per-album sales, Global customer base"},
                    "expected_plan": {
                        "steps": [
                            {"id": 1, "description": "Use sql_tool to get required information", "action_type": "tool_call", "tool": "sql_tool"},
                            {"id": 2, "description": "Use rag_tool to get required information", "action_type": "tool_call", "tool": "rag_tool"},
                            {"id": 3, "description": "Generate final answer", "action_type": "answer", "tool": None}
                        ]
                    }
                }
            }
    
    # Revenue queries - add currency conversion
    if "revenue" in base_question.lower() and expected_value > 0:
        currencies = ["EUR", "GBP", "CAD", "AUD", "JPY"]
        currency = currencies[query_id % len(currencies)]
        converted = convert_from_usd(expected_value, currency)
        rate = {"EUR": 1.18, "GBP": 1.37, "CAD": 1.25, "AUD": 1.35, "JPY": 110.0}[currency]
        
        return {
            "id": f"comp_{query_id}",
            "question": f"{base_question} What would that amount be in {currency} based on the store's currency policy?",
            "requires_tools": ["sql_tool", "rag_tool", "api_tool"],
            "base_query_id": query_id,
            "notes": "Revenue query + currency conversion",
            "expected_answer": {
                "revenue_usd": expected_value,
                f"revenue_{currency.lower()}": round(converted, 2),
                "conversion_rate": f"1 USD = {rate} {currency}"
            },
            "ground_truth_sql": sql,
            "expected_tool_calls": {
                "sql_tool": {"expected_result_value": expected_value},
                "rag_tool": {"expected_content": f"Currency conversion rate: 1 USD = {rate} {currency}"},
                "api_tool": {"expected_calls": [{"tool": "convert_currency_from_usd", "input": {"amount_usd": expected_value, "target_currency": currency}}]},
                "expected_plan": {
                    "steps": [
                        {"id": 1, "description": "Use sql_tool to get required information", "action_type": "tool_call", "tool": "sql_tool"},
                        {"id": 2, "description": "Use rag_tool to get required information", "action_type": "tool_call", "tool": "rag_tool"},
                        {"id": 3, "description": "Use api_tool to get required information", "action_type": "tool_call", "tool": "api_tool"},
                        {"id": 4, "description": "Generate final answer", "action_type": "answer", "tool": None}
                    ]
                }
            }
        }
    
    # Quantity queries - add pricing calculation
    if "quantity" in base_question.lower() and expected_value > 0:
        estimated_value = calculate_total_value(expected_value, 1.05)  # Average price per track
        currencies = ["EUR", "GBP", "CAD"]
        currency = currencies[query_id % len(currencies)]
        converted = convert_from_usd(estimated_value, currency)
        rate = {"EUR": 1.18, "GBP": 1.37, "CAD": 1.25}[currency]
        
        return {
            "id": f"comp_{query_id}",
            "question": f"{base_question} What would be the estimated total value in {currency} if each track costs the average price according to the store's pricing policy?",
            "requires_tools": ["sql_tool", "rag_tool", "api_tool"],
            "base_query_id": query_id,
            "notes": "Quantity query + pricing policy + currency conversion",
            "expected_answer": {
                "quantity": int(expected_value),
                "estimated_value_usd": estimated_value,
                f"estimated_value_{currency.lower()}": round(converted, 2),
                "average_price_per_track": 1.05,
                "conversion_rate": f"1 USD = {rate} {currency}"
            },
            "ground_truth_sql": sql,
            "expected_tool_calls": {
                "sql_tool": {"expected_result_value": expected_value},
                "rag_tool": {"expected_content": f"Average price per track: 1.05 USD and currency conversion rate: 1 USD = {rate} {currency}"},
                "api_tool": {"expected_calls": [
                    {"tool": "calculate_total_value", "input": {"quantity": expected_value, "unit_price": 1.05}},
                    {"tool": "convert_currency_from_usd", "input": {"amount_usd": estimated_value, "target_currency": currency}}
                ]},
                "expected_plan": {
                    "steps": [
                        {"id": 1, "description": "Use sql_tool to get required information", "action_type": "tool_call", "tool": "sql_tool"},
                        {"id": 2, "description": "Use rag_tool to get required information", "action_type": "tool_call", "tool": "rag_tool"},
                        {"id": 3, "description": "Use api_tool to get required information", "action_type": "tool_call", "tool": "api_tool"},
                        {"id": 4, "description": "Generate final answer", "action_type": "answer", "tool": None}
                    ]
                }
            }
        }
    
    # Length/duration queries - add hours format
    if "length" in base_question.lower() or "minutes" in base_question.lower():
        hours_info = format_duration_hours(expected_value)
        
        return {
            "id": f"comp_{query_id}",
            "question": f"{base_question} How many hours and minutes is that?",
            "requires_tools": ["sql_tool", "api_tool"],
            "base_query_id": query_id,
            "notes": "Duration query + time formatting",
            "expected_answer": {
                "minutes": expected_value,
                "hours": hours_info["hours"],
                "remaining_minutes": hours_info["minutes"],
                "formatted": hours_info["formatted"]
            },
            "ground_truth_sql": sql,
            "expected_tool_calls": {
                "sql_tool": {"expected_result_value": expected_value},
                "api_tool": {"expected_calls": [{"tool": "format_duration_hours", "input": {"minutes": expected_value}}]},
                "expected_plan": {
                    "steps": [
                        {"id": 1, "description": "Use sql_tool to get required information", "action_type": "tool_call", "tool": "sql_tool"},
                        {"id": 2, "description": "Use api_tool to get required information", "action_type": "tool_call", "tool": "api_tool"},
                        {"id": 3, "description": "Generate final answer", "action_type": "answer", "tool": None}
                    ]
                }
            }
        }
    
    # Count queries - add value estimation
    if "how many" in base_question.lower() and "count" not in base_question.lower():
        estimated_value = calculate_total_value(expected_value, 1.05)
        currencies = ["GBP", "EUR"]
        currency = currencies[query_id % len(currencies)]
        converted = convert_from_usd(estimated_value, currency)
        rate = {"EUR": 1.18, "GBP": 1.37}[currency]
        
        return {
            "id": f"comp_{query_id}",
            "question": f"{base_question} What would be the estimated total value in {currency} if each track costs the average price according to the store's pricing policy?",
            "requires_tools": ["sql_tool", "rag_tool", "api_tool"],
            "base_query_id": query_id,
            "notes": "Count query + pricing policy + currency conversion",
            "expected_answer": {
                "count": int(expected_value),
                "estimated_value_usd": estimated_value,
                f"estimated_value_{currency.lower()}": round(converted, 2),
                "average_price_per_track": 1.05,
                "conversion_rate": f"1 USD = {rate} {currency}"
            },
            "ground_truth_sql": sql,
            "expected_tool_calls": {
                "sql_tool": {"expected_result_value": expected_value},
                "rag_tool": {"expected_content": f"Average price per track: 1.05 USD and currency conversion rate: 1 USD = {rate} {currency}"},
                "api_tool": {"expected_calls": [
                    {"tool": "calculate_total_value", "input": {"quantity": expected_value, "unit_price": 1.05}},
                    {"tool": "convert_currency_from_usd", "input": {"amount_usd": estimated_value, "target_currency": currency}}
                ]},
                "expected_plan": {
                    "steps": [
                        {"id": 1, "description": "Use sql_tool to get required information", "action_type": "tool_call", "tool": "sql_tool"},
                        {"id": 2, "description": "Use rag_tool to get required information", "action_type": "tool_call", "tool": "rag_tool"},
                        {"id": 3, "description": "Use api_tool to get required information", "action_type": "tool_call", "tool": "api_tool"},
                        {"id": 4, "description": "Generate final answer", "action_type": "answer", "tool": None}
                    ]
                }
            }
        }
    
    # Average queries - add RAG info about the artist/entity
    if "average" in base_question.lower():
        if entity_type and entity_name:
            rag_content = get_rag_content_for_entity(entity_type, entity_name)
            entity_question = {
                "artist": "What information does the store have about this artist?",
                "genre": "What information does the store have about this genre?",
                "composer": "What information does the store have about this composer?",
                "playlist": "What information does the store have about this playlist?"
            }.get(entity_type, "What information does the store have about this?")
            
            return {
                "id": f"comp_{query_id}",
                "question": f"{base_question} {entity_question}",
                "requires_tools": ["sql_tool", "rag_tool"],
                "base_query_id": query_id,
                "notes": f"Average query + {entity_type} info from RAG",
                "expected_answer": {
                    "average": expected_value,
                    "rag_info": rag_content
                },
                "ground_truth_sql": sql,
                "expected_tool_calls": {
                    "sql_tool": {"expected_result_value": expected_value},
                    "rag_tool": {"expected_content": rag_content},
                    "expected_plan": {
                        "steps": [
                            {"id": 1, "description": "Use sql_tool to get required information", "action_type": "tool_call", "tool": "sql_tool"},
                            {"id": 2, "description": "Use rag_tool to get required information", "action_type": "tool_call", "tool": "rag_tool"},
                            {"id": 3, "description": "Generate final answer", "action_type": "answer", "tool": None}
                        ]
                    }
                }
            }
        # Fallback for average queries without entity
        return {
            "id": f"comp_{query_id}",
            "question": f"{base_question} What information does the store have about this?",
            "requires_tools": ["sql_tool", "rag_tool"],
            "base_query_id": query_id,
            "notes": "Average query + store info from RAG",
            "expected_answer": {
                "average": expected_value,
                "rag_info": "Digital Music Retailer, Per-track and per-album sales, Global customer base"
            },
            "ground_truth_sql": sql,
            "expected_tool_calls": {
                "sql_tool": {"expected_result_value": expected_value},
                "rag_tool": {"expected_content": "Digital Music Retailer, Per-track and per-album sales, Global customer base"},
                "expected_plan": {
                    "steps": [
                        {"id": 1, "description": "Use sql_tool to get required information", "action_type": "tool_call", "tool": "sql_tool"},
                        {"id": 2, "description": "Use rag_tool to get required information", "action_type": "tool_call", "tool": "rag_tool"},
                        {"id": 3, "description": "Generate final answer", "action_type": "answer", "tool": None}
                    ]
                }
            }
        }
    
    # Default: SQL + RAG - use entity if available
    if entity_type and entity_name:
        rag_content = get_rag_content_for_entity(entity_type, entity_name)
        entity_question = {
            "artist": "What information does the store have about this artist?",
            "genre": "What information does the store have about this genre?",
            "composer": "What information does the store have about this composer?",
            "playlist": "What information does the store have about this playlist?"
        }.get(entity_type, "What information does the store have about this?")
        
        return {
            "id": f"comp_{query_id}",
            "question": f"{base_question} {entity_question}",
            "requires_tools": ["sql_tool", "rag_tool"],
            "base_query_id": query_id,
            "notes": f"Query + {entity_type} info from RAG",
            "expected_answer": {
                "result": expected_value,
                "rag_info": rag_content
            },
            "ground_truth_sql": sql,
            "expected_tool_calls": {
                "sql_tool": {"expected_result_value": expected_value},
                "rag_tool": {"expected_content": rag_content},
                "expected_plan": {
                    "steps": [
                        {"id": 1, "description": "Use sql_tool to get required information", "action_type": "tool_call", "tool": "sql_tool"},
                        {"id": 2, "description": "Use rag_tool to get required information", "action_type": "tool_call", "tool": "rag_tool"},
                        {"id": 3, "description": "Generate final answer", "action_type": "answer", "tool": None}
                    ]
                }
            }
        }
    
    # Final fallback
    return {
        "id": f"comp_{query_id}",
        "question": f"{base_question} What information does the store have about this?",
        "requires_tools": ["sql_tool", "rag_tool"],
        "base_query_id": query_id,
        "notes": "Query + store info from RAG",
        "expected_answer": {
            "result": expected_value,
            "rag_info": "Digital Music Retailer, Per-track and per-album sales, Global customer base"
        },
        "ground_truth_sql": sql,
        "expected_tool_calls": {
            "sql_tool": {"expected_result_value": expected_value},
            "rag_tool": {"expected_content": "Digital Music Retailer, Per-track and per-album sales, Global customer base"},
            "expected_plan": {
                "steps": [
                    {"id": 1, "description": "Use sql_tool to get required information", "action_type": "tool_call", "tool": "sql_tool"},
                    {"id": 2, "description": "Use rag_tool to get required information", "action_type": "tool_call", "tool": "rag_tool"},
                    {"id": 3, "description": "Generate final answer", "action_type": "answer", "tool": None}
                ]
            }
        }
    }


def main():
    """Generate all composite queries."""
    composite_queries = []
    
    for query_id, question, sql, expected_answer in QUERIES:
        try:
            composite = generate_composite_query(query_id, question, sql, expected_answer)
            composite_queries.append(composite)
            print(f"Generated comp_{query_id}: {composite['question'][:60]}...")
        except Exception as e:
            print(f"Error generating comp_{query_id}: {e}")
            continue
    
    # Write to file
    output_file = "composite_queries_extended.jsonl"
    with open(output_file, "w") as f:
        for query in composite_queries:
            f.write(json.dumps(query) + "\n")
    
    print(f"\nGenerated {len(composite_queries)} composite queries in {output_file}")


if __name__ == "__main__":
    main()

