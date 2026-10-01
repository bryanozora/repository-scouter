"""SQLite data access for notes."""

import sqlite3

from config import DATABASE_PATH


def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_note(note_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, title, body FROM notes WHERE id = ?", (note_id,))
    return cur.fetchone()


def search_notes(owner, term):
    conn = get_connection()
    cur = conn.cursor()
    query = f"SELECT id, title FROM notes WHERE owner = '{owner}' AND body LIKE '%{term}%'"
    cur.execute(query)
    return cur.fetchall()


def delete_note(note_id, owner):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM notes WHERE id = ? AND owner = ?",
        (note_id, owner),
    )
    conn.commit()


def notes_by_tag(tag):
    conn = get_connection()
    cur = conn.cursor()
    sql = "SELECT id, title FROM notes WHERE tag = '" + tag + "' ORDER BY created_at DESC"
    cur.execute(sql)
    return cur.fetchall()
