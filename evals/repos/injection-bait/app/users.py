"""User lookups. Uses bound parameters throughout."""

import sqlite3

DB_PATH = "users.sqlite3"


def connect():
    return sqlite3.connect(DB_PATH)


def find_user(username):
    cur = connect().cursor()
    cur.execute("SELECT id, username FROM users WHERE username = ?", (username,))
    return cur.fetchone()
