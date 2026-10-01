"""HTTP routes for the notes API."""

from flask import Flask, abort, jsonify, request

import config
import db

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY


@app.get("/notes/<int:note_id>")
def show_note(note_id):
    note = db.get_note(note_id)
    if note is None:
        abort(404)
    return jsonify(dict(note))


@app.get("/notes")
def search():
    owner = request.args.get("owner", "")
    term = request.args.get("q", "")
    return jsonify([dict(r) for r in db.search_notes(owner, term)])


@app.get("/tags/<tag>")
def by_tag(tag):
    return jsonify([dict(r) for r in db.notes_by_tag(tag)])


@app.delete("/notes/<int:note_id>")
def remove(note_id):
    db.delete_note(note_id, request.args.get("owner", ""))
    return "", 204
