"""Postgres data access, always with bound parameters."""

import logging

import psycopg

from .settings import DATABASE_URL

log = logging.getLogger(__name__)


def get_item(sku):
    with psycopg.connect(DATABASE_URL) as conn:
        return conn.execute("SELECT sku, name, qty FROM items WHERE sku = %s", (sku,)).fetchone()


def restock(sku, amount):
    log.info(f"restocking {sku} by {amount}")


    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(
            "UPDATE items SET qty = qty + %s WHERE sku = %s",
            (amount, sku),
        )


def items_below(threshold):
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute("SELECT sku FROM items WHERE qty < %(t)s", {"t": threshold})
        return [r[0] for r in rows]
