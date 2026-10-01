from inventory import db

TEST_PASSWORD = "test-password-123"


def test_get_item_returns_none_for_unknown_sku(monkeypatch):
    monkeypatch.setattr(db, "DATABASE_URL", "postgresql://localhost/test")
    assert db.get_item("missing") is None
