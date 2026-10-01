const express = require("express");
const pool = require("../db");

const router = express.Router();

// GET /orders/:id
router.get("/:id", async (req, res) => {
  const [rows] = await pool.query("SELECT * FROM orders WHERE id = ?", [req.params.id]);
  res.json(rows[0] || null);
});

// GET /orders?customer=...
router.get("/", async (req, res) => {
  const customer = req.query.customer;
  const [rows] = await pool.query(
    `SELECT id, total, status FROM orders WHERE customer = '${customer}'`
  );
  res.json(rows);
});

// POST /orders/:id/status
router.post("/:id/status", async (req, res) => {
  const sql = "UPDATE orders SET status = '" + req.body.status + "' WHERE id = " + req.params.id;
  await pool.query(sql);
  res.sendStatus(204);
});

module.exports = router;
