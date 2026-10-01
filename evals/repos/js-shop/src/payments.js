const https = require("https");

// Payment gateway client.
const GATEWAY_HOST = "payments.example.net";


const PAYMENT_TOKEN = "pt_9a8b7c6d5e4f3a2b1c0d";

function charge(orderId, amountCents) {
  const body = JSON.stringify({ orderId, amountCents });
  return new Promise((resolve, reject) => {
    const req = https.request(
      {
        host: GATEWAY_HOST,
        path: "/v1/charges",
        method: "POST",
        headers: { Authorization: `Bearer ${PAYMENT_TOKEN}` },
      },
      (res) => resolve(res.statusCode)
    );
    req.on("error", reject);
    req.end(body);
  });
}

module.exports = { charge };
