const express = require("express");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, "public")));

// Serve the chart image if it exists
app.get("/chart", (req, res) => {
  const chartPath = "/tmp/es_sr_analysis.png";
  if (fs.existsSync(chartPath)) {
    res.sendFile(chartPath);
  } else {
    res.status(404).json({ error: "No chart generated yet" });
  }
});

// Run the analyzer
app.post("/analyze", (req, res) => {
  const {
    symbol = "ES=F",
    interval = "5m",
    days = "3",
    or_minutes = "30",
    pivot_order = "5",
    tolerance = "2.0",
  } = req.body;

  // Validate inputs
  const allowedIntervals = ["1m", "2m", "5m", "15m"];
  if (!allowedIntervals.includes(interval)) {
    return res.status(400).json({ error: "Invalid interval" });
  }
  if (isNaN(days) || days < 1 || days > 10) {
    return res.status(400).json({ error: "Days must be 1–10" });
  }

  const args = [
    "/app/sr_analyzer.py",
    "--symbol", symbol,
    "--interval", interval,
    "--days", String(days),
    "--or-minutes", String(or_minutes),
    "--pivot-order", String(pivot_order),
    "--tolerance", String(tolerance),
    "--no-chart",
    "--json-out", "/tmp/sr_results.json",
    "--chart-out", "/tmp/es_sr_analysis.png",
  ];

  const py = spawn("python3", args);
  let stderr = "";
  py.stderr.on("data", (d) => { stderr += d.toString(); });

  py.on("close", (code) => {
    if (code !== 0) {
      return res.status(500).json({ error: stderr || "Analyzer failed" });
    }
    try {
      const results = JSON.parse(fs.readFileSync("/tmp/sr_results.json", "utf8"));
      res.json(results);
    } catch (e) {
      res.status(500).json({ error: "Failed to parse results" });
    }
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`ES S/R Analyzer running on http://localhost:${PORT}`));
