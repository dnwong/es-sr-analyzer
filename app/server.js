const express = require("express");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, "public")));

const CHART_PATH = "/tmp/es_sr_analysis.png";
const JSON_PATH  = "/tmp/sr_results.json";
const PY_SCRIPT  = "/app/sr_analyzer.py";
const TIMEOUT_MS = 120_000; // 2 min max for data fetch + analysis

// Serve chart image
app.get("/chart", (req, res) => {
  if (fs.existsSync(CHART_PATH)) {
    res.setHeader("Cache-Control", "no-store");
    res.sendFile(CHART_PATH);
  } else {
    res.status(404).json({ error: "No chart generated yet" });
  }
});

// Run the analyzer
app.post("/analyze", (req, res) => {
  const {
    symbol      = "ES=F",
    interval    = "5m",
    days        = "3",
    or_minutes  = "30",
    pivot_order = "5",
    tolerance   = "2.0",
  } = req.body;

  // Input validation
  const allowedIntervals = ["1m", "2m", "5m", "15m"];
  if (!allowedIntervals.includes(interval))
    return res.status(400).json({ error: "Invalid interval" });
  if (isNaN(days) || Number(days) < 1 || Number(days) > 10)
    return res.status(400).json({ error: "Days must be 1–10" });

  const args = [
    PY_SCRIPT,
    "--symbol",      symbol,
    "--interval",    interval,
    "--days",        String(days),
    "--or-minutes",  String(or_minutes),
    "--pivot-order", String(pivot_order),
    "--tolerance",   String(tolerance),
    "--json-out",    JSON_PATH,
    "--chart-out",   CHART_PATH,
    // no --no-chart so the chart always renders
  ];

  let stdout = "";
  let stderr = "";
  let timedOut = false;

  const py = spawn("python3", args);

  const timer = setTimeout(() => {
    timedOut = true;
    py.kill();
    res.status(504).json({ error: "Analysis timed out after 2 minutes" });
  }, TIMEOUT_MS);

  py.stdout.on("data", (d) => { stdout += d.toString(); });
  py.stderr.on("data", (d) => { stderr += d.toString(); });

  py.on("close", (code) => {
    clearTimeout(timer);
    if (timedOut) return;

    if (code !== 0) {
      console.error("Python error:\n", stderr);
      return res.status(500).json({
        error: stderr.split("\n").filter(Boolean).pop() || "Analyzer failed"
      });
    }

    try {
      const results = JSON.parse(fs.readFileSync(JSON_PATH, "utf8"));
      res.json(results);
    } catch (e) {
      res.status(500).json({ error: "Failed to parse results: " + e.message });
    }
  });

  py.on("error", (err) => {
    clearTimeout(timer);
    res.status(500).json({ error: "Failed to start Python: " + err.message });
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, "0.0.0.0", () =>
  console.log(`ES S/R Analyzer running on http://0.0.0.0:${PORT}`)
);
