# 🌊 TideTales

> Turning coastal and climate time-series data into clear, grounded, bilingual stories for coastal communities.

Founded and independently developed by **Punyajit Mishra**.

---

## 🎯 Purpose & Principles

TideTales is built for community climate champions, fishing and farming communities, researchers, and local decision-makers. It turns time-series observations into verified narratives in **English** and **Odia (ଓଡ଼ିଆ)** without hallucinating scientific facts or generating misleading forecasts.

### Core Boundaries
- **Strict Fact Pack First**: All mathematical facts (net change, percent change, peak, trough, range, volatility, linear fit) are computed deterministically *before* model generation. The AI model is never asked to calculate numbers.
- **Offline Deterministic Fallback**: Works without any API key out-of-the-box.
- **Fact Validation Layer**: Model responses are checked against the Fact Pack; ungrounded percentage claims are blocked and replaced with the deterministic fallback.
- **Clear Demarcation**: Explicitly distinguishes between measured observations, derived calculations, unverified user context, and AI explanations.
- **No Training Claims**: User uploads are never saved as training data. Future model and corpus integrations are isolated behind clean architectural interfaces.

---

## 🏗️ Architecture

```
TideTales/
├── package.json              # Server deps + scripts (npm run dev / start / build)
├── .env.example
├── server/                   # Express API (ES modules)
│   ├── index.js              # API routes
│   ├── services/
│   │   ├── modelProvider.js     # Gemini, Deterministic Fallback, Custom LM stub
│   │   ├── factValidator.js     # Guard against ungrounded numbers
│   │   ├── corpusInterface.js   # Future Odia corpus & TEK registry (disabled)
│   │   └── climateAssistant.js  # Grounded climate science Q&A
│   └── db/store.js           # In-memory run history (SQLite-ready)
└── client/                   # React app (Create React App)
    ├── public/index.html
    └── src/
        ├── App.jsx
        ├── demoData.js          # Illustrative Paradip demo dataset
        └── utils/
            ├── factPackEngine.js  # Deterministic arithmetic engine
            └── odiaFormatter.js   # Odia numeral utilities
```

---

## 🚀 Quickstart

### Prerequisites
- Node.js 18+

### 1. Clone & Install
```bash
git clone https://github.com/punyajitmishra/TideTales.git
cd TideTales
npm run install:all
```

### 2. Configure Environment (Optional)
```bash
cp .env.example .env
```
Add `GEMINI_API_KEY=your_key` to enable Gemini 2.5 Flash. If left empty, the app runs in offline deterministic mode.

### 3. Start Development Servers
```bash
# API on :5000, React app on :3000
npm run dev
```

Other scripts: `npm start` (API only), `npm run build` (production client build).

---

## ⚖️ Disclaimer

TideTales is an educational analysis and storytelling tool. It is **not** an official emergency alert system or predictive weather forecast. For real-time cyclone and sea-state advisories, consult the India Meteorological Department (IMD) and INCOIS.
