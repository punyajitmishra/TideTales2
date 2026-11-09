/**
 * TideTales API Server
 * Built with Express and Google AI Studio integration
 */
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { GeminiProvider, DeterministicFallbackProvider, FutureCustomModelProvider } from './services/modelProvider.js';
import { validateNarrative } from './services/factValidator.js';
import { corpusRegistry } from './services/corpusInterface.js';
import { askClimateAssistant } from './services/climateAssistant.js';
import { historyStore } from './db/store.js';

dotenv.config();

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

const apiKey = process.env.GEMINI_API_KEY;
const gemini = new GeminiProvider(apiKey);
const fallback = new DeterministicFallbackProvider();
const futureModel = new FutureCustomModelProvider();

// Health & Capability Endpoint
app.get('/api/health', async (req, res) => {
  const geminiStatus = await gemini.getStatus();
  const fallbackStatus = await fallback.getStatus();
  const customStatus = await futureModel.getStatus();
  const corpusStatus = corpusRegistry.getStatus();

  res.json({
    status: 'ok',
    app: 'TideTales',
    developer: 'Punyajit Mishra',
    providers: {
      gemini: geminiStatus,
      fallback: fallbackStatus,
      futureCustom: customStatus
    },
    corpus: corpusStatus
  });
});

// Story Narrative Generation Route
app.post('/api/generate', async (req, res) => {
  const { factPack, language = 'English', audience = 'community', length = 'short', userContext = '' } = req.body;

  if (!factPack || typeof factPack !== 'object') {
    return res.status(400).json({ error: 'Valid computed Fact Pack is required.' });
  }

  let narrative = null;
  let mode = 'deterministic_fallback';
  let validationResult = { valid: true, issues: [] };

  if (apiKey) {
    try {
      narrative = await gemini.generateNarrative({ factPack, language, audience, length, userContext });
      mode = 'gemini_2.5_flash';
      validationResult = validateNarrative(narrative, factPack);

      if (!validationResult.valid) {
        // Guard against hallucinated facts: switch to safe fallback
        console.warn('Model output failed strict fact validation. Falling back to deterministic narrative.', validationResult.issues);
        narrative = await fallback.generateNarrative({ factPack, language, audience, length, userContext });
        mode = 'deterministic_fallback_after_validation_error';
      }
    } catch (err) {
      console.error('Gemini generation error, falling back to deterministic engine:', err.message);
      narrative = await fallback.generateNarrative({ factPack, language, audience, length, userContext });
      mode = 'deterministic_fallback_error';
    }
  } else {
    // Zero API key configured: fully offline deterministic story
    narrative = await fallback.generateNarrative({ factPack, language, audience, length, userContext });
    mode = 'deterministic_fallback';
  }

  // Save to History
  const savedRecord = historyStore.saveRun({
    datasetName: factPack.datasetName,
    location: factPack.location,
    language,
    factPack,
    narrative,
    mode,
    status: 'Completed'
  });

  return res.json({
    success: true,
    mode,
    validation: validationResult,
    narrative,
    recordId: savedRecord.id
  });
});

// Climate Science Assistant Route
app.post('/api/assistant', async (req, res) => {
  const { question, language = 'English' } = req.body;
  if (!question || !question.trim()) {
    return res.status(400).json({ error: 'Question is required.' });
  }

  try {
    const result = await askClimateAssistant({ question, language, apiKey });
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: 'Failed to process inquiry', details: err.message });
  }
});

// Run History Routes
app.get('/api/history', (req, res) => {
  res.json(historyStore.getAll());
});

app.delete('/api/history/:id', (req, res) => {
  historyStore.deleteRun(req.params.id);
  res.json({ success: true });
});

app.delete('/api/history', (req, res) => {
  historyStore.clearAll();
  res.json({ success: true });
});

app.listen(port, () => {
  console.log(`🌊 TideTales server listening on port ${port}`);
  console.log(`Model mode: ${apiKey ? 'Gemini 2.5 Flash active' : 'Deterministic offline fallback active'}`);
});
