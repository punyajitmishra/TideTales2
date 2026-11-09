/**
 * Model Provider Abstraction for TideTales
 * Founded and independently developed by Punyajit Mishra.
 */
import { GoogleGenAI } from '@google/genai';

export class BaseModelProvider {
  async generateNarrative(input) {
    throw new Error('generateNarrative must be implemented by provider');
  }
  async getStatus() {
    return { available: false, name: 'base' };
  }
}

/**
 * 1. Gemini Server-Side Provider
 */
export class GeminiProvider extends BaseModelProvider {
  constructor(apiKey) {
    super();
    this.apiKey = apiKey;
    this.ai = apiKey ? new GoogleGenAI({ apiKey }) : null;
  }

  async getStatus() {
    return {
      available: Boolean(this.apiKey),
      name: 'Gemini 2.5 Flash (Server-Side)',
      mode: this.apiKey ? 'active' : 'unconfigured_fallback'
    };
  }

  async generateNarrative({ factPack, language = 'English', audience = 'community', length = 'short', userContext = '' }) {
    if (!this.ai) {
      throw new Error('GEMINI_API_KEY not configured. Use deterministic fallback.');
    }

    const systemInstruction = `
You are TideTales, a careful climate-data storyteller.
Use only the supplied Fact Pack and clearly labeled user-provided context.
Do not invent numbers, dates, units, sources, causes, impacts, forecasts, folklore, quotes, or community knowledge.
Do not present correlation as causation.
If the supplied facts do not support a conclusion, say so plainly.
Preserve all quantities and units exactly. Do not add a statistic unless it appears in the Fact Pack.
Write in the selected language (${language}) and audience style (${audience}).
If Odia wording or a cultural expression is uncertain, use simple, respectful language and do not invent a local idiom or cultural reference.

Respond in strict JSON with the following keys:
{
  "title": "Short descriptive title",
  "summary": "Plain-language 2-3 sentence overview",
  "whatDataShows": ["Bullet 1 with exact facts", "Bullet 2 with exact facts"],
  "whatDataDoesNotTellUs": ["Limitation 1", "Limitation 2"],
  "communityContextProvided": "Echo only user-provided context if provided, else null",
  "disclaimer": "This is an educational explanation of the provided dataset, not an official forecast, warning, or emergency alert."
}
`;

    const prompt = `
FACT PACK:
${JSON.stringify(factPack, null, 2)}

USER-PROVIDED UNVERIFIED CONTEXT:
${userContext ? userContext : 'None provided'}

LANGUAGE: ${language}
TARGET AUDIENCE: ${audience}
LENGTH PREFERENCE: ${length}
`;

    const response = await this.ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        systemInstruction,
        responseMimeType: 'application/json',
        temperature: 0.1 // Low temperature to maximize adherence to facts
      }
    });

    const parsed = JSON.parse(response.text);
    return parsed;
  }
}

/**
 * 2. Deterministic Fallback Provider (Offline, Rule-Based)
 * Ensures TideTales operates without third-party API keys or internet access.
 */
export class DeterministicFallbackProvider extends BaseModelProvider {
  async getStatus() {
    return { available: true, name: 'Deterministic Rule Engine', mode: 'offline' };
  }

  async generateNarrative({ factPack, language = 'English', audience = 'community', userContext = '' }) {
    const isOdia = language.toLowerCase().includes('odia');
    const {
      datasetName,
      location,
      measurementName,
      units,
      observationCount,
      startDate,
      endDate,
      firstValue,
      lastValue,
      netChange,
      percentChange,
      trendDirection,
      peak,
      trough,
      volatility
    } = factPack;

    const unitStr = units ? ` ${units}` : '';

    if (isOdia) {
      return {
        title: `${datasetName || 'ଉପକୂଳ ତଥ୍ୟ'}: ବିଶ୍ଳେଷଣ ରିପୋର୍ଟ (${startDate} ରୁ ${endDate})`,
        summary: `ଏହି ତଥ୍ୟାବଳୀ ${startDate} ରୁ ${endDate} ମଧ୍ୟରେ ${observationCount} ଟି ନିରୀକ୍ଷଣ ଉପରେ ଆଧାରିତ। ଏହି ସମୟ ମଧ୍ୟରେ ମାପିତ ମୂଲ୍ୟ ${firstValue}${unitStr} ରୁ ଆରମ୍ଭ ହୋଇ ${lastValue}${unitStr} ରେ ଶେଷ ହୋଇଛି, ଯାହା ସମୁଦାୟ ${netChange}${unitStr} ର ପରିବର୍ତ୍ତନ ଦର୍ଶାଉଛି।`,
        whatDataShows: [
          `ସର୍ବମୋଟ ବୈଧ ନିରୀକ୍ଷଣ ସଂଖ୍ୟା: ${observationCount}।`,
          `ପ୍ରାରମ୍ଭିକ ମୂଲ୍ୟ (${startDate}): ${firstValue}${unitStr} ଏବଂ ଅନ୍ତିମ ମୂଲ୍ୟ (${endDate}): ${lastValue}${unitStr}।`,
          `ମୋଟ ପରିବର୍ତ୍ତନ (Net Change): ${netChange}${unitStr}${percentChange !== 'Not available' ? ` (${percentChange}%)` : ''}।`,
          `ସାମଗ୍ରିକ ପ୍ରବୃତ୍ତି (Trend): ${trendDirection} (ସାଧାରଣ ଲିନିୟର ଫିଟ୍ ଦ୍ୱାରା ନିର୍ଦ୍ଧାରିତ) |`,
          `ସର୍ବୋଚ୍ଚ ମୂଲ୍ୟ (Peak): ${peak.value}${unitStr} (${peak.date} ତାରିଖରେ)।`,
          `ସର୍ବନିମ୍ନ ମୂଲ୍ୟ (Trough): ${trough.value}${unitStr} (${trough.date} ତାରିଖରେ)।`,
          `ପରିବର୍ତ୍ତନଶୀଳତା (Standard Deviation): ${volatility}${unitStr}।`
        ],
        whatDataDoesNotTellUs: [
          'ଏହି ତଥ୍ୟ କୌଣସି ପରିବର୍ତ୍ତନର ପ୍ରତ୍ୟକ୍ଷ କାରଣ (ଉଦାହରଣ ସ୍ୱରୂପ: ବାତ୍ୟା, ମାନବୀୟ କାର୍ଯ୍ୟକଳାପ ବା ଜଳବାୟୁ ପରିବର୍ତ୍ତନ) ପ୍ରମାଣ କରେ ନାହିଁ।',
          'ଏହା କୌଣସି ଭବିଷ୍ୟତର ପୂର୍ବାନୁମାନ ବା ସରକାରୀ ସତର୍କ ସୂଚନା ନୁହେଁ।',
          'ଆଞ୍ଚଳିକ ବଣ୍ଟନ ପୂର୍ବରୁ ଏହି ଓଡ଼ିଆ ଅନୁବାଦକୁ ଜଣେ ଭାଷାଜ୍ଞଙ୍କ ଦ୍ୱାରା ଯାଞ୍ଚ କରିବା ଅନୁରୋଧ।'
        ],
        communityContextProvided: userContext ? `ଉପଭୋକ୍ତାଙ୍କ ଦ୍ୱାରା ଦିଆଯାଇଥିବା ଅସତ୍ୟାପିତ ଟିପ୍ପଣୀ: ${userContext}` : null,
        disclaimer: 'ଏହା ଏକ ତଥ୍ୟ-ଆଧାରିତ ବ୍ୟାଖ୍ୟା ଅଟେ; ଏହା କୌଣସି ସରକାରୀ ସତର୍କତା ବା ଜରୁରୀକାଳୀନ ପୂର୍ବାନୁମାନ ନୁହେଁ।'
      };
    }

    // Default English Narrative
    return {
      title: `${datasetName || 'Coastal Time-Series'}: Grounded Summary (${startDate} to ${endDate})`,
      summary: `Between ${startDate} and ${endDate}, ${observationCount} valid observations were recorded for ${measurementName || 'measurement'}${location ? ` at ${location}` : ''}. Over this period, the measured values changed from ${firstValue}${unitStr} to ${lastValue}${unitStr}, representing a net change of ${netChange}${unitStr}.`,
      whatDataShows: [
        `Observations: ${observationCount} recorded data points from ${startDate} to ${endDate}.`,
        `Net Change: Values shifted by ${netChange}${unitStr}${percentChange !== 'Not available' ? ` (${percentChange}%)` : ''}.`,
        `Observed Trend: ${trendDirection} based on least-squares linear calculation.`,
        `Peak Reading: ${peak.value}${unitStr} on ${peak.date}.`,
        `Trough Reading: ${trough.value}${unitStr} on ${trough.date}.`,
        `Variability: Standard deviation of ${volatility}${unitStr} across recorded intervals.`
      ],
      whatDataDoesNotTellUs: [
        'The underlying causes behind these fluctuations (e.g. atmospheric circulation, anthropogenic influence, or dredging).',
        'Future projections, flood risks, or official warnings.',
        'Unrecorded intervals or localized variations outside of the recorded sampling points.'
      ],
      communityContextProvided: userContext ? `User-submitted unverified context: "${userContext}"` : null,
      disclaimer: 'This story is strictly derived from the provided dataset and does not constitute an official warning or predictive forecast.'
    };
  }
}

/**
 * 3. Future Custom Model Provider Placeholder
 * Reserved for Punyajit Mishra's future fine-tuned or domain-adapted climate models.
 */
export class FutureCustomModelProvider extends BaseModelProvider {
  async getStatus() {
    return {
      available: false,
      name: 'TideTales Custom Coastal LM (Future)',
      mode: 'disabled_no_training_corpus'
    };
  }

  async generateNarrative() {
    throw new Error('Future Custom Model is currently unconfigured. No model weights or curated training corpus have been loaded.');
  }
}
