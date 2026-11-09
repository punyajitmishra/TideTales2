/**
 * Climate Science Assistant (First Version)
 * Grounded educational responses about climate science, ocean hazards, and coastal India/Odisha.
 */
import { GoogleGenAI } from '@google/genai';

export async function askClimateAssistant({ question, language = 'English', apiKey }) {
  if (!apiKey) {
    return getEducationalFallbackAnswer(question, language);
  }

  const ai = new GoogleGenAI({ apiKey });
  const systemInstruction = `
You are the TideTales Climate Science Assistant.
Explain established climate science, ocean systems, coastal hazards (sea level rise, cyclonic activity, erosion), and coastal Odisha / India contexts clearly.
Rules:
1. Ground answers in established scientific consensus (IPCC, WMO, NASA, NOAA, IMD, INCOIS).
2. Distinguish:
   - General Science Explanation
   - Source-grounded points
   - Uncertainty & limitations
3. Never issue active storm or flood warnings. Direct users to the India Meteorological Department (IMD) or local State Disaster Management Authorities (OSDMA) for emergency advisories.
4. If asked in Odia, provide clear, respectful Odia with a note that scientific translations should be reviewed by local educators.
`;

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: `Question: ${question}\nLanguage: ${language}`,
    config: { systemInstruction, temperature: 0.2 }
  });

  return {
    answer: response.text,
    type: 'general_science_explanation',
    sourceGrounding: ['IPCC Sixth Assessment Report', 'INCOIS Coastal Hazard Bulletins', 'IMD'],
    disclaimer: 'For real-time cyclone, tsunami, or weather warnings, consult official bulletins from IMD (mausam.imd.gov.in) and INCOIS.'
  };
}

function getEducationalFallbackAnswer(question, language) {
  const isOdia = language.toLowerCase().includes('odia');
  if (isOdia) {
    return {
      answer: `ଜଳବାୟୁ ବିଜ୍ଞାନ ଏବଂ ସମୁଦ୍ର ତଟବର୍ତ୍ତୀ ପ୍ରକ୍ରିୟା ସମ୍ବନ୍ଧୀୟ ପ୍ରଶ୍ନ ପାଇଁ ଧନ୍ୟବାଦ। ପାଣିପାଗ (Weather) ହେଉଛି ଦୈନନ୍ଦିନ ବାୟୁମଣ୍ଡଳୀୟ ଅବସ୍ଥା, କିନ୍ତୁ ଜଳବାୟୁ (Climate) ହେଉଛି ଅତି କମରେ ୩୦ ବର୍ଷର ହାରାହାରି ପରିବର୍ତ୍ତନ। ଭାରତ ତଥା ଓଡ଼ିଶା ଉପକୂଳରେ ବାରମ୍ବାର ଆସୁଥିବା ବାତ୍ୟା ଓ ସମୁଦ୍ର ପତ୍ତନ ବୃଦ୍ଧି ଅଧ୍ୟୟନ ପାଇଁ IMD ଏବଂ INCOIS ର ତଥ୍ୟ ସର୍ବୋତ୍ତମ।`,
      type: 'general_science_explanation (offline)',
      sourceGrounding: ['IMD', 'INCOIS', 'IPCC'],
      disclaimer: 'ସକ୍ରିୟ ପାଣିପାଗ ସତର୍କତା ପାଇଁ ଦୟାକରି ଭାରତୀୟ ପାଣିପାଗ ବିଭାଗ (IMD) ର ସରକାରୀ ୱେବସାଇଟ୍ ଦେଖନ୍ତୁ।'
    };
  }
  return {
    answer: `Weather refers to short-term atmospheric conditions (hours or days), whereas climate refers to long-term statistical patterns (typically 30 years or more). In coastal regions like the Bay of Bengal, sea surface temperatures, monsoonal dynamics, and coastal topography interact closely. Scientific insights must be drawn from long-term observations compiled by institutions like IPCC, IMD, and INCOIS.`,
    type: 'general_science_explanation (offline)',
    sourceGrounding: ['IPCC Working Group I', 'INCOIS Coastal Information Services', 'IMD'],
    disclaimer: 'For real-time operational weather and storm alerts, consult official channels at mausam.imd.gov.in or OSDMA.'
  };
}
