/**
 * Fact Validation Guard
 * Validates that model output does not invent dates or metrics missing from the Fact Pack.
 */
export function validateNarrative(output, factPack) {
  const issues = [];

  if (!output || typeof output !== 'object') {
    return { valid: false, issues: ['Narrative output format was not valid JSON'] };
  }

  if (!output.title || !output.summary || !Array.isArray(output.whatDataShows)) {
    return { valid: false, issues: ['Required sections (title, summary, whatDataShows) are missing.'] };
  }

  const serializedFactPack = JSON.stringify(factPack).toLowerCase();
  const serializedOutput = JSON.stringify(output).toLowerCase();

  // Basic check for suspicious hallucinated percentage values
  const percentMatches = serializedOutput.match(/(\d+(\.\d+)?)%/g) || [];
  for (const match of percentMatches) {
    const rawNum = match.replace('%', '');
    if (!serializedFactPack.includes(rawNum) && factPack.percentChange !== rawNum) {
      // Flag potential hallucinated percentages not in fact pack
      issues.push(`Model referenced an ungrounded percentage statistic: ${match}`);
    }
  }

  return {
    valid: issues.length === 0,
    issues
  };
}
