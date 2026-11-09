/**
 * Utility to format Arabic numerals into Odia script numerals
 * 0 -> ୦, 1 -> ୧, 2 -> ୨, etc.
 */
export function toOdiaNumerals(input) {
  if (input === null || input === undefined) return '';
  const odiaDigits = ['୦', '୧', '୨', '୩', '୪', '୫', '୬', '୭', '୮', '୯'];
  return String(input).replace(/[0-9]/g, (digit) => odiaDigits[parseInt(digit, 10)]);
}
