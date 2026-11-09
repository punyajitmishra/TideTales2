/**
 * Deterministic Fact Pack Calculator
 * Computes all statistical facts strictly in code before any model involvement.
 */
export function computeFactPack(rows, dateCol, valCol, metadata = {}) {
  const issues = [];
  const validPoints = [];

  rows.forEach((row, idx) => {
    const rawDate = row[dateCol];
    const rawVal = row[valCol];

    if (!rawDate || rawVal === undefined || rawVal === null || rawVal === '') {
      issues.push(`Row ${idx + 1}: Missing date or value.`);
      return;
    }

    const num = Number(rawVal);
    if (isNaN(num)) {
      issues.push(`Row ${idx + 1}: Value "${rawVal}" is not a valid number.`);
      return;
    }

    validPoints.push({
      date: String(rawDate).trim(),
      value: num
    });
  });

  if (validPoints.length < 2) {
    return {
      isValid: false,
      error: 'At least two valid data points are required to calculate facts.',
      issues
    };
  }

  // Sort by date chronologically if formatted ISO or standard
  validPoints.sort((a, b) => (a.date > b.date ? 1 : -1));

  const observationCount = validPoints.length;
  const firstPoint = validPoints[0];
  const lastPoint = validPoints[validPoints.length - 1];

  const firstValue = firstPoint.value;
  const lastValue = lastPoint.value;

  // Absolute Net Change
  const netChange = Number((lastValue - firstValue).toFixed(3));

  // Percentage Change (Edge-case safe)
  let percentChange = 'Not available';
  if (firstValue !== 0) {
    const pct = ((lastValue - firstValue) / Math.abs(firstValue)) * 100;
    percentChange = Number(pct.toFixed(2));
  }

  // Peak & Trough
  let peak = validPoints[0];
  let trough = validPoints[0];
  let sum = 0;

  validPoints.forEach(p => {
    sum += p.value;
    if (p.value > peak.value) peak = p;
    if (p.value < trough.value) trough = p;
  });

  const mean = sum / observationCount;
  const range = Number((peak.value - trough.value).toFixed(3));

  // Volatility (Sample Standard Deviation)
  const variance = validPoints.reduce((acc, p) => acc + Math.pow(p.value - mean, 2), 0) / (observationCount - 1);
  const volatility = Number(Math.sqrt(variance).toFixed(3));

  // Ordinary Least Squares Linear Trend Fit
  let xSum = 0;
  let ySum = 0;
  let xySum = 0;
  let xxSum = 0;

  for (let i = 0; i < observationCount; i++) {
    xSum += i;
    ySum += validPoints[i].value;
    xySum += i * validPoints[i].value;
    xxSum += i * i;
  }

  const denominator = observationCount * xxSum - xSum * xSum;
  let slope = 0;
  if (denominator !== 0) {
    slope = (observationCount * xySum - xSum * ySum) / denominator;
  }

  let trendDirection = 'Relatively stable / no significant slope';
  if (slope > 0.05) trendDirection = 'Upward trend';
  else if (slope < -0.05) trendDirection = 'Downward trend';

  return {
    isValid: true,
    datasetName: metadata.datasetName || 'Coastal Dataset',
    location: metadata.location || 'Unspecified Coastal Site',
    measurementName: metadata.measurementName || 'Value',
    units: metadata.units || '',
    source: metadata.source || 'User Upload',
    userContext: metadata.userContext || '',
    observationCount,
    startDate: firstPoint.date,
    endDate: lastPoint.date,
    firstValue: Number(firstValue.toFixed(3)),
    lastValue: Number(lastValue.toFixed(3)),
    netChange,
    percentChange,
    trendDirection,
    trendMethod: 'Ordinary least squares linear regression slope across observation intervals',
    peak: { date: peak.date, value: Number(peak.value.toFixed(3)) },
    trough: { date: trough.date, value: Number(trough.value.toFixed(3)) },
    range,
    volatility,
    volatilityMethod: 'Sample standard deviation',
    missingOrExcludedCount: issues.length,
    validPoints,
    issues
  };
}
