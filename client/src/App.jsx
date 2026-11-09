import React, { useState, useEffect, useMemo } from 'react';
import Papa from 'papaparse';
import { computeFactPack } from './utils/factPackEngine';
import { DEMO_CSV, DEMO_METADATA } from './demoData';

export default function App() {
  // Navigation
  const [activeTab, setActiveTab] = useState('home'); // 'home' | 'workspace' | 'assistant' | 'history'

  // Dataset & Configuration
  const [csvRaw, setCsvRaw] = useState('');
  const [columns, setColumns] = useState([]);
  const [selectedDateCol, setSelectedDateCol] = useState('');
  const [selectedValueCol, setSelectedValueCol] = useState('');
  const [parsedRows, setParsedRows] = useState([]);
  const [isDemoActive, setIsDemoActive] = useState(false);

  // Metadata
  const [metadata, setMetadata] = useState({
    datasetName: '',
    location: '',
    measurementName: '',
    units: '',
    source: '',
    userContext: ''
  });

  // Story Generation Options
  const [language, setLanguage] = useState('English'); // 'English' | 'Odia'
  const [audience, setAudience] = useState('Community members');
  const [storyLength, setStoryLength] = useState('short');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationResult, setGenerationResult] = useState(null);
  const [generationError, setGenerationError] = useState(null);

  // Assistant State
  const [assistantQuery, setAssistantQuery] = useState('');
  const [assistantLoading, setAssistantLoading] = useState(false);
  const [assistantResponse, setAssistantResponse] = useState(null);

  // History State
  const [historyRuns, setHistoryRuns] = useState([]);

  // Compute Fact Pack reactively
  const factPack = useMemo(() => {
    if (!parsedRows.length || !selectedDateCol || !selectedValueCol) return null;
    return computeFactPack(parsedRows, selectedDateCol, selectedValueCol, metadata);
  }, [parsedRows, selectedDateCol, selectedValueCol, metadata]);

  // Load demo dataset automatically on workspace entry if empty
  const handleLoadDemo = () => {
    setIsDemoActive(true);
    setMetadata(DEMO_METADATA);
    processCsvText(DEMO_CSV);
    setActiveTab('workspace');
  };

  // CSV Processing
  const processCsvText = (text) => {
    Papa.parse(text, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (results.data && results.data.length > 0) {
          const cols = Object.keys(results.data[0]);
          setColumns(cols);
          setParsedRows(results.data);

          // Auto-detect Date and Measurement columns
          const dateMatch = cols.find(c => /date|year|month|time|day/i.test(c)) || cols[0];
          const valMatch = cols.find(c => /val|anomaly|height|temp|level|salinity|tide/i.test(c) && c !== dateMatch) || cols[1] || cols[0];

          setSelectedDateCol(dateMatch);
          setSelectedValueCol(valMatch);
          setGenerationResult(null);
          setGenerationError(null);
        }
      }
    });
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setIsDemoActive(false);
      const reader = new FileReader();
      reader.onload = (evt) => {
        setCsvRaw(evt.target.result);
        processCsvText(evt.target.result);
      };
      reader.readAsText(file);
    }
  };

  // Trigger Story Generation
  const handleGenerateStory = async () => {
    if (!factPack || !factPack.isValid) return;

    setIsGenerating(true);
    setGenerationError(null);

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          factPack,
          language,
          audience,
          length: storyLength,
          userContext: metadata.userContext
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Failed to generate narrative');
      }

      setGenerationResult(data);
      fetchHistory();
    } catch (err) {
      setGenerationError(err.message);
    } finally {
      setIsGenerating(false);
    }
  };

  // Fetch History
  const fetchHistory = async () => {
    try {
      const res = await fetch('/api/history');
      const data = await res.json();
      setHistoryRuns(data);
    } catch (err) {
      console.warn('Could not load history');
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  // Assistant Query
  const handleAskAssistant = async (e) => {
    e.preventDefault();
    if (!assistantQuery.trim()) return;

    setAssistantLoading(true);
    try {
      const res = await fetch('/api/assistant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: assistantQuery, language })
      });
      const data = await res.json();
      setAssistantResponse(data);
    } catch (err) {
      setAssistantResponse({ answer: 'Failed to contact assistant.', disclaimer: '' });
    } finally {
      setAssistantLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#F5F0E8] text-[#0F283C] font-sans antialiased flex flex-col">
      {/* Top Header */}
      <header className="bg-[#0F283C] text-[#FBFDFD] border-b border-[#0D9488]/40 shadow-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('home')}>
            <span className="text-2xl">🌊</span>
            <div>
              <span className="text-xl font-bold tracking-tight text-[#FBFDFD]">TideTales</span>
              <span className="hidden sm:inline-block ml-2 text-xs bg-[#0D9488] text-white px-2 py-0.5 rounded-full font-medium">
                Grounded Coastal Stories
              </span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="flex items-center space-x-1 sm:space-x-3 text-sm font-medium">
            <button
              onClick={() => setActiveTab('home')}
              className={`px-3 py-1.5 rounded-md transition ${activeTab === 'home' ? 'bg-[#0D9488] text-white' : 'text-slate-300 hover:text-white'}`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab('workspace')}
              className={`px-3 py-1.5 rounded-md transition ${activeTab === 'workspace' ? 'bg-[#0D9488] text-white' : 'text-slate-300 hover:text-white'}`}
            >
              Story Workspace
            </button>
            <button
              onClick={() => setActiveTab('assistant')}
              className={`px-3 py-1.5 rounded-md transition ${activeTab === 'assistant' ? 'bg-[#0D9488] text-white' : 'text-slate-300 hover:text-white'}`}
            >
              Climate Assistant
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`px-3 py-1.5 rounded-md transition ${activeTab === 'history' ? 'bg-[#0D9488] text-white' : 'text-slate-300 hover:text-white'}`}
            >
              History ({historyRuns.length})
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-6xl w-full mx-auto p-4 sm:p-6">
        {/* ========================================================== */}
        {/* TAB 1: OVERVIEW & HOME */}
        {/* ========================================================== */}
        {activeTab === 'home' && (
          <div className="space-y-8 py-4">
            <div className="bg-white rounded-2xl p-6 sm:p-10 shadow-sm border border-[#E6DCCE]">
              <div className="max-w-3xl">
                <span className="text-xs font-bold text-[#0D9488] uppercase tracking-wider">Independent Coastal Initiative</span>
                <h1 className="text-3xl sm:text-4xl font-extrabold text-[#0F283C] mt-2 mb-4">
                  Turning coastal data into clear, verified bilingual stories.
                </h1>
                <p className="text-base sm:text-lg text-slate-700 leading-relaxed mb-6">
                  TideTales is founded and independently developed by <strong>Punyajit Mishra</strong>. It bridges the gap between environmental monitoring instruments and coastal community members, climate champions, and local decision-makers in <strong>English</strong> and <strong>Odia (ଓଡ଼ିଆ)</strong>.
                </p>

                <div className="flex flex-wrap gap-4">
                  <button
                    onClick={() => {
                      if (!parsedRows.length) handleLoadDemo();
                      setActiveTab('workspace');
                    }}
                    className="bg-[#0D9488] hover:bg-[#0f766e] text-white px-6 py-3 rounded-lg font-semibold shadow-md transition flex items-center space-x-2"
                  >
                    <span>Create a Story</span>
                    <span>→</span>
                  </button>
                  <button
                    onClick={handleLoadDemo}
                    className="bg-[#E6DCCE] hover:bg-[#d8ccbc] text-[#0F283C] px-5 py-3 rounded-lg font-medium transition"
                  >
                    Explore Illustrative Demo Dataset
                  </button>
                </div>
              </div>
            </div>

            {/* Workflow Pillars */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white p-6 rounded-xl border border-[#E6DCCE] shadow-sm">
                <div className="w-10 h-10 rounded-lg bg-[#0F283C] text-white flex items-center justify-center font-bold mb-4">
                  1
                </div>
                <h3 className="font-bold text-lg text-[#0F283C] mb-2">Time-Series Ingestion</h3>
                <p className="text-sm text-slate-600 leading-relaxed">
                  Upload local CSV measurements (water levels, salinity, temperatures). No raw data is ever sent to external training systems.
                </p>
              </div>

              <div className="bg-white p-6 rounded-xl border border-[#E6DCCE] shadow-sm">
                <div className="w-10 h-10 rounded-lg bg-[#0D9488] text-white flex items-center justify-center font-bold mb-4">
                  2
                </div>
                <h3 className="font-bold text-lg text-[#0F283C] mb-2">Deterministic Fact Pack</h3>
                <p className="text-sm text-slate-600 leading-relaxed">
                  All metrics, trends, peaks, troughs, and volatility are calculated strictly mathematically before narrative generation. AI is never asked to calculate facts.
                </p>
              </div>

              <div className="bg-white p-6 rounded-xl border border-[#E6DCCE] shadow-sm">
                <div className="w-10 h-10 rounded-lg bg-[#F5F0E8] border border-[#0D9488]/50 text-[#0F283C] flex items-center justify-center font-bold mb-4">
                  3
                </div>
                <h3 className="font-bold text-lg text-[#0F283C] mb-2">Grounded Bilingual Stories</h3>
                <p className="text-sm text-slate-600 leading-relaxed">
                  Generates verified narratives in English or Odia. Strict boundaries prevent fabricated statistics, folklore, or unwarranted causal claims.
                </p>
              </div>
            </div>

            {/* Scientific & Safety Boundaries Notice */}
            <div className="bg-[#E6DCCE]/50 border-l-4 border-[#0D9488] p-5 rounded-r-xl text-sm text-slate-700">
              <h4 className="font-bold text-[#0F283C] mb-1">Notice on Purpose and Boundaries</h4>
              <p>
                TideTales is an educational storytelling and analytical aid. It is <strong>not</strong> an active storm warning system, official marine forecast, or substitute for advisories from the India Meteorological Department (IMD) or State Disaster Management Authorities (OSDMA).
              </p>
            </div>
          </div>
        )}

        {/* ========================================================== */}
        {/* TAB 2: STORY WORKSPACE */}
        {/* ========================================================== */}
        {activeTab === 'workspace' && (
          <div className="space-y-6">
            {/* Step 1: Data Ingestion Banner */}
            <div className="bg-white p-5 rounded-xl border border-[#E6DCCE] shadow-sm">
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-4">
                <div>
                  <h2 className="text-lg font-bold text-[#0F283C]">1. Dataset Selection & Ingestion</h2>
                  <p className="text-xs text-slate-500">Provide coastal time-series data or explore the illustrative demo.</p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={handleLoadDemo}
                    className="text-xs font-semibold px-3 py-1.5 bg-[#F5F0E8] text-[#0D9488] border border-[#0D9488]/40 rounded hover:bg-[#0D9488] hover:text-white transition"
                  >
                    Load Paradip Sample
                  </button>
                  {parsedRows.length > 0 && (
                    <button
                      onClick={() => {
                        setParsedRows([]);
                        setColumns([]);
                        setCsvRaw('');
                        setIsDemoActive(false);
                      }}
                      className="text-xs font-medium px-3 py-1.5 text-red-600 hover:bg-red-50 rounded border border-red-200 transition"
                    >
                      Clear Data
                    </button>
                  )}
                </div>
              </div>

              {isDemoActive && (
                <div className="mb-4 bg-amber-50 border border-amber-200 text-amber-800 text-xs px-3 py-2 rounded-md">
                  ⚠️ <strong>Illustrative Sample Loaded:</strong> This data represents an educational demonstration of Paradip coastal water level anomalies. It does not reflect official live observations.
                </div>
              )}

              {/* Upload Dropzone */}
              <div className="border-2 border-dashed border-[#E6DCCE] hover:border-[#0D9488] transition rounded-lg p-6 text-center bg-[#FBFDFD]">
                <input
                  type="file"
                  id="csvFile"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <label htmlFor="csvFile" className="cursor-pointer flex flex-col items-center">
                  <span className="text-3xl mb-1">📄</span>
                  <span className="text-sm font-semibold text-[#0D9488]">Click to choose a CSV file or drag & drop</span>
                  <span className="text-xs text-slate-500 mt-1">Requires at least one date column and one numeric measurement column</span>
                </label>
              </div>

              {/* Column Selection & Metadata */}
              {columns.length > 0 && (
                <div className="mt-6 pt-6 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-bold uppercase text-slate-600 mb-1">Detected Date Column</label>
                    <select
                      value={selectedDateCol}
                      onChange={(e) => setSelectedDateCol(e.target.value)}
                      className="w-full text-sm border border-slate-300 rounded px-3 py-2 bg-white"
                    >
                      {columns.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-bold uppercase text-slate-600 mb-1">Detected Value Column</label>
                    <select
                      value={selectedValueCol}
                      onChange={(e) => setSelectedValueCol(e.target.value)}
                      className="w-full text-sm border border-slate-300 rounded px-3 py-2 bg-white"
                    >
                      {columns.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>

                  {/* Metadata Fields */}
                  <div>
                    <label className="block text-xs font-semibold text-slate-600 mb-1">Dataset / Station Name</label>
                    <input
                      type="text"
                      value={metadata.datasetName}
                      placeholder="e.g. Paradip Port Tide Gauge"
                      onChange={(e) => setMetadata({ ...metadata, datasetName: e.target.value })}
                      className="w-full text-sm border border-slate-300 rounded px-3 py-1.5"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-600 mb-1">Location</label>
                    <input
                      type="text"
                      value={metadata.location}
                      placeholder="e.g. Bay of Bengal, Odisha"
                      onChange={(e) => setMetadata({ ...metadata, location: e.target.value })}
                      className="w-full text-sm border border-slate-300 rounded px-3 py-1.5"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-600 mb-1">Measurement Type & Units</label>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        placeholder="Type (e.g. Sea Level)"
                        value={metadata.measurementName}
                        onChange={(e) => setMetadata({ ...metadata, measurementName: e.target.value })}
                        className="w-2/3 text-sm border border-slate-300 rounded px-3 py-1.5"
                      />
                      <input
                        type="text"
                        placeholder="Units (e.g. mm, °C)"
                        value={metadata.units}
                        onChange={(e) => setMetadata({ ...metadata, units: e.target.value })}
                        className="w-1/3 text-sm border border-slate-300 rounded px-3 py-1.5"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-600 mb-1">Data Source / Provenance</label>
                    <input
                      type="text"
                      value={metadata.source}
                      placeholder="e.g. Community sensor, Research buoy"
                      onChange={(e) => setMetadata({ ...metadata, source: e.target.value })}
                      className="w-full text-sm border border-slate-300 rounded px-3 py-1.5"
                    />
                  </div>

                  {/* Unverified Context */}
                  <div className="md:col-span-2">
                    <label className="block text-xs font-semibold text-slate-600 mb-1">
                      Optional Local / Community Context Note <span className="text-amber-700">(Labeled as unverified user context)</span>
                    </label>
                    <input
                      type="text"
                      value={metadata.userContext}
                      placeholder="e.g. Community reported heightened shoreline breach in early autumn"
                      onChange={(e) => setMetadata({ ...metadata, userContext: e.target.value })}
                      className="w-full text-sm border border-slate-300 rounded px-3 py-1.5"
                    />
                    <p className="text-[11px] text-slate-500 mt-1">
                      Context is echoed verbatim and is never treated by the model as verified scientific measurement.
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Step 2: Computed Fact Pack & Chart */}
            {factPack && factPack.isValid && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* SVG Time-Series Chart */}
                <div className="lg:col-span-2 bg-white p-5 rounded-xl border border-[#E6DCCE] shadow-sm flex flex-col justify-between">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <h3 className="font-bold text-sm text-[#0F283C]">
                        Time-Series: {factPack.measurementName} {factPack.units ? `(${factPack.units})` : ''}
                      </h3>
                      <span className="text-xs bg-teal-50 text-[#0D9488] font-medium px-2 py-0.5 rounded">
                        {factPack.observationCount} observations
                      </span>
                    </div>

                    {/* Chart Container */}
                    <div className="w-full h-64 bg-slate-50 rounded-lg p-2 border border-slate-200 relative">
                      <TimeSeriesSvgChart points={factPack.validPoints} units={factPack.units} />
                    </div>
                  </div>

                  <div className="mt-3 flex justify-between text-xs text-slate-500 border-t border-slate-100 pt-2">
                    <span>Start: <strong>{factPack.startDate}</strong></span>
                    <span>Linear Trend: <strong>{factPack.trendDirection}</strong></span>
                    <span>End: <strong>{factPack.endDate}</strong></span>
                  </div>
                </div>

                {/* Fact Pack Details Panel */}
                <div className="bg-[#FBFDFD] p-5 rounded-xl border border-[#E6DCCE] shadow-sm text-sm">
                  <div className="flex justify-between items-center mb-3">
                    <h3 className="font-bold text-xs uppercase tracking-wide text-slate-600">Deterministic Fact Pack</h3>
                    <span className="text-[10px] bg-emerald-100 text-emerald-800 font-semibold px-2 py-0.5 rounded">
                      Verified Arithmetic
                    </span>
                  </div>

                  <div className="space-y-2.5 text-xs">
                    <div className="flex justify-between py-1 border-b border-slate-100">
                      <span className="text-slate-500">Net Absolute Change:</span>
                      <span className="font-bold text-[#0F283C]">{factPack.netChange} {factPack.units}</span>
                    </div>

                    <div className="flex justify-between py-1 border-b border-slate-100">
                      <span className="text-slate-500">Percentage Change:</span>
                      <span className="font-bold text-[#0F283C]">
                        {factPack.percentChange !== 'Not available' ? `${factPack.percentChange}%` : 'N/A (base is zero)'}
                      </span>
                    </div>

                    <div className="flex justify-between py-1 border-b border-slate-100">
                      <span className="text-slate-500">Peak Value:</span>
                      <span className="font-bold text-[#0F283C]">{factPack.peak.value} {factPack.units} ({factPack.peak.date})</span>
                    </div>

                    <div className="flex justify-between py-1 border-b border-slate-100">
                      <span className="text-slate-500">Trough Value:</span>
                      <span className="font-bold text-[#0F283C]">{factPack.trough.value} {factPack.units} ({factPack.trough.date})</span>
                    </div>

                    <div className="flex justify-between py-1 border-b border-slate-100">
                      <span className="text-slate-500">Range:</span>
                      <span className="font-bold text-[#0F283C]">{factPack.range} {factPack.units}</span>
                    </div>

                    <div className="flex justify-between py-1 border-b border-slate-100">
                      <span className="text-slate-500">Volatility (Std Dev):</span>
                      <span className="font-bold text-[#0F283C]">{factPack.volatility} {factPack.units}</span>
                    </div>

                    <div className="pt-2 text-[11px] text-slate-500 italic">
                      Method: {factPack.trendMethod}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Step 3: Grounded Story Generator Controls */}
            {factPack && factPack.isValid && (
              <div className="bg-white p-5 rounded-xl border border-[#E6DCCE] shadow-sm">
                <h3 className="font-bold text-sm text-[#0F283C] mb-3">3. Narrative Synthesis Controls</h3>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-600 mb-1">Target Language</label>
                    <select
                      value={language}
                      onChange={(e) => setLanguage(e.target.value)}
                      className="w-full text-sm border border-slate-300 rounded px-3 py-2 bg-white"
                    >
                      <option value="English">English</option>
                      <option value="Odia">Odia (ଓଡ଼ିଆ)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-600 mb-1">Target Audience</label>
                    <select
                      value={audience}
                      onChange={(e) => setAudience(e.target.value)}
                      className="w-full text-sm border border-slate-300 rounded px-3 py-2 bg-white"
                    >
                      <option value="Community members">Community Members</option>
                      <option value="Climate champions">Climate Champions</option>
                      <option value="Researchers">Researchers</option>
                      <option value="Local decision-makers">Local Decision-Makers</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-600 mb-1">Narrative Length</label>
                    <select
                      value={storyLength}
                      onChange={(e) => setStoryLength(e.target.value)}
                      className="w-full text-sm border border-slate-300 rounded px-3 py-2 bg-white"
                    >
                      <option value="short">Concise Summary</option>
                      <option value="detailed">Detailed Breakdown</option>
                    </select>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-2">
                  <div className="text-xs text-slate-500">
                    Grounded strictly in the Fact Pack. Never invents folklore or predictions.
                  </div>
                  <button
                    onClick={handleGenerateStory}
                    disabled={isGenerating}
                    className="bg-[#0F283C] hover:bg-[#164E63] text-white px-6 py-2.5 rounded-lg text-sm font-semibold shadow transition disabled:opacity-50 flex items-center space-x-2"
                  >
                    {isGenerating ? (
                      <>
                        <span className="inline-block animate-spin">⏳</span>
                        <span>Grounding story...</span>
                      </>
                    ) : (
                      <>
                        <span>Generate Bilingual Story</span>
                        <span>✨</span>
                      </>
                    )}
                  </button>
                </div>

                {generationError && (
                  <div className="mt-4 p-3 bg-red-50 border border-red-200 text-red-700 text-xs rounded">
                    Error: {generationError}
                  </div>
                )}
              </div>
            )}

            {/* Generated Narrative Display */}
            {generationResult && (
              <div className="bg-white p-6 sm:p-8 rounded-2xl border border-[#0D9488]/40 shadow-sm space-y-6">
                <div className="flex flex-wrap items-center justify-between border-b border-slate-100 pb-4 gap-2">
                  <div>
                    <span className="text-xs font-bold text-[#0D9488] uppercase tracking-wide">
                      {generationResult.mode.includes('fallback') ? 'Deterministic Grounded Narrative' : 'AI-Assisted Story (Gemini 2.5 Flash)'}
                    </span>
                    <h2 className="text-2xl font-bold text-[#0F283C] mt-1">{generationResult.narrative.title}</h2>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => {
                        const jsonStr = JSON.stringify(factPack, null, 2);
                        navigator.clipboard.writeText(jsonStr);
                        alert('Fact Pack JSON copied to clipboard!');
                      }}
                      className="text-xs font-medium px-3 py-1.5 border border-slate-300 rounded hover:bg-slate-50 transition"
                    >
                      Export Fact Pack (JSON)
                    </button>
                    <button
                      onClick={() => {
                        const md = `# ${generationResult.narrative.title}\n\n${generationResult.narrative.summary}\n\n## What the data shows\n${generationResult.narrative.whatDataShows.map(b => `- ${b}`).join('\n')}\n\n## What this does not tell us\n${generationResult.narrative.whatDataDoesNotTellUs.map(b => `- ${b}`).join('\n')}\n\n_${generationResult.narrative.disclaimer}_`;
                        navigator.clipboard.writeText(md);
                        alert('Narrative Markdown copied to clipboard!');
                      }}
                      className="text-xs font-semibold px-3 py-1.5 bg-[#0D9488] text-white rounded hover:bg-[#0f766e] transition"
                    >
                      Copy Markdown
                    </button>
                  </div>
                </div>

                {/* Odia Fluent Speaker Review Notice */}
                {language === 'Odia' && (
                  <div className="bg-amber-50 border-l-4 border-amber-500 p-3 rounded-r text-xs text-amber-900">
                    <strong>ଓଡ଼ିଆ ଅନୁବାଦ ସତର୍କତା:</strong> ଏହା ଏକ କମ୍ପ୍ୟୁଟର-ସୃଷ୍ଟ ଅନୁବାଦ। ସମୁଦାୟରେ ବିତରଣ କରିବା ପୂର୍ବରୁ ଜଣେ ସ୍ଥାନୀୟ ଓଡ଼ିଆ ଭାଷାଜ୍ଞଙ୍କ ଦ୍ୱାରା ଏହାକୁ ଯାଞ୍ଚ କରିବା ପାଇଁ ପରାମର୍ଶ ଦିଆଯାଉଛି।
                  </div>
                )}

                {/* Plain-Language Summary */}
                <div>
                  <h4 className="text-xs font-bold uppercase text-slate-500 mb-1">Summary</h4>
                  <p className="text-base text-slate-800 leading-relaxed bg-[#FBFDFD] p-4 rounded-lg border border-slate-100">
                    {generationResult.narrative.summary}
                  </p>
                </div>

                {/* What the data shows */}
                <div>
                  <h4 className="text-xs font-bold uppercase text-emerald-700 mb-2">What The Data Shows</h4>
                  <ul className="space-y-1.5">
                    {generationResult.narrative.whatDataShows.map((item, idx) => (
                      <li key={idx} className="text-sm text-slate-700 flex items-start space-x-2">
                        <span className="text-emerald-500 font-bold">✓</span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* What this does not tell us */}
                <div>
                  <h4 className="text-xs font-bold uppercase text-slate-500 mb-2">What This Does Not Tell Us (Boundaries & Limits)</h4>
                  <ul className="space-y-1.5">
                    {generationResult.narrative.whatDataDoesNotTellUs.map((item, idx) => (
                      <li key={idx} className="text-sm text-slate-600 flex items-start space-x-2">
                        <span className="text-slate-400 font-bold">•</span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Community context note */}
                {generationResult.narrative.communityContextProvided && (
                  <div className="p-3 bg-slate-50 rounded border border-slate-200 text-xs text-slate-600">
                    <strong>Unverified Community Context:</strong> {generationResult.narrative.communityContextProvided}
                  </div>
                )}

                {/* Non-Forecast Disclaimer */}
                <div className="pt-4 border-t border-slate-100 text-xs text-slate-400 italic">
                  {generationResult.narrative.disclaimer}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ========================================================== */}
        {/* TAB 3: CLIMATE SCIENCE ASSISTANT */}
        {/* ========================================================== */}
        {activeTab === 'assistant' && (
          <div className="max-w-3xl mx-auto space-y-6">
            <div className="bg-white p-6 rounded-2xl border border-[#E6DCCE] shadow-sm">
              <h2 className="text-2xl font-bold text-[#0F283C] mb-2">Climate Science Assistant</h2>
              <p className="text-sm text-slate-600 mb-4">
                Inquire about Earth systems, sea level trends, weather versus climate, and coastal hazards in the Bay of Bengal and Odisha.
              </p>

              <form onSubmit={handleAskAssistant} className="space-y-3">
                <textarea
                  rows={3}
                  value={assistantQuery}
                  onChange={(e) => setAssistantQuery(e.target.value)}
                  placeholder="e.g. How do sea surface temperatures influence tropical cyclones in the Bay of Bengal?"
                  className="w-full text-sm border border-slate-300 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-[#0D9488]"
                />

                <div className="flex justify-between items-center">
                  <div className="text-xs text-slate-400">
                    Sources grounded in IPCC, IMD, INCOIS consensus.
                  </div>
                  <button
                    type="submit"
                    disabled={assistantLoading || !assistantQuery.trim()}
                    className="bg-[#0D9488] hover:bg-[#0f766e] text-white px-5 py-2 rounded-lg text-sm font-semibold transition disabled:opacity-50"
                  >
                    {assistantLoading ? 'Analyzing...' : 'Ask Assistant'}
                  </button>
                </div>
              </form>
            </div>

            {assistantResponse && (
              <div className="bg-white p-6 rounded-2xl border border-[#0D9488]/30 shadow-sm space-y-4">
                <div className="flex items-center justify-between text-xs text-slate-500 border-b border-slate-100 pb-2">
                  <span className="font-semibold text-[#0D9488]">Classification: {assistantResponse.type}</span>
                  <span>Grounding: {assistantResponse.sourceGrounding?.join(', ') || 'Scientific Consensus'}</span>
                </div>

                <div className="text-sm text-slate-800 leading-relaxed whitespace-pre-line">
                  {assistantResponse.answer}
                </div>

                {assistantResponse.disclaimer && (
                  <div className="p-3 bg-amber-50 border border-amber-200 text-amber-900 rounded text-xs">
                    ⚠️ {assistantResponse.disclaimer}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ========================================================== */}
        {/* TAB 4: RUN HISTORY & LOCAL DATA */}
        {/* ========================================================== */}
        {activeTab === 'history' && (
          <div className="max-w-4xl mx-auto space-y-6">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-2xl font-bold text-[#0F283C]">Story History</h2>
                <p className="text-xs text-slate-500">Locally saved runs and exported Fact Packs.</p>
              </div>

              {historyRuns.length > 0 && (
                <button
                  onClick={async () => {
                    if (window.confirm('Clear all stored runs?')) {
                      await fetch('/api/history', { method: 'DELETE' });
                      fetchHistory();
                    }
                  }}
                  className="text-xs font-semibold px-3 py-1.5 text-red-600 border border-red-200 hover:bg-red-50 rounded transition"
                >
                  Delete All Data
                </button>
              )}
            </div>

            {historyRuns.length === 0 ? (
              <div className="bg-white p-8 rounded-xl border border-[#E6DCCE] text-center text-slate-500 text-sm">
                No past runs recorded. Generate a story in the workspace to see it listed here.
              </div>
            ) : (
              <div className="space-y-3">
                {historyRuns.map(run => (
                  <div key={run.id} className="bg-white p-4 rounded-xl border border-[#E6DCCE] shadow-sm flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
                    <div>
                      <h4 className="font-bold text-sm text-[#0F283C]">{run.datasetName}</h4>
                      <p className="text-xs text-slate-500">
                        {run.location} • {run.language} • {new Date(run.createdAt).toLocaleString()} • Mode: <span className="font-mono">{run.mode}</span>
                      </p>
                    </div>

                    <div className="flex gap-2">
                      <button
                        onClick={() => {
                          setGenerationResult({ narrative: run.narrative, mode: run.mode });
                          setActiveTab('workspace');
                        }}
                        className="text-xs px-3 py-1 bg-teal-50 text-[#0D9488] font-semibold rounded hover:bg-[#0D9488] hover:text-white transition"
                      >
                        Reopen Story
                      </button>
                      <button
                        onClick={async () => {
                          await fetch(`/api/history/${run.id}`, { method: 'DELETE' });
                          fetchHistory();
                        }}
                        className="text-xs px-2 py-1 text-slate-400 hover:text-red-600 transition"
                      >
                        ✕
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-[#0F283C] text-[#FBFDFD] text-xs py-6 mt-12 border-t border-[#0D9488]/30">
        <div className="max-w-6xl mx-auto px-4 flex flex-col sm:flex-row justify-between items-center gap-4">
          <div>
            <span className="font-bold">TideTales</span> — Founded and independently developed by Punyajit Mishra.
          </div>
          <div className="text-slate-400 text-center sm:text-right">
            Grounding environmental data for coastal champions. Not an active warning or forecast system.
          </div>
        </div>
      </footer>
    </div>
  );
}

/**
 * Accessible SVG Time-Series Chart Component
 */
function TimeSeriesSvgChart({ points, units }) {
  if (!points || points.length === 0) return null;

  const width = 600;
  const height = 240;
  const padding = 36;

  const values = points.map(p => p.value);
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const valRange = maxVal - minVal || 1;

  const getX = (idx) => padding + (idx / (points.length - 1)) * (width - 2 * padding);
  const getY = (val) => height - padding - ((val - minVal) / valRange) * (height - 2 * padding);

  const pathData = points.reduce((acc, p, idx) => {
    const x = getX(idx);
    const y = getY(p.value);
    return idx === 0 ? `M ${x} ${y}` : `${acc} L ${x} ${y}`;
  }, '');

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full overflow-visible" role="img" aria-label="Time series chart">
      {/* Background Grid Lines */}
      <line x1={padding} y1={getY(minVal)} x2={width - padding} y2={getY(minVal)} stroke="#e2e8f0" strokeDasharray="3 3" />
      <line x1={padding} y1={getY((minVal + maxVal) / 2)} x2={width - padding} y2={getY((minVal + maxVal) / 2)} stroke="#e2e8f0" strokeDasharray="3 3" />
      <line x1={padding} y1={getY(maxVal)} x2={width - padding} y2={getY(maxVal)} stroke="#e2e8f0" strokeDasharray="3 3" />

      {/* Axis Labels */}
      <text x={padding - 6} y={getY(maxVal) + 4} textAnchor="end" className="text-[10px] fill-slate-400 font-mono">
        {maxVal.toFixed(1)}
      </text>
      <text x={padding - 6} y={getY(minVal) + 4} textAnchor="end" className="text-[10px] fill-slate-400 font-mono">
        {minVal.toFixed(1)}
      </text>

      {/* Chart Line */}
      <path d={pathData} fill="none" stroke="#0D9488" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />

      {/* Data Points */}
      {points.map((p, idx) => (
        <circle
          key={idx}
          cx={getX(idx)}
          cy={getY(p.value)}
          r="3.5"
          className="fill-white stroke-[#0F283C] stroke-2 hover:scale-150 transition-all cursor-pointer"
        >
          <title>{`${p.date}: ${p.value} ${units || ''}`}</title>
        </circle>
      ))}
    </svg>
  );
}
