/**
 * In-Memory Run History Store (Easily swappable with SQLite/Postgres)
 */
class HistoryStore {
  constructor() {
    this.runs = [];
  }

  getAll() {
    return this.runs;
  }

  saveRun(run) {
    const record = {
      id: run.id || `run_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      createdAt: new Date().toISOString(),
      datasetName: run.datasetName || 'Untitled Dataset',
      location: run.location || 'Unknown',
      language: run.language || 'English',
      status: run.status || 'Completed',
      factPack: run.factPack,
      narrative: run.narrative,
      mode: run.mode || 'deterministic'
    };
    this.runs.unshift(record);
    if (this.runs.length > 50) this.runs.pop(); // keep last 50
    return record;
  }

  deleteRun(id) {
    this.runs = this.runs.filter(r => r.id !== id);
    return true;
  }

  clearAll() {
    this.runs = [];
    return true;
  }
}

export const historyStore = new HistoryStore();
