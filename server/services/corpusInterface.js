/**
 * Future Corpus & Traditional Ecological Knowledge (TEK) Interface
 * Explicitly disabled until curated, consented resources are integrated.
 * Designed to hold consent, provenance, license, and community review status.
 */
export class CorpusInterface {
  constructor() {
    this.records = [];
    this.isEnabled = false; // Disabled by default
  }

  getStatus() {
    return {
      enabled: this.isEnabled,
      itemCount: this.records.length,
      note: 'No external or traditional ecological knowledge corpus is active. Uploaded data is never stored or used as training data.'
    };
  }

  async queryCorpus(terms, location) {
    if (!this.isEnabled) {
      return [];
    }
    // Future implementation: retrieve approved glossaries and verified Odia terminology
    return [];
  }
}

export const corpusRegistry = new CorpusInterface();
