import { releaseURL, validateRelease } from './format.js';

export async function verifiedJSON(response, sha256) {
  if (!sha256) return response.json();
  if (!/^[a-f0-9]{64}$/.test(sha256) || !globalThis.crypto?.subtle) throw new Error('Verified parameter samples require HTTPS or localhost and browser cryptography support.');
  const text = await response.text();
  const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  const actual = [...new Uint8Array(bytes)].map(value => value.toString(16).padStart(2, '0')).join('');
  if (actual !== sha256) throw new Error('Parameter sample checksum mismatch. Retry this graph.');
  return JSON.parse(text);
}

// Shared requests have no consumer's AbortSignal: changing one graph cannot
// cancel another. Each card checks its own request generation before rendering.
export class DataClient {
  constructor(base, { budget = 64 * 1024 * 1024, fetcher = (...args) => fetch(...args) } = {}) {
    this.base = base; this.budget = budget; this.fetcher = fetcher;
    this.cache = new Map(); this.pending = new Map(); this.bytes = 0; this.manifest = null;
  }
  async json(url, { cache = true, sha256 = null } = {}) {
    const key = String(url) + (sha256 ? '#' + sha256 : '');
    if (cache && this.cache.has(key)) {
      const entry = this.cache.get(key); this.cache.delete(key); this.cache.set(key, entry); return entry.value;
    }
    if (cache && this.pending.has(key)) return this.pending.get(key);
    const request = (async () => {
      const response = await this.fetcher(url, { cache: cache ? 'default' : 'no-cache' });
      if (!response.ok) throw new Error(`Archive request failed (${response.status}). Retry this graph.`);
      const value = await verifiedJSON(response, sha256);
      // Conservative decoded payload estimate; bounded separately from card data.
      const size = JSON.stringify(value).length * 2;
      if (cache && size <= this.budget) {
        while (this.bytes + size > this.budget && this.cache.size) {
          const oldest = this.cache.keys().next().value;
          this.bytes -= this.cache.get(oldest).size; this.cache.delete(oldest);
        }
        this.cache.set(key, { value, size }); this.bytes += size;
      }
      return value;
    })();
    if (cache) this.pending.set(key, request);
    try { return await request; } finally { if (cache) this.pending.delete(key); }
  }
  async pin() {
    if (!this.pinning) this.pinning = this.json(new URL('data/manifest.json', this.base), { cache: false }).then(manifest => {
      if (!['1.1.0', '1.2.0', '2.0.0'].includes(manifest.schema_version) || !/^[a-f0-9]{64}$/.test(manifest.release_id)) throw new Error('This archive does not include a supported data explorer.');
      this.manifest = manifest; return manifest;
    }).catch(error => { this.pinning = null; throw error; });
    return this.pinning;
  }
  async release(path) {
    const manifest = await this.pin();
    return validateRelease(await this.json(releaseURL(path, this.base, manifest.release_id)), manifest);
  }
  clear() { this.cache.clear(); this.bytes = 0; }
  async newerRelease() {
    const latest = await this.json(new URL('data/manifest.json', this.base), { cache: false });
    return latest.release_id !== this.manifest?.release_id;
  }
}
