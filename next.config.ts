import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Without this Next walks up to the home directory looking for a lockfile.
  outputFileTracingRoot: import.meta.dirname,
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          // transformers.js needs SharedArrayBuffer for multi-threaded WASM, which
          // needs cross-origin isolation. `credentialless` rather than `require-corp`
          // so the model files still load from the Hugging Face CDN.
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "credentialless" },
        ],
      },
    ];
  },
};

export default nextConfig;
