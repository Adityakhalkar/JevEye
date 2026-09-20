/**
 * Server-side Jev access. The API key lives here and nowhere else — in
 * particular it never reaches the browser, which is why a server exists at all.
 */
import { TypeSafeClient } from "@typesafe-ai/sdk";

/** Jev 1.13 input-token price, used to report what a question cost. */
export const USD_PER_MILLION_INPUT_TOKENS = 0.042;

let client: TypeSafeClient | null = null;

export function jev(): TypeSafeClient {
  client ??= new TypeSafeClient();
  return client;
}

export function usdCost(inputTokens: number): number {
  return (inputTokens / 1_000_000) * USD_PER_MILLION_INPUT_TOKENS;
}

/** Shapes an SDK or configuration failure into something the UI can show. */
export function jevError(error: unknown): { message: string; status: number } {
  const message = error instanceof Error ? error.message : String(error);
  if (/TYPESAFE_API_KEY|api key/i.test(message)) {
    return { message: "TYPESAFE_API_KEY is not set on the server.", status: 500 };
  }
  if (/rate limit/i.test(message)) {
    return { message: "Jev is rate limiting; try again in a moment.", status: 429 };
  }
  return { message, status: 502 };
}
