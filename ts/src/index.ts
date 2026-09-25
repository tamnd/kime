// @kime/sdk: the TypeScript client for kime-serve, Jev or impossibl.
//
// It follows @typesafe-ai/sdk 0.6.0 (MIT, Copyright (c) 2026 TypeSafe, see THIRD_PARTY.md)
// method for method: the client options, systemOne, models.list, the APIPromise with
// .withResponse(), .asResponse() and .map(), the noul, choice and score builders, the answer types
// inferred from the question map, the error classes and their messages, the retry policy and the
// logging. The differences are the ones kime needs:
//
// - The base URL defaults to http://127.0.0.1:8000, where `kime serve` listens.
// - The API key is only required for api.typesafe.ai, since `kime serve` runs without keys.
// - KIME_API_KEY, KIME_BASE_URL, KIME_DEFAULT_MODEL and KIME_LOG_LEVEL come first, and the
//   TYPESAFE_ names are read when they are not set.
// - A request can carry a typed `kime` options field (precision, confidence, extensions,
//   deadline_ms), and the answers it adds are typed too.

/** Parsed data with its HTTP response and request ID. */
export interface WithResponse<T> {
  /** The parsed response body. */
  data: T;
  /** The HTTP response, with its body consumed by parsing. */
  response: Response;
  /** Request ID from `x-typesafe-request-id`, or `undefined` when absent. */
  requestId: string | undefined;
}

const requestIdFrom = (headers: Headers): string | undefined =>
  headers.get("x-typesafe-request-id") ?? undefined;

/**
 * A promise for the parsed result with access to the HTTP response.
 *
 * Non-2xx responses reject with an `APIError`, including through `asResponse()`.
 */
export class APIPromise<T> extends Promise<T> {
  #responsePromise: Promise<Response>;
  #parseResponse: (response: Response) => Promise<T>;
  #parsed: Promise<T> | undefined;

  constructor(responsePromise: Promise<Response>, parseResponse: (response: Response) => Promise<T>) {
    super((resolve) => resolve(undefined as T));
    this.#responsePromise = responsePromise;
    this.#parseResponse = parseResponse;
  }

  // Promise methods such as Promise.all build a new promise from this constructor's species.
  static override get [Symbol.species]() {
    return Promise;
  }

  /**
   * Resolves to the raw `Response` without parsing the body. The body has been buffered under the
   * request timeout, and reading it is up to the caller, who should not also await the result.
   */
  asResponse(): Promise<Response> {
    return this.#responsePromise;
  }

  /** Return the parsed result, HTTP response, and request ID. */
  async withResponse(): Promise<WithResponse<T>> {
    const [data, response] = await Promise.all([this.#parse(), this.#responsePromise]);
    return { data, response, requestId: requestIdFrom(response.headers) };
  }

  /** Transform the parsed result, sharing the HTTP response and a single body parse. */
  map<U>(fn: (data: T) => U): APIPromise<U> {
    return new APIPromise(this.#responsePromise, () => this.#parse().then(fn));
  }

  #parse(): Promise<T> {
    this.#parsed ??= this.#responsePromise.then(this.#parseResponse);
    return this.#parsed;
  }

  override then<TResult1 = T, TResult2 = never>(
    onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | null,
    onrejected?: ((reason: unknown) => TResult2 | PromiseLike<TResult2>) | null,
  ): Promise<TResult1 | TResult2> {
    return this.#parse().then(onfulfilled, onrejected);
  }

  override catch<TResult = never>(
    onrejected?: ((reason: unknown) => TResult | PromiseLike<TResult>) | null,
  ): Promise<T | TResult> {
    return this.#parse().catch(onrejected);
  }

  override finally(onfinally?: (() => void) | null): Promise<T> {
    return this.#parse().finally(onfinally);
  }
}

// Types

/** A JSON-compatible value. */
export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };
/** Text, a JSON object or array, or `null` for state, instructions, and criteria. */
export type EntryType = string | { [key: string]: JsonValue } | JsonValue[] | null;
/** A criterion description; `null` leaves the label undescribed. */
export type Description = EntryType;

/** A yes/no question with optional descriptions for either outcome. */
export interface NoulQuestion {
  type: "noul";
  /** The question as text, a JSON object, or an array; optional or `null`. */
  instructions?: EntryType;
  /** Optional descriptions of the yes and no outcomes. */
  criteria?: {
    /** Description of the yes outcome. */
    true?: EntryType;
    /** Description of the no outcome. */
    false?: EntryType;
  } | null;
}
/** Labels mapped to descriptions, or `null` for undescribed labels. */
export type ChoiceCriteria = { [label: string]: Description };
/** A question that selects between named alternatives. */
export interface ChoiceQuestion<T extends ChoiceCriteria = ChoiceCriteria> {
  type: "choice";
  instructions?: EntryType;
  criteria: T;
}
/** At least two descriptions indexed by score from zero; `null` leaves a score undescribed. */
export type ScoreCriteria = readonly [EntryType, EntryType, ...EntryType[]];
/** A question that assigns a score using an ordered rubric. */
export interface ScoreQuestion<T extends ScoreCriteria = ScoreCriteria> {
  type: "score";
  instructions?: EntryType;
  criteria: T;
}
/** A question identified by its `type` field. */
export type Question = NoulQuestion | ScoreQuestion | ChoiceQuestion;
/** Questions keyed by the names used to identify their answers. */
export interface Questions {
  [name: string]: Question;
}

/** A yes/no answer. */
export interface NoulResponse {
  readonly type: "noul";
  /** Probability of a yes answer, from zero to one. */
  readonly noul: number;
  /** Only with `kime.extensions`: the confidence of the noul. */
  readonly confidence?: number;
}
/** A selected label and its probabilities. */
export interface ChoiceResponse<T extends ChoiceCriteria = ChoiceCriteria> {
  readonly type: "choice";
  readonly choice: keyof T & string;
  readonly confidence: number;
  readonly probabilities: { readonly [label in keyof T]: number };
}
/** Score keys inferred from the rubric; a fixed-length tuple yields its indices, otherwise `number`. */
export type ScoreOf<T extends ScoreCriteria> = number extends T["length"] ? number : Extract<keyof T, `${number}`>;
/** Rubric descriptions keyed by score. */
export type ScoreLegend<T extends ScoreCriteria> = { readonly [score in ScoreOf<T>]: T[score] };
/** An expected score with its rubric and probabilities. */
export interface ScoreResponse<T extends ScoreCriteria = ScoreCriteria> {
  readonly type: "score";
  readonly score: number;
  readonly confidence: number;
  readonly legend: ScoreLegend<T>;
  readonly probabilities: { readonly [score in ScoreOf<T>]: number };
}
/** The answer type for a question, preserving its criteria keys. */
export type ResultFor<T extends Question> = T extends NoulQuestion
  ? NoulResponse
  : T extends ScoreQuestion<infer S>
    ? ScoreResponse<S>
    : T extends ChoiceQuestion<infer E>
      ? ChoiceResponse<E>
      : never;

/** Token usage for a request. */
export interface Usage {
  readonly input_tokens: number;
  readonly output_tokens: number;
}

/** kime's options for one request, all optional. See spec/03-api.md. */
export interface KimeOptions {
  /** Decimal places the probabilities are rounded to, 0 to 6, or `null` for full precision. Default: 2. */
  precision?: number | null;
  /** `"jev"` (the default) or `"entropy"`, one minus the normalized entropy. */
  confidence?: "jev" | "entropy";
  /** Adds the `kime` block to the response, with timing and per answer details. */
  extensions?: boolean;
  /** Fail with 504 when the answer cannot be ready in this many milliseconds. */
  deadline_ms?: number;
  /** Other options from spec/03, passed through. */
  [option: string]: JsonValue | undefined;
}

/** The `kime` block of a response when `kime.extensions` is on. */
export interface KimeExtensions {
  readonly request_id?: string;
  readonly timing_us?: { readonly [stage: string]: number };
  readonly answers?: { readonly [name: string]: { readonly [field: string]: JsonValue } };
  readonly [field: string]: JsonValue | undefined | object;
}

/** Answers keyed by question name, with model and usage metadata. */
export interface SystemOneResult<Q extends Questions> {
  readonly model: string;
  readonly answers: { readonly [K in keyof Q]: ResultFor<Q[K]> };
  readonly usage: Usage;
  /** Present when the request set `kime.extensions`. */
  readonly kime?: KimeExtensions;
}

/** Metadata for an available model. */
export interface ModelCard {
  readonly name: string;
  readonly description: string;
  readonly release_date: string;
}

/**
 * State and named questions for `systemOne`.
 *
 * Additional properties on a request variable are forwarded, including `null` values.
 */
export interface SystemOneRequest<Q extends Questions = Questions> {
  /** Text, a JSON object or array, or `null` to evaluate. */
  state: EntryType;
  /** Nonempty questions keyed by the names used to identify their answers. */
  questions: Q;
  /** Model override; omitted values inherit `defaultModel`. */
  model?: string;
  /** kime's options; Jev and other servers ignore them. */
  kime?: KimeOptions;
}
/** Request body for `POST /v1/systemone`, with the model resolved. */
export interface SystemOneRequestPayload extends SystemOneRequest {
  model: string;
}

/** Retry configuration. Partial overrides inherit unset fields from the client or SDK defaults. */
export interface RetryPolicy {
  /** Maximum retries after the initial attempt; `0` disables retries. Default: 2. */
  readonly maxRetries: number;
  /** First backoff delay in milliseconds, doubled up to `backoffMaxMs`. Default: 500. */
  readonly backoffInitialMs: number;
  /** Maximum backoff delay in milliseconds. Default: 5000. */
  readonly backoffMaxMs: number;
  /** Fraction of each backoff delay randomly subtracted, from 0 to 1. Default: 0.25. */
  readonly backoffJitter: number;
  /** HTTP status codes to retry. Default: 408, 429, and 500 to 599. */
  readonly httpStatuses: ReadonlySet<number>;
  /** Honor `Retry-After` and `retry-after-ms` up to `maxRetryAfterMs`. Default: true. */
  readonly respectRetryAfter: boolean;
  /** Maximum server retry delay in milliseconds; longer delays use backoff. Default: 60000. */
  readonly maxRetryAfterMs: number;
  /** Retry connection failures, including interrupted response bodies. Default: true. */
  readonly apiConnectionError: boolean;
  /** Whether to retry `APITimeoutError`. Default: true. */
  readonly apiTimeoutError: boolean;
}

/** Per-call options that override client settings. */
export interface RequestOptions {
  /** Cancellation signal for the request and pending retries. */
  signal?: AbortSignal;
  /** Timeout per attempt in milliseconds; there is no total retry budget. */
  timeout?: number;
  /** Retry overrides for this call; omitted fields inherit client settings. */
  retry?: Partial<RetryPolicy>;
  /** Additional headers, merged over `defaultHeaders`. */
  headers?: Record<string, string>;
}

/** HTTP fetch implementation compatible with the global `fetch`. */
export type Fetch = (input: string, init?: RequestInit) => Promise<Response>;
/** Log verbosity; `off` disables logging. */
export type LogLevel = "debug" | "info" | "warn" | "error" | "off";
/** Log methods accepting a message and structured values; compatible with `console`. */
export interface Logger {
  debug(message: string, ...args: unknown[]): void;
  info(message: string, ...args: unknown[]): void;
  warn(message: string, ...args: unknown[]): void;
  error(message: string, ...args: unknown[]): void;
}

/** Client options. Explicit values take precedence over environment variables, then SDK defaults. */
export interface TypeSafeClientConfig {
  /** API key; falls back to `KIME_API_KEY`, then `TYPESAFE_API_KEY`. Required only for api.typesafe.ai. */
  apiKey?: string;
  /** API root; falls back to `KIME_BASE_URL`, then `TYPESAFE_BASE_URL`, then `http://127.0.0.1:8000`. */
  baseURL?: string;
  /** Default model; falls back to `KIME_DEFAULT_MODEL`, then `TYPESAFE_DEFAULT_MODEL`, then `jev-latest`. */
  defaultModel?: string;
  /**
   * Log level; falls back to `KIME_LOG_LEVEL`, then `TYPESAFE_LOG_LEVEL`, then `warn`.
   * `info` logs request summaries; `debug` adds headers and bodies.
   * Known credential headers are redacted; bodies are not.
   */
  logLevel?: LogLevel;
  /** Logger filtered to `logLevel` and above. Default: prefixed `console`. */
  logger?: Logger;
  /** Retry overrides; omitted fields use the defaults in `RetryPolicy`. */
  retry?: Partial<RetryPolicy>;
  /** Timeout per attempt in milliseconds, without a total retry budget. Default: 10000. */
  timeout?: number;
  /** Additional request headers; per-call headers take precedence. */
  defaultHeaders?: Record<string, string>;
  /** Allow browser use, exposing the API key to page users. Default: false. */
  dangerouslyAllowBrowser?: boolean;
  /** Custom HTTP fetch implementation for transport configuration or tests. Default: global `fetch`. */
  fetch?: Fetch;
}

// Environment

/** Environment variable names for client configuration. Explicit options take precedence. */
export const ENV = {
  apiKey: "KIME_API_KEY",
  baseURL: "KIME_BASE_URL",
  defaultModel: "KIME_DEFAULT_MODEL",
  logLevel: "KIME_LOG_LEVEL",
} as const;
/** The TypeSafe SDK's names, read when the kime ones are not set. */
export const TYPESAFE_ENV = {
  apiKey: "TYPESAFE_API_KEY",
  baseURL: "TYPESAFE_BASE_URL",
  defaultModel: "TYPESAFE_DEFAULT_MODEL",
  logLevel: "TYPESAFE_LOG_LEVEL",
} as const;
export type EnvVar = (typeof ENV)[keyof typeof ENV] | (typeof TYPESAFE_ENV)[keyof typeof TYPESAFE_ENV];

const g = globalThis as Record<string, any>;

const readOne = (name: string): string | undefined => {
  const env = g.process?.env ?? g.Deno?.env?.toObject?.();
  if (!env) return undefined;
  const v = env[name];
  return typeof v === "string" ? v.trim() || undefined : undefined;
};

// Deno throws when the environment is read without --allow-env, which counts as unset.
const readEnvSafe = (name: string): string | undefined => {
  try {
    return readOne(name);
  } catch {
    return undefined;
  }
};

/** A trimmed environment value from the kime name, then the TypeSafe one. */
const readEnv = (key: keyof typeof ENV): [string, string] | undefined => {
  for (const name of [ENV[key], TYPESAFE_ENV[key]]) {
    const v = readEnvSafe(name);
    if (v !== undefined) return [v, name];
  }
  return undefined;
};

// Retries

const range = (from: number, to: number) => Array.from({ length: to - from }, (_, i) => from + i);

const DEFAULT_RETRY_POLICY: RetryPolicy = {
  maxRetries: 2,
  backoffInitialMs: 500,
  backoffMaxMs: 5e3,
  backoffJitter: 0.25,
  httpStatuses: new Set([408, 429, ...range(500, 600)]),
  respectRetryAfter: true,
  maxRetryAfterMs: 6e4,
  apiConnectionError: true,
  apiTimeoutError: true,
};

/**
 * Parse `retry-after-ms` or `Retry-After` into milliseconds, preferring `retry-after-ms`.
 * Returns `undefined` when neither header holds a valid delay.
 */
export const parseRetryAfter = (headers: Headers, now = Date.now()): number | undefined => {
  const ms = Number(headers.get("retry-after-ms"));
  if (headers.has("retry-after-ms") && Number.isFinite(ms) && ms >= 0) return ms;
  const raw = headers.get("retry-after");
  if (raw === null) return undefined;
  const seconds = Number(raw);
  if (Number.isFinite(seconds)) return seconds >= 0 ? seconds * 1e3 : undefined;
  const date = Date.parse(raw);
  if (!Number.isNaN(date)) return Math.max(0, date - now);
  return undefined;
};

/** The delay before retry `attempt + 1`: the server's when allowed, else capped backoff with jitter. */
export const retryDelayMs = (
  attempt: number,
  headers: Headers | undefined,
  policy: RetryPolicy = DEFAULT_RETRY_POLICY,
  random: () => number = Math.random,
): number => {
  if (policy.respectRetryAfter && headers !== undefined) {
    const retryAfter = parseRetryAfter(headers);
    if (retryAfter !== undefined && retryAfter <= policy.maxRetryAfterMs) return retryAfter;
  }
  const exponential = Math.min(policy.backoffInitialMs * 2 ** attempt, policy.backoffMaxMs);
  return Math.round(exponential * (1 - random() * policy.backoffJitter));
};

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason);
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal?.reason);
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });

// Errors

/** Base class for SDK errors. */
export class TypeSafeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = new.target.name;
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null;

const describeValidationErrors = (errors: unknown[]): string | undefined => {
  const parts = errors.flatMap((e) => {
    if (!isRecord(e) || typeof e.msg !== "string") return [];
    const loc = Array.isArray(e.loc) ? e.loc.filter((x) => x !== "body").join(".") : "";
    return [loc ? `${loc}: ${e.msg}` : e.msg];
  });
  return parts.length > 0 ? parts.join("; ") : undefined;
};

const extractMessage = (body: unknown): string | undefined => {
  if (typeof body === "string") return body || undefined;
  if (!isRecord(body)) return undefined;
  const { error, message, detail } = body;
  if (typeof error === "string") return error;
  if (isRecord(error) && typeof error.message === "string") return error.message;
  if (typeof message === "string") return message;
  if (typeof detail === "string") return detail;
  if (isRecord(detail) && typeof detail.message === "string") return detail.message;
  if (Array.isArray(detail)) return describeValidationErrors(detail);
  return undefined;
};

const MAX_RAW_BODY_IN_MESSAGE = 200;

/** An unsuccessful HTTP response from the API. */
export class APIError extends TypeSafeError {
  /** HTTP response status code. */
  readonly status: number;
  /** HTTP response headers. */
  readonly headers: Headers;
  /** Parsed JSON, response text, or `undefined` for an empty body. */
  readonly body: unknown;
  /** Request ID from `x-typesafe-request-id`, or `undefined` when absent. */
  readonly requestId: string | undefined;

  constructor(status: number, body: unknown, headers: Headers, message?: string) {
    super(message ?? APIError.describe(status, body));
    this.status = status;
    this.body = body;
    this.headers = headers;
    this.requestId = requestIdFrom(headers);
  }

  private static describe(status: number, body: unknown): string {
    const detail = extractMessage(body);
    if (detail) return `${status} ${detail}`;
    if (body === undefined) return `${status} status code (no body)`;
    const raw = typeof body === "string" ? body : JSON.stringify(body);
    return `${status} ${raw.length > MAX_RAW_BODY_IN_MESSAGE ? `${raw.slice(0, MAX_RAW_BODY_IN_MESSAGE)}…` : raw}`;
  }

  /** Create the error subclass for an HTTP status code. */
  static fromResponse(status: number, body: unknown, headers: Headers): APIError {
    if (status === 400) return new BadRequestError(status, body, headers);
    if (status === 401) return new AuthenticationError(status, body, headers);
    if (status === 403) return new PermissionDeniedError(status, body, headers);
    if (status === 404) return new NotFoundError(status, body, headers);
    if (status === 422) return new UnprocessableEntityError(status, body, headers);
    if (status === 429) return new RateLimitError(status, body, headers);
    if (status >= 500) return new InternalServerError(status, body, headers);
    return new APIError(status, body, headers);
  }
}
/** HTTP 400: the request is invalid. */
export class BadRequestError extends APIError {}
/** HTTP 401: authentication failed. */
export class AuthenticationError extends APIError {}
/** HTTP 403: access is denied. */
export class PermissionDeniedError extends APIError {}
/** HTTP 404: the resource was not found. */
export class NotFoundError extends APIError {}
/** HTTP 422: request validation failed. */
export class UnprocessableEntityError extends APIError {}
/** HTTP 429: the rate limit was exceeded. */
export class RateLimitError extends APIError {
  /** Server retry delay in milliseconds, or `undefined` when absent or invalid. */
  readonly retryAfterMs: number | undefined = parseRetryAfter(this.headers);
}
/** HTTP 5xx: the server failed to handle the request. */
export class InternalServerError extends APIError {}
/** The request or response-body delivery failed (DNS, TLS, connection closed, etc.). */
export class APIConnectionError extends TypeSafeError {
  constructor(message = "Connection error.", options?: ErrorOptions) {
    super(message, options);
  }
}
/** The full response did not arrive within the timeout. A kind of `APIConnectionError`. */
export class APITimeoutError extends APIConnectionError {
  /** Configured timeout in milliseconds. */
  readonly timeoutMs: number;
  constructor(timeoutMs: number, options?: ErrorOptions) {
    super(`Request timed out after ${timeoutMs}ms.`, options);
    this.timeoutMs = timeoutMs;
  }
}
/** The caller cancelled the request through an `AbortSignal`. */
export class APIUserAbortError extends TypeSafeError {
  constructor(message = "Request was aborted.", options?: ErrorOptions) {
    super(message, options);
  }
}

// Logging

/** Supported log levels, from most to least verbose. */
export const LOG_LEVELS: readonly LogLevel[] = ["debug", "info", "warn", "error", "off"];

const isLogLevel = (value: string): value is LogLevel => (LOG_LEVELS as readonly string[]).includes(value);

const parseLogLevel = (value: string, source: string): LogLevel => {
  if (isLogLevel(value)) return value;
  throw new TypeSafeError(`Invalid log level "${value}" from ${source}. Expected one of: ${LOG_LEVELS.join(", ")}.`);
};

const PREFIX = "[kime-sdk]";
const consoleLogger: Logger = {
  debug: (message, ...args) => console.debug(`${PREFIX} ${message}`, ...args),
  info: (message, ...args) => console.info(`${PREFIX} ${message}`, ...args),
  warn: (message, ...args) => console.warn(`${PREFIX} ${message}`, ...args),
  error: (message, ...args) => console.error(`${PREFIX} ${message}`, ...args),
};
const RANK: Record<LogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3, off: 4 };
const drop = () => {};
const withLevel = (sink: Logger, level: LogLevel): Logger => {
  const enabled = (at: LogLevel) => RANK[at] >= RANK[level];
  return {
    debug: enabled("debug") ? (message, ...args) => sink.debug(message, ...args) : drop,
    info: enabled("info") ? (message, ...args) => sink.info(message, ...args) : drop,
    warn: enabled("warn") ? (message, ...args) => sink.warn(message, ...args) : drop,
    error: enabled("error") ? (message, ...args) => sink.error(message, ...args) : drop,
  };
};

const KEY_HEADERS = new Set(["authorization", "proxy-authorization", "x-api-key"]);
const OPAQUE_HEADERS = new Set(["cookie", "set-cookie"]);
const redactKey = (value: string) => {
  const [scheme, secret] = value.includes(" ") ? value.split(/\s+/, 2) : [undefined, value];
  const tail = secret && secret.length > 8 ? secret.slice(-4) : "";
  return `${scheme ? `${scheme} ` : ""}***${tail}`;
};
const redact = (name: string, value: string) => {
  const lower = name.toLowerCase();
  if (KEY_HEADERS.has(lower)) return redactKey(value);
  if (OPAQUE_HEADERS.has(lower)) return "***";
  return value;
};
const redactHeaders = (headers: Record<string, string>) =>
  Object.fromEntries(Object.entries(headers).map(([name, value]) => [name, redact(name, value)]));

// Questions

/** Create a yes/no question with optional descriptions for either outcome. */
export const noul = (instructions: EntryType = null, criteria?: NoulQuestion["criteria"]): NoulQuestion => ({
  type: "noul",
  instructions,
  criteria,
});

/** Create a score question using an ordered rubric of at least two descriptions. */
export const score = <const T extends ScoreCriteria>(instructions: EntryType, criteria: T): ScoreQuestion<T> => {
  if (!Array.isArray(criteria)) {
    throw new TypeSafeError("Score criteria must be a list of descriptions indexed by score from zero, not a map.");
  }
  return { type: "score", instructions, criteria };
};

/** Create a question that selects between named alternatives. */
export const choice = <const T extends ChoiceCriteria>(instructions: EntryType, criteria: T): ChoiceQuestion<T> => {
  if (Array.isArray(criteria)) {
    throw new TypeSafeError("Choice criteria must be a map of labels to descriptions, not a list.");
  }
  return { type: "choice", instructions, criteria };
};

const validateQuestions = (questions: Questions) => {
  if (Object.keys(questions).length === 0) throw new TypeSafeError("At least one question is required.");
  for (const [name, question] of Object.entries(questions)) {
    if (question.type !== "score") continue;
    if (!Array.isArray(question.criteria)) {
      throw new TypeSafeError(
        `Score question "${name}" has criteria that are not a list; score criteria must be a list of descriptions indexed by score from zero.`,
      );
    }
    if (question.criteria.length < 2) {
      throw new TypeSafeError(
        `Score question "${name}" has ${question.criteria.length} criteria; at least two scores are required.`,
      );
    }
  }
};

// Models

/** Per-call transport options with an optional JSON body. */
interface RawRequestOptions extends RequestOptions {
  body?: unknown;
}
interface Transport {
  request<T>(method: "GET" | "POST", path: string, options?: RawRequestOptions): APIPromise<T>;
  readonly defaultModel: string;
}

const unwrapModels = (wire: { models?: unknown }): ModelCard[] => {
  if (Array.isArray(wire?.models)) return wire.models as ModelCard[];
  throw new TypeSafeError("Unexpected response shape from GET /v1/models; expected { models: [...] }.");
};

/** Access to the Models API resource. */
export class Models {
  #transport: Transport;
  constructor(transport: Transport) {
    this.#transport = transport;
  }
  /** List the models the server offers. */
  list(options: RequestOptions = {}): APIPromise<ModelCard[]> {
    return this.#transport.request<{ models?: unknown }>("GET", "/v1/models", options).map(unwrapModels);
  }
}

// Client

const isBrowser = () =>
  typeof g.window !== "undefined" && typeof g.window.document !== "undefined" && typeof g.navigator !== "undefined";

const describeRuntime = () => {
  const platform = g.process?.platform && g.process?.arch ? ` (${g.process.platform}; ${g.process.arch})` : "";
  if (g.Bun?.version) return `bun/${g.Bun.version}${platform}`;
  if (g.Deno?.version?.deno) return `deno/${g.Deno.version.deno}${platform}`;
  if (g.EdgeRuntime !== undefined) return "vercel-edge";
  if (g.navigator?.userAgent === "Cloudflare-Workers") return "cloudflare-workers";
  if (g.process?.versions?.node) return `node/${g.process.versions.node}${platform}`;
  if (isBrowser()) return "browser";
  return "unknown";
};

/** The version of this package. */
export const VERSION = "0.0.21";
/** The TypeSafe SDK version this package follows. */
export const TYPESAFE_SDK_VERSION = "0.6.0";
/** The API that requires a key. */
export const TYPESAFE_BASE_URL = "https://api.typesafe.ai";
/** Where `kime serve` listens by default. */
export const DEFAULT_BASE_URL = "http://127.0.0.1:8000";

const defaultFetch: Fetch = (input, init) => globalThis.fetch(input, init);

const assertNonNegativeInteger = (name: string, value: number) => {
  if (!Number.isInteger(value) || value < 0) {
    throw new TypeSafeError(`\`${name}\` must be a non-negative integer, got ${String(value)}.`);
  }
  return value;
};
const assertPositiveMs = (name: string, value: number) => {
  if (!Number.isFinite(value) || value <= 0) {
    throw new TypeSafeError(`\`${name}\` must be a positive number of milliseconds, got ${String(value)}.`);
  }
  return value;
};
const assertNonNegativeMs = (name: string, value: number) => {
  if (!Number.isFinite(value) || value < 0) {
    throw new TypeSafeError(`\`${name}\` must be a non-negative number of milliseconds, got ${String(value)}.`);
  }
  return value;
};
const assertFraction = (name: string, value: number) => {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new TypeSafeError(`\`${name}\` must be between 0 and 1, got ${String(value)}.`);
  }
  return value;
};
const assertStatusSet = (name: string, statuses: ReadonlySet<number>) => {
  for (const status of statuses) {
    if (!Number.isInteger(status) || status < 100 || status > 999) {
      throw new TypeSafeError(`\`${name}\` must contain HTTP status codes, got ${String(status)}.`);
    }
  }
  return statuses;
};

const resolveRetryPolicy = (base: RetryPolicy, overrides?: Partial<RetryPolicy>): RetryPolicy => {
  const o = overrides ?? {};
  return {
    maxRetries: o.maxRetries === undefined ? base.maxRetries : assertNonNegativeInteger("retry.maxRetries", o.maxRetries),
    backoffInitialMs:
      o.backoffInitialMs === undefined ? base.backoffInitialMs : assertNonNegativeMs("retry.backoffInitialMs", o.backoffInitialMs),
    backoffMaxMs: o.backoffMaxMs === undefined ? base.backoffMaxMs : assertNonNegativeMs("retry.backoffMaxMs", o.backoffMaxMs),
    backoffJitter: o.backoffJitter === undefined ? base.backoffJitter : assertFraction("retry.backoffJitter", o.backoffJitter),
    httpStatuses: new Set(
      o.httpStatuses === undefined ? base.httpStatuses : assertStatusSet("retry.httpStatuses", o.httpStatuses),
    ),
    respectRetryAfter: o.respectRetryAfter ?? base.respectRetryAfter,
    maxRetryAfterMs:
      o.maxRetryAfterMs === undefined ? base.maxRetryAfterMs : assertNonNegativeMs("retry.maxRetryAfterMs", o.maxRetryAfterMs),
    apiConnectionError: o.apiConnectionError ?? base.apiConnectionError,
    apiTimeoutError: o.apiTimeoutError ?? base.apiTimeoutError,
  };
};

const isRetryableError = (err: unknown, policy: RetryPolicy) => {
  if (err instanceof APITimeoutError) return policy.apiTimeoutError;
  if (err instanceof APIConnectionError) return policy.apiConnectionError;
  return false;
};

const resolveLogLevel = (fromCode: LogLevel | undefined): LogLevel => {
  if (fromCode !== undefined) return parseLogLevel(fromCode, "the `logLevel` option");
  const fromEnv = readEnv("logLevel");
  if (fromEnv !== undefined) return parseLogLevel(fromEnv[0], fromEnv[1]);
  return "warn";
};

const stripTrailingSlashes = (url: string) => url.replace(/\/+$/, "");

/** Last value wins regardless of casing; undefined removes a header. */
const mergeHeaders = (...sources: Record<string, string | undefined>[]): Record<string, string> => {
  const entries = new Map<string, [string, string]>();
  for (const source of sources) {
    for (const [name, value] of Object.entries(source)) {
      if (value === undefined) entries.delete(name.toLowerCase());
      else entries.set(name.toLowerCase(), [name, value]);
    }
  }
  return Object.fromEntries(entries.values());
};

/** Drain a clone so the original response keeps its metadata and a readable, buffered body. */
const bufferResponse = async (response: Response, signal: AbortSignal) => {
  const reader = response.clone().body?.getReader();
  if (!reader) return;
  const cancel = () => {
    reader.cancel(signal.reason).catch(() => {});
    response.body?.cancel(signal.reason).catch(() => {});
  };
  signal.addEventListener("abort", cancel, { once: true });
  try {
    if (signal.aborted) cancel();
    signal.throwIfAborted();
    while (!(await reader.read()).done) signal.throwIfAborted();
    signal.throwIfAborted();
  } finally {
    signal.removeEventListener("abort", cancel);
    reader.releaseLock();
  }
};

const parseBody = async (res: Response): Promise<unknown> => {
  const text = await res.text();
  if (text.length === 0) return undefined;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
};

const RUNTIME = describeRuntime();

interface ResolvedRequest {
  method: "GET" | "POST";
  path: string;
  body: unknown;
  headers: Record<string, string>;
  signal: AbortSignal | undefined;
  timeout: number;
  retry: RetryPolicy;
}

/** Client for kime-serve, Jev or any System One endpoint. */
export class TypeSafeClient {
  #apiKey: string | undefined;
  /** API root with trailing slashes removed. */
  readonly baseURL: string;
  /** Model used when a request omits `model`. */
  readonly defaultModel: string;
  /** Configured log verbosity. */
  readonly logLevel: LogLevel;
  /** The configured logger, filtered to `logLevel`. */
  readonly logger: Logger;
  /** Retry settings with constructor overrides applied. */
  readonly retry: RetryPolicy;
  /** Timeout per attempt in milliseconds. */
  readonly timeout: number;
  /** Additional headers sent with each request. */
  readonly defaultHeaders: Readonly<Record<string, string>>;
  /** HTTP fetch implementation. */
  readonly fetch: Fetch;
  /** The models the server offers. */
  readonly models: Models;
  #requestCount = 0;

  /**
   * Explicit options take precedence over environment variables, then the defaults. Empty or
   * whitespace-only environment values are ignored.
   *
   * @throws {TypeSafeError} api.typesafe.ai without a key, invalid configuration, or a browser.
   */
  constructor(config: TypeSafeClientConfig = {}) {
    if (isBrowser() && !config.dangerouslyAllowBrowser) {
      throw new TypeSafeError(
        "TypeSafeClient is running in a browser, which would expose your API key to anyone using the page. Call the API from a server instead, or pass `dangerouslyAllowBrowser: true` if you understand the risk.",
      );
    }
    this.baseURL = stripTrailingSlashes(config.baseURL ?? readEnv("baseURL")?.[0] ?? DEFAULT_BASE_URL);
    this.#apiKey = config.apiKey ?? readEnv("apiKey")?.[0];
    if (!this.#apiKey && this.baseURL === TYPESAFE_BASE_URL) {
      throw new TypeSafeError(
        `No API key was provided. Pass \`apiKey\` to the TypeSafeClient constructor or set the ${ENV.apiKey} environment variable.`,
      );
    }
    this.defaultModel = config.defaultModel ?? readEnv("defaultModel")?.[0] ?? "jev-latest";
    this.logLevel = resolveLogLevel(config.logLevel);
    this.logger = withLevel(config.logger ?? consoleLogger, this.logLevel);
    this.retry = resolveRetryPolicy(DEFAULT_RETRY_POLICY, config.retry);
    this.timeout = assertPositiveMs("timeout", config.timeout ?? 1e4);
    this.defaultHeaders = { ...config.defaultHeaders };
    if (config.fetch === undefined && typeof globalThis.fetch !== "function") {
      throw new TypeSafeError(
        "No global `fetch` is available in this runtime. Pass a `fetch` implementation to the TypeSafeClient constructor.",
      );
    }
    this.fetch = config.fetch ?? defaultFetch;
    this.models = new Models({
      request: (method, path, options) => this.#request(method, path, options),
      defaultModel: this.defaultModel,
    });
  }

  /**
   * Answer named questions about text or structured state.
   *
   * @throws {TypeSafeError} Questions are empty, or score criteria are not a list of at least two entries.
   * @throws {APIError} The server returns a non-2xx response after retries.
   * @throws {APIConnectionError} The request cannot connect or times out after retries.
   * @throws {APIUserAbortError} The caller aborts the request.
   *
   * @example
   * ```ts
   * const { answers } = await client.systemOne({
   *   state: "I was charged twice. Please help.",
   *   questions: { billing: noul("Is this about billing?") },
   * });
   * console.log(answers.billing.noul);
   * ```
   */
  systemOne<const Q extends Questions>(
    request: SystemOneRequest<Q>,
    options: RequestOptions = {},
  ): APIPromise<SystemOneResult<Q>> {
    validateQuestions(request.questions);
    const body = { ...request, model: request.model ?? this.defaultModel };
    return this.#request("POST", "/v1/systemone", { ...options, body });
  }

  #request<T>(method: "GET" | "POST", path: string, options: RawRequestOptions = {}): APIPromise<T> {
    const resolved: ResolvedRequest = {
      method,
      path,
      body: options.body,
      headers: mergeHeaders(this.defaultHeaders, options.headers ?? {}),
      signal: options.signal,
      timeout: options.timeout === undefined ? this.timeout : assertPositiveMs("timeout", options.timeout),
      retry: resolveRetryPolicy(this.retry, options.retry),
    };
    const tag = `#${++this.#requestCount} ${method} ${path}`;
    return new APIPromise(this.#fetchWithRetries(tag, resolved), async (res) => {
      const parsed = await parseBody(res);
      this.logger.debug(`${tag} <- body`, parsed);
      return parsed as T;
    });
  }

  async #fetchWithRetries(tag: string, req: ResolvedRequest): Promise<Response> {
    const url = `${this.baseURL}${req.path}`;
    const headers = mergeHeaders(req.headers, {
      Authorization: this.#apiKey ? `Bearer ${this.#apiKey}` : undefined,
      Accept: "application/json",
      "User-Agent": `kime-sdk/${VERSION}`,
      "X-TypeSafe-SDK": `kime-sdk/${VERSION}`,
      "X-TypeSafe-Runtime": RUNTIME,
      "Content-Type": req.body === undefined ? undefined : "application/json",
      "X-TypeSafe-Retry-Count": undefined,
    });
    const body = req.body === undefined ? undefined : JSON.stringify(req.body);
    for (let attempt = 0; ; attempt++) {
      const retriesLeft = req.retry.maxRetries - attempt;
      const attemptHeaders = attempt === 0 ? headers : { ...headers, "X-TypeSafe-Retry-Count": String(attempt) };
      this.logger.debug(`${tag} -> ${url}`, { headers: redactHeaders(attemptHeaders), body: req.body });
      const started = Date.now();
      let res: Response;
      try {
        res = await this.#attempt(tag, url, { method: req.method, headers: attemptHeaders, body }, req);
      } catch (err) {
        if (err instanceof APIUserAbortError || retriesLeft <= 0) throw err;
        if (!isRetryableError(err, req.retry)) throw err;
        await this.#backOff(tag, attempt, retriesLeft, (err as Error).message, undefined, req);
        continue;
      }
      const requestId = requestIdFrom(res.headers);
      this.logger.info(`${tag} <- ${res.status} in ${Date.now() - started}ms${requestId ? ` (request ${requestId})` : ""}`);
      if (res.ok) return res;
      const errorBody = await parseBody(res);
      this.logger.debug(`${tag} <- error body`, errorBody);
      const error = APIError.fromResponse(res.status, errorBody, res.headers);
      if (retriesLeft <= 0 || !req.retry.httpStatuses.has(res.status)) throw error;
      await this.#backOff(tag, attempt, retriesLeft, `${res.status}`, res.headers, req);
    }
  }

  // One round trip, body included, under the timeout. The caller's signal and the timer abort
  // the same controller, and which one fired picks the error class.
  async #attempt(tag: string, url: string, init: RequestInit, { signal, timeout }: ResolvedRequest): Promise<Response> {
    const controller = new AbortController();
    const abortFromCaller = () => controller.abort(signal?.reason);
    if (signal?.aborted) abortFromCaller();
    signal?.addEventListener("abort", abortFromCaller, { once: true });
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, timeout);
    const started = Date.now();
    const elapsed = () => `${Date.now() - started}ms`;
    try {
      const response = await this.fetch(url, { ...init, signal: controller.signal });
      await bufferResponse(response, controller.signal);
      return response;
    } catch (err) {
      if (signal?.aborted) {
        this.logger.info(`${tag} aborted by caller after ${elapsed()}`);
        throw new APIUserAbortError(undefined, { cause: err });
      }
      if (timedOut) {
        this.logger.info(`${tag} timed out after ${elapsed()}`);
        throw new APITimeoutError(timeout, { cause: err });
      }
      this.logger.info(`${tag} connection error after ${elapsed()}`, err);
      throw new APIConnectionError(err instanceof Error ? `Connection error: ${err.message}` : undefined, { cause: err });
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abortFromCaller);
    }
  }

  async #backOff(
    tag: string,
    attempt: number,
    retriesLeft: number,
    reason: string,
    headers: Headers | undefined,
    { retry, signal }: ResolvedRequest,
  ) {
    const delay = retryDelayMs(attempt, headers, retry);
    this.logger.info(`${tag} retrying in ${delay}ms (retry ${attempt + 1}/${attempt + retriesLeft}) after ${reason}`);
    try {
      await sleep(delay, signal);
    } catch (err) {
      this.logger.info(`${tag} aborted by caller while waiting to retry`);
      throw new APIUserAbortError(undefined, { cause: err });
    }
  }
}
