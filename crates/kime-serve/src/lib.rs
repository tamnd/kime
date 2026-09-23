//! The HTTP server. `/v1/systemone` and the rest of the API in spec/03-api.md, keys, rate limits, overload handling and metrics, as a thin layer over `kime-engine`. See spec/11-serving.md.

#![forbid(unsafe_code)]
