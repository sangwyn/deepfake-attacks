# Versioned experiment ledger

This directory holds compact evidence needed to understand and reproduce a
campaign: resolved configurations, manifest snapshots or hashes, immutable job
specifications, status JSON, provenance, summaries, norm audits, and reviews.

Do not commit generated images, model weights, queue databases, API keys, full
agent transcripts, or large logs. Their external paths and SHA-256 digests are
recorded here instead. A status may say `passed` only after the deterministic
verifier accepts the corresponding attempt directory.

Every real run uses a unique path:

`tracking/runs/<campaign-id>/<task-id>/attempt-<number>/`

Existing attempts are immutable. Retry by incrementing the attempt number.
