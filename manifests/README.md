# Dataset manifests

`celebA/test_{fake,real}.jsonl` freeze the 200-image evaluation set. Paths are
relative to the configured `TEST` directory; labels are still verified from
`TEST_FAKE` (1) and `TEST_REAL` (0), and every image is checked against its
SHA-256 digest before evaluation. Image data is intentionally not committed.

Treat these manifests as immutable protocol artifacts. Create a separately
named manifest if the dataset snapshot changes.
