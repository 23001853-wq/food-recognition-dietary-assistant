# Tests

See [executed test report](../TEST_REPORT.md). Unit, PostgreSQL integration, HTTP smoke, and CPU ML smoke are implemented. Run pytest from the repository root. Integration tests require TEST_DATABASE_URL with a separate database ending in _test; no application DB reset is allowed. ML smoke tests use temporary synthetic fixtures and random CNN initialization, never external data or pretrained downloads. Install the ml and dev extras to run the complete suite.
