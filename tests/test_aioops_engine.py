import unittest

from aioops_engine import CircuitBreaker, OllamaClient, parse_log_line


class ParseLogLineTests(unittest.TestCase):
    def test_parses_structured_error_event(self):
        event = parse_log_line(
            '{"timestamp":"2026-11-03T08:30:00Z","level":"ERROR",'
            '"service":"payments","error_code":"DB_POOL","message":"pool exhausted"}'
        )
        self.assertEqual(event["service"], "payments")
        self.assertEqual(event["error_code"], "DB_POOL")

    def test_ignores_non_errors_and_parses_raw_errors(self):
        self.assertIsNone(parse_log_line('{"level":"INFO","message":"ready"}'))
        self.assertIsNone(parse_log_line("server started successfully"))
        event = parse_log_line("2026-11-03 CRITICAL payments: connection refused")
        self.assertEqual(event["level"], "CRITICAL")


class CircuitBreakerTests(unittest.TestCase):
    def setUp(self):
        self.breaker = CircuitBreaker(minimum_confidence=0.75)
        self.recommendation = {
            "summary": "Check the service state.",
            "command": "systemctl status payments-api.service",
            "confidence": 0.9,
            "confidence_flags": {
                "diagnosis": True,
                "runbook_support": True,
                "command_safety": True,
            },
        }

    def test_accepts_safe_recommendation_without_executing_it(self):
        self.assertIsNone(self.breaker.validate(self.recommendation))

    def test_blocks_high_risk_command_and_opens_breaker(self):
        self.recommendation["command"] = "chmod 777 /tmp/service"
        self.assertIn("High-risk", self.breaker.validate(self.recommendation))
        self.assertTrue(self.breaker.is_open)
        self.assertIsNotNone(self.breaker.validate({"command": "systemctl status x"}))

    def test_blocks_risky_command_wrapped_in_shell_argument(self):
        self.recommendation["command"] = 'bash -c "rm -rf /tmp/service"'
        self.assertIn("High-risk", self.breaker.validate(self.recommendation))

    def test_low_confidence_opens_breaker(self):
        self.recommendation["confidence"] = 0.4
        self.assertIn("confidence", self.breaker.validate(self.recommendation))
        self.assertTrue(self.breaker.is_open)

    def test_false_confidence_flag_opens_breaker(self):
        self.recommendation["confidence_flags"]["runbook_support"] = False
        self.assertIn("confidence", self.breaker.validate(self.recommendation))

    def test_rejects_non_finite_confidence(self):
        self.recommendation["confidence"] = float("nan")
        self.assertIn("schema", self.breaker.validate(self.recommendation))

    def test_rejects_shell_chaining(self):
        self.recommendation["command"] = "systemctl status x; systemctl restart x"
        self.assertIn("Shell operators", self.breaker.validate(self.recommendation))


class OllamaLocalOnlyTests(unittest.TestCase):
    def test_rejects_remote_ollama_host(self):
        with self.assertRaises(ValueError):
            OllamaClient("https://example.com", "qwen2.5:7b", "nomic-embed-text")


if __name__ == "__main__":
    unittest.main()
