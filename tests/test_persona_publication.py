"""Read-only publication identity reports must never echo authentication data."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("publish_persona_vectors_modal", ROOT / "scripts/publish_persona_vectors_modal.py")
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


class PublicationIdentityTests(unittest.TestCase):
    def identity(self, payload):
        fake_hub = ModuleType("huggingface_hub")
        fake_hub.HfApi = lambda: type("FakeApi", (), {"whoami": lambda self: payload})()
        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            return publisher.identity_remote()

    def test_oauth_missing_role_is_unknown_and_credentials_are_omitted(self):
        result = self.identity({"name": "example-owner", "auth": {
            "type": "oauth", "oauth": {"access_token": "private-token", "scopes": "private-scope"}}})
        self.assertEqual(result, {"owner": "example-owner", "auth_type": "oauth", "auth_role": None,
                                  "write_role_reported": None})
        self.assertNotIn("private", repr(result))

    def test_read_write_and_fine_grained_roles_are_distinguished(self):
        for role, writable in (("read", False), ("write", True), ("fineGrained", None)):
            with self.subTest(role=role):
                result = self.identity({"name": "example-owner", "auth": {
                    "type": "access_token", "accessToken": {"role": role, "token": "private-token"}}})
                self.assertEqual(result["auth_role"], role)
                self.assertEqual(result["write_role_reported"], writable)
                self.assertNotIn("private-token", repr(result))

    def test_unrecognized_auth_values_are_not_echoed(self):
        result = self.identity({"name": "example-owner", "auth": {
            "type": {"credential": "private-token"}, "accessToken": {"role": "private-token"}}})
        self.assertIsNone(result["auth_type"])
        self.assertIsNone(result["auth_role"])
        self.assertNotIn("private-token", repr(result))


if __name__ == "__main__":
    unittest.main()
