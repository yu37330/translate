"""Bedrockクライアントのモックによるユニットテスト。実AWSへは送信しない。"""
import unittest
from unittest.mock import patch

import bedrock_text


class FakeClient:
    def __init__(self):
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {"output": {"message": {"content": [{"text": "  検証結果  "}]}}}


class BedrockTextTest(unittest.TestCase):
    def test_disabled_without_model(self):
        with patch.object(bedrock_text, "BEDROCK_MODEL_ID", ""):
            with self.assertRaises(RuntimeError):
                bedrock_text.generate_text("指示", "入力")

    def test_converse_format_and_model(self):
        fake = FakeClient()
        with patch.object(bedrock_text, "BEDROCK_MODEL_ID", "test-profile"), patch.object(
            bedrock_text, "_runtime_client", return_value=fake
        ):
            result = bedrock_text.generate_text("指示", "入力", 120)
        self.assertEqual(result, "検証結果")
        self.assertEqual(fake.calls[0]["modelId"], "test-profile")
        self.assertEqual(fake.calls[0]["messages"][0]["content"][0]["text"], "入力")
        self.assertEqual(fake.calls[0]["inferenceConfig"], {"maxTokens": 120})

    def test_translation_opt_out(self):
        fake = FakeClient()
        with patch.object(bedrock_text, "BEDROCK_MODEL_ID", "test-profile"), patch.object(
            bedrock_text, "_runtime_client", return_value=fake
        ):
            self.assertEqual(bedrock_text.translate_text("Hello", "en", "none"), "")
            self.assertEqual(len(fake.calls), 0)
            result = bedrock_text.translate_text("Hello", "en", "ja")
        self.assertEqual(result, "検証結果")
        self.assertEqual(len(fake.calls), 1)
        self.assertIn("日本語", fake.calls[0]["messages"][0]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
