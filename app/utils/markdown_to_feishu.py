import re


class MarkdownToFeishuConverter:
    def convert_to_interactive(self, text: str) -> dict:
        if not text or not text.strip():
            return self._empty_interactive()
        
        return {
            "schema": "2.0",
            "config": {
                "update_multi": True
            },
            "body": {
                "elements": [
                    {
                        "tag": "markdown",
                        "content": self._escape_for_card(text)
                    }
                ]
            }
        }
    
    def _empty_interactive(self) -> dict:
        return {
            "schema": "2.0",
            "config": {
                "update_multi": True
            },
            "body": {
                "elements": [
                    {
                        "tag": "markdown",
                        "content": ""
                    }
                ]
            }
        }
    
    def _escape_for_card(self, text: str) -> str:
        text = text.replace('\\', '&#92;')
        return text