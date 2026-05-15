import re


class MarkdownToFeishuConverter:
    def convert_to_interactive(self, text: str, image_keys: list = None) -> dict:
        if not text or not text.strip():
            return self._empty_interactive()
        
        elements = [
            {
                "tag": "markdown",
                "content": self._escape_for_card(text)
            }
        ]
        
        if image_keys:
            for key in image_keys:
                elements.append({
                    "tag": "img",
                    "img_key": key,
                    "alt": {"tag": "plain_text", "content": "数据图表"}
                })
        
        return {
            "schema": "2.0",
            "config": {
                "update_multi": True
            },
            "body": {
                "elements": elements
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