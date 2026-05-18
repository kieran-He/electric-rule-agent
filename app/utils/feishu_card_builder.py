from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FeishuCardBuilder:
    def build_example_questions_card(self, questions: list[dict], message_id: Optional[str] = None) -> dict:
        elements = []
        
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": "**示例问题**\n点击按钮选择问题："
            }
        })
        
        for i, q in enumerate(questions, 1):
            question_text = q.get("question", "")
            question_id = q.get("question_id", f"q_{i}")
            
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"{i}. {question_text[:35]}{'...' if len(question_text) > 35 else ''}"
                },
                "extra": {
                    "tag": "button",
                    "text": {
                        "tag": "plain_text",
                        "content": "提问"
                    },
                    "type": "primary",
                    "size": "small",
                    "value": {
                        "action": "ask_question",
                        "question_id": question_id,
                        "question": question_text
                    }
                }
            })
        
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": ""
            },
            "extra": {
                "tag": "button",
                "text": {
                    "tag": "plain_text",
                    "content": "换一批"
                },
                "type": "default",
                "size": "small",
                "value": {
                    "action": "refresh"
                }
            }
        })
        
        return {
            "schema": "2.0",
            "config": {
                "update_multi": True
            },
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": "示例问题"
                },
                "template": "blue"
            },
            "body": {
                "elements": elements
            }
        }
    
    def parse_action_value(self, value) -> Optional[dict]:
        if isinstance(value, dict):
            return value
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse action value: {e}")
            return None