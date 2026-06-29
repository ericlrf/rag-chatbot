import json
from datetime import datetime, timezone

from app.config import Settings
from app.schemas import FeedbackRequest


class FeedbackService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.feedback_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, feedback: FeedbackRequest) -> None:
        payload = feedback.model_dump()
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
        with self.settings.feedback_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
