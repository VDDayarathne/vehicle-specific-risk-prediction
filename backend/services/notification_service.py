"""
backend/services/notification_service.py
Firebase Cloud Messaging helper for sending push alerts to mobile devices.
"""
import json
from functools import lru_cache
from typing import Optional

import firebase_admin
from firebase_admin import credentials, messaging

from backend.config.settings import FIREBASE_SERVICE_ACCOUNT_KEY, FCM_DEFAULT_TOPIC


class NotificationService:
    """Singleton wrapper around Firebase Admin SDK."""

    _instance: "NotificationService | None" = None

    def __init__(self) -> None:
        self._initialized = False
        self._ensure_initialized()

    @classmethod
    def get_instance(cls) -> "NotificationService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_ready(self) -> bool:
        return self._initialized

    def _ensure_initialized(self) -> None:
        if firebase_admin._apps:
            self._initialized = True
            return

        if not FIREBASE_SERVICE_ACCOUNT_KEY:
            self._initialized = False
            return

        try:
            if FIREBASE_SERVICE_ACCOUNT_KEY.strip().startswith("{"):
                cred_data = json.loads(FIREBASE_SERVICE_ACCOUNT_KEY)
                cred = credentials.Certificate(cred_data)
            else:
                cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_KEY)
            firebase_admin.initialize_app(cred)
            self._initialized = True
        except Exception as exc:
            print(f"[NotificationService] Firebase initialization failed: {exc}")
            self._initialized = False

    def send_device_alert(
        self,
        token: str,
        title: str,
        body: str,
        risk_level: str,
        recommended_speed_kmh: Optional[int] = None,
    ) -> Optional[str]:
        """Send a single-device alert via FCM and return the message ID on success."""
        if not self._initialized or not token:
            return None

        message = messaging.Message(
            token=token,
            notification=messaging.Notification(title=title, body=body),
            data={
                "risk_level": risk_level,
                "recommended_speed_kmh": str(recommended_speed_kmh or ""),
                "topic": FCM_DEFAULT_TOPIC,
            },
        )
        result = messaging.send(message)
        return result

    def send_topic_alert(
        self,
        title: str,
        body: str,
        risk_level: str,
        recommended_speed_kmh: Optional[int] = None,
        topic: Optional[str] = None,
    ) -> Optional[str]:
        """Send a topic-based alert to subscribed devices."""
        if not self._initialized:
            return None

        message = messaging.Message(
            topic=topic or FCM_DEFAULT_TOPIC,
            notification=messaging.Notification(title=title, body=body),
            data={
                "risk_level": risk_level,
                "recommended_speed_kmh": str(recommended_speed_kmh or ""),
            },
        )
        return messaging.send(message)
