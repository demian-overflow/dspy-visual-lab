import json
import logging


class Logger:


    def __init__(self):

        self.logger = logging.getLogger(
            "creative-agent"
        )


    def info(
        self,
        event,
        data=None
    ):

        self.logger.info(
            json.dumps(
                {
                    "event": event,
                    "data": data or {}
                }
            )
        )


    def error(
        self,
        event,
        error
    ):

        self.logger.error(
            json.dumps(
                {
                    "event": event,
                    "error": str(error)
                }
            )
        )
