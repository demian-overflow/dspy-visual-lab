import hashlib
import json


class Cache:


    def __init__(self):

        self.data = {}



    def key(
        self,
        value
    ):

        return hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True
            ).encode()
        ).hexdigest()



    def get(
        self,
        value
    ):

        return self.data.get(
            self.key(value)
        )



    def put(
        self,
        value,
        result
    ):

        self.data[
            self.key(value)
        ] = result
