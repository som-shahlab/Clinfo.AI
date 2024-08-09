from typing import Union
from pydantic import BaseModel


class Payload(BaseModel):
    question: str
