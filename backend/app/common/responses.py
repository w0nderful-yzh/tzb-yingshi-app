from pydantic import BaseModel


class ApiResponse[DataT](BaseModel):
    code: int = 0
    message: str = "success"
    data: DataT | None = None
    request_id: str
