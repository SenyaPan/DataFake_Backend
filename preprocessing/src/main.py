from fastapi import FastAPI, Depends

from preprocessing.src.files.router_v1 import router as router_v1
from preprocessing.src.files.router_v2 import router as router_v2

app = FastAPI(
    title="Preprocessing",
    description="This API allows you to upload a file to our service and get the results of it's analysis.",
    summary="API of main microservice for the DatAFake work.",
    version="1.0",
    root_path="/api"
)

app.include_router(router_v1)
app.include_router(router_v2)


# app.include_router(router_operation)

# current_user = fastapi_users.current_user()
#
#
# @app.get("/protected-route")
# def protected_route(user: User = Depends(current_user)):
#     return f"Hello, {user.nickname}"
#
#
# @app.get("/unprotected-route")
# def unprotected_route(user: User = Depends(current_user)):
#     return f"Hello, Someone!"

# fake_users = [
#     {"id": 1, "role": "admin", "name": "Bob"},
#     {"id": 2, "role": "investor", "name": "John"},
#     {"id": 3, "role": "trader", "name": "Matt"},
# ]
#
#
# class DegreeType(Enum):
#     newbie = "newbie"
#     expert = "expert"
#
#
# class Degree(BaseModel):
#     id: int
#     created_at: datetime
#     type_degree: DegreeType
#
#
# class User(BaseModel):
#     id: int
#     role: str
#     name: str
#     degree: Optional[List[Degree]] = []
#
#
# @app.get("/users/{user_id}", response_model=List[User]) #сначала название приложения
# def hello(user_id: int):
#     return [user for user in fake_users if user.get("id") == user_id]
#
#
# fake_trades = [
#     {'id': 1, 'users_id': 1, 'currency': 'BTC', "side": "buy", 'price': 123, 'amount': 2.12},
#     {'id': 2, 'users_id': 1, 'currency': 'BTC', "side": "buy", 'price': 123, 'amount': 2.12},
# ]
#
#
# @app.get("/trades")
# def get_trades(limit: int = 1, offset: int = 0):
#     return fake_trades[offset:][:limit]
#
#
# @app.post("/users/{user_id}")
# def change_user_name(user_id: int, new_name: str):
#     current_user = list(filter(lambda user: user.get('id') == user_id, fake_users))[0]
#     current_user['name'] = new_name
#     return {'status': 200, 'data': current_user}
#
#
# class Trade(BaseModel):
#     id: int
#     user_id: int
#     currency: str = Field(max_length=5)
#     side: str
#     price: float = Field(ge=0)
#     amount: float
#
#
# @app.post("/trades")
# def add_trades(trades: List[Trade]):
#     fake_trades.extend(trades)
#     return {"status": 200, "data": fake_trades}
#     pass
