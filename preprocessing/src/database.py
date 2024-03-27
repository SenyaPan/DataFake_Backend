import datetime
from typing import AsyncGenerator

from fastapi import Depends
from fastapi_users.db import SQLAlchemyUserDatabase, SQLAlchemyBaseUserTable
from sqlalchemy import String, Boolean, Column, Integer, ForeignKey, TIMESTAMP
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from preprocessing.src.config import DB_HOST, DB_NAME, DB_PASS, DB_USER, DB_PORT

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Base: DeclarativeBase = declarative_base()


class Base(DeclarativeBase):
    pass


engine = create_async_engine(DATABASE_URL)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

# убираем так как таблицы созданы заранее и мы не создаем их снова при запуске приложения
# async def create_db_and_tables():
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session

# Depends передает параметры, необходимые для вызова функции, из той фукнции, которая в скобочках,
# в ту, в которой используется

