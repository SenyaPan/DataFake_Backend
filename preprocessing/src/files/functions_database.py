from fastapi import APIRouter, Depends
from sqlalchemy import select, insert
from sqlalchemy.ext.asyncio import AsyncSession

from preprocessing.src.database import get_async_session
from preprocessing.src.operations.models import operation
from preprocessing.src.operations.schemas import OperationCreate

router = APIRouter(
    prefix="/operations",
    tags=["Operation"]
)


async def get_result_by_hash(file_hash: str, session: AsyncSession = Depends(get_async_session)):
    query = select(result).where(result.c.hash == file_hash)
    result = await session.execute(query)
    return result.mappings().all()


async def add_operation(new_operation: ResultsCreate, session: AsyncSession = Depends(get_async_session)):
    stmt = insert(operation).values(**new_operation.dict())
    await session.execute(stmt)
    await session.commit()
    return {"status": "success"}
