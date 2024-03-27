from sqlalchemy import Table, Column, Integer, String, MetaData, JSON

metadata = MetaData()

result = Table(
    "result",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("hash", String, nullable=False),
    Column("id_file", Integer, ),
    Column("result", JSON, nullable=False)
)

file = Table(
    "file",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("path", String, nullable=False)
)