import asyncio
from app.database.postgres import AsyncSessionLocal
from app.ingestion.portfolio_sync import sync_portfolio_knowledge_base

async def main():
    async with AsyncSessionLocal() as session:
        print('Starting portfolio ingestion...')
        result = await sync_portfolio_knowledge_base(session, force_reingest=True)
        await session.commit()
        print('Ingestion Complete!')
        print('Result:', result)

if __name__ == '__main__':
    asyncio.run(main())
