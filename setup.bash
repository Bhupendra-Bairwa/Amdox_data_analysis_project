# Setup script (setup.sh)
#!/bin/bash

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Initialize database
docker-compose up -d postgres redis

# Wait for database to be ready
sleep 10

# Run database migrations
python -c "from sqlalchemy import create_engine; engine = create_engine('postgresql://user:password@localhost:5432/crypto_analytics'); print('Database ready')"

# Start the dashboard
streamlit run src/dashboard/app.py --server.port=8501

# In another terminal, start live model trainer
python -c "import asyncio; from src.models.live_trainer import LiveModelTrainer; trainer = LiveModelTrainer(); asyncio.run(trainer.stream_websocket_data('btcusdt'))"