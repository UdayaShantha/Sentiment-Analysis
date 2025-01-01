import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',

    handlers=[
        logging.FileHandler('app.log',mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)