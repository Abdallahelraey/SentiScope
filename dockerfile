FROM python:3.11-slim

WORKDIR /app

COPY requirements_prod.txt .
COPY setup.py ./
COPY README.md ./
COPY src/ ./src/   


RUN pip install -r requirements_prod.txt

# Add specific package
RUN pip install python-box==7.3.0
COPY . .


EXPOSE 5000 8080

CMD ["./start.sh"]