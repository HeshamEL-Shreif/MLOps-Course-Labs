from python
# Set work directory
WORKDIR /app
# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy app code
COPY . .
# Expose port
EXPOSE 8000
# Run the app
CMD ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"]