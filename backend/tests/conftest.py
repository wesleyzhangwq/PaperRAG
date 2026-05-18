"""Global test fixtures — set env vars before any app imports."""
import os

os.environ.setdefault("MYSQL_HOST", "localhost")
os.environ.setdefault("MYSQL_DATABASE", "paperrag_test")
