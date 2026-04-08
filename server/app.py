import uvicorn
from server.main import app

def main():
    uvicorn.run("server.main:app", host="0.0.0.0", port=7860, workers=1)

if __name__ == "__main__":
    main()

__all__ = ["app", "main"]
