from fastapi import FastAPI

app = FastAPI(title="Selene")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
