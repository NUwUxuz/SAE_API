from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from models import Album, Track, User, Playlist

import uvicorn
import os

app = FastAPI()

# Configuration BDD
db_host = os.getenv("DB_HOST", "localhost")
DATABASE_URL = "postgresql://postgres:@localhost:5432/postgres"

engine = create_engine(DATABASE_URL)
       
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
@app.get("/album") 
def get_all_albums(db: Session = Depends(get_db)):
    return db.query(Album).all()

@app.get("/track") 
def get_all_track(db: Session = Depends(get_db)):
    return db.query(Track).all()

@app.get("/playlist") 
def get_all_track(db: Session = Depends(get_db)):
    return db.query(Playlist).all()


@app.get("/user/{email}")
def get_user_by_email(email: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.user_email == email).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    
    return user

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
