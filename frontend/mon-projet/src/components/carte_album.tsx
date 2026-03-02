import { useState } from "react"
import Coeur from "./Coeur"
import GeneratedCover from "./GeneratedCover"

type CarteAlbumProps = {
  title: string
  artist: string
  pochette?: string
  isConnected: boolean
}

function CarteAlbum({ title, artist, isConnected }: CarteAlbumProps) {
  const [isFavorite, setIsFavorite] = useState(false)

  const toggleFavorite = () => {
    setIsFavorite((prev) => !prev)
  }

  return (
    <div className="carte-album" id="carte-album">
      <div className="pochette-wrapper">
        
          <GeneratedCover title={title} />
        

        <Coeur
          isFavorite={isFavorite}
          isConnected={isConnected}
          toggleFavorite={toggleFavorite}
        />
      </div>

      <article className="description">
        <div>
          <h3>{title}</h3>
          <p>{artist}</p>
        </div>

        <button
          className="btn-plus"
          onClick={() => console.log("Ajouter à la playlist")}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="30"
            height="30"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth="2.75"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M5 12h14" />
            <path d="M12 5v14" />
          </svg>
        </button>
      </article>
    </div>
  )
}

export default CarteAlbum