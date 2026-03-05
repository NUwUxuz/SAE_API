import { useState, useEffect } from "react"
import Coeur from "./coeur"

import GeneratedCover from "./GeneratedCover"

type TrackData = {
    track_id: number
    track_title: string
    track_duration: number
    artist_name?: string
    track_composer?: string
}

type AlbumData = {
    album_id: number
    album_title: string
    artist_name: string
    album_image_file?: string
    album_listens: number
    track_count: number
    album_date_released?: string
}

type AlbumDetailProps = {
    albumId: number
    isConnected: boolean
}

function AlbumTrackRow({ track, index, isConnected }: { track: TrackData, index: number, isConnected: boolean }) {
    const [isHovered, setIsHovered] = useState(false)
    const [isFavorite, setIsFavorite] = useState(false)

    const toggleFavorite = () => {
        setIsFavorite(!isFavorite)
    }


    const formatDuration = (seconds: number) => {
        const mins = Math.floor(seconds / 60)
        const secs = Math.floor(seconds % 60)
        return `${mins}:${secs.toString().padStart(2, "0")}`
    }

    return (
        <div
            className="track-row"
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            <div className="col-index">
                {isHovered ? (
                    <svg viewBox="0 0 24 24" fill="currentColor" width="16" height="16" style={{ color: "var(--color-text)" }}>
                        <path d="M8 5v14l11-7z" />
                    </svg>
                ) : (
                    index + 1
                )}
            </div>

            <div className="col-title">
                <span className="track-name" style={{ color: isHovered ? "var(--color-primary)" : "var(--color-text)" }}>
                    {track.track_title}
                </span>
                <span className="track-artist">
                    {track.artist_name || track.track_composer || "Artiste inconnu"}
                </span>
            </div>

            <div className="col-actions">
                {(isHovered || isFavorite) && (
                    <Coeur
                        isFavorite={isFavorite}
                        isConnected={isConnected}
                        toggleFavorite={toggleFavorite}
                    />
                )}
            </div>

            <div className="col-duration">{formatDuration(track.track_duration)}</div>
        </div>
    )
}

export default function AlbumDetail({ albumId, isConnected }: AlbumDetailProps) {
    const [album, setAlbum] = useState<AlbumData | null>(null)
    const [tracks, setTracks] = useState<TrackData[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        async function loadData() {
            setLoading(true)
            try {
                const resA = await fetch(`http://127.0.0.1:8000/album/${albumId}`)
                if (resA.ok) setAlbum(await resA.json())

                const resT = await fetch(`http://127.0.0.1:8000/album/${albumId}/tracks`)
                if (resT.ok) setTracks(await resT.json())

            } catch (err) {
                console.error(err)
            } finally {
                setLoading(false)
            }
        }
        loadData()
    }, [albumId])

    if (loading) return <div className="playlist-detail-container">Chargement...</div>

    return (
        <div className="playlist-detail-container">
            <header className="playlist-header" style={{ background: "linear-gradient(to bottom, rgba(59, 130, 246, 0.15) 0%, var(--color-bg) 100%)" }}>
                <div className="playlist-cover-large">
                    <GeneratedCover title={album?.album_title || "Album"} />
                </div>

                <div className="playlist-info">
                    <span className="playlist-type">ALBUM</span>
                    <h1 className="playlist-title-huge">{album?.album_title}</h1>
                    <div className="playlist-meta-info">
                        <span className="creator-bold">{album?.artist_name || "Artiste"}</span>
                        <span className="bullet">•</span>
                        <span>{album?.album_date_released ? new Date(album.album_date_released).getFullYear() : "Année inconnue"}</span>
                        <span className="bullet">•</span>
                        <span>{tracks.length} titres</span>
                        <span className="bullet">•</span>
                        <span>{album?.album_listens} écoutes</span>
                    </div>
                </div>
            </header>

            <div className="playlist-action-bar">
                <button className="btn-play-large" title="Écouter l'album">
                    <svg viewBox="0 0 24 24" fill="black" width="28" height="28" style={{ marginLeft: "4px" }}>
                        <path d="M8 5v14l11-7z" />
                    </svg>
                </button>

                <button className="btn-icon-secondary" title="Ajouter aux favoris">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="32" height="32">
                        <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
                    </svg>
                </button>

                <button className="btn-icon-secondary" title="Plus d'options">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="32" height="32">
                        <circle cx="12" cy="12" r="1"></circle>
                        <circle cx="19" cy="12" r="1"></circle>
                        <circle cx="5" cy="12" r="1"></circle>
                    </svg>
                </button>
            </div>

            <div className="tracklist-container">
                <div className="tracklist-header">
                    <div className="col-index">#</div>
                    <div className="col-title">Titre</div>
                    <div className="col-actions"></div>
                    <div className="col-duration">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                            <circle cx="12" cy="12" r="10"></circle>
                            <polyline points="12 6 12 12 16 14"></polyline>
                        </svg>
                    </div>
                </div>

                <div className="tracks-list">
                    {tracks.map((track, idx) => (
                        <AlbumTrackRow
                            key={track.track_id}
                            track={track}
                            index={idx}
                            isConnected={isConnected}
                        />
                    ))}
                </div>
            </div>
        </div>
    )
}
