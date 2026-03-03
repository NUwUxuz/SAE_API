import { useState, useEffect } from "react"
import Coeur from "./Coeur"
import GeneratedCover from "./GeneratedCover"

// --- TYPES BASÉS SUR TON SCHÉMA SQLALCHEMY ---
type TrackData = {
    track_id: number
    track_title: string
    track_duration: number // Mapped[Optional[float]]
    track_composer?: string | null
    artist_name?: string | null // Issu de ViewTrackMaterialise
}

type PlaylistData = {
    playlist_id: number
    playlist_name: string
    playlist_listens: number
    user_id: number
    creator_pseudo?: string
}

type PlaylistDetailProps = {
    playlistId: number
    isConnected: boolean
}

// ============================================================================
// SOUS-COMPOSANT : Ligne de musique
// ============================================================================
function LigneMusique({
    track,
    index,
    isConnected,
}: {
    track: TrackData
    index: number
    isConnected: boolean
}) {
    const [isFavorite, setIsFavorite] = useState(false)
    const [isHovered, setIsHovered] = useState(false)

    const toggleFavorite = async () => {
        if (!isConnected) return
        const token = localStorage.getItem("token")
        if (!token) return

        try {
            if (!isFavorite) {
                const res = await fetch("http://127.0.0.1:8000/trackUserFavorite", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        Authorization: `Bearer ${token}`,
                    },
                    body: JSON.stringify({ track_id: track.track_id }), // MAJ: track_id
                })
                if (res.ok) setIsFavorite(true)
            } else {
                const res = await fetch(`http://127.0.0.1:8000/trackUserFavorite/${track.track_id}`, {
                    method: "DELETE",
                    headers: { Authorization: `Bearer ${token}` },
                })
                if (res.ok) setIsFavorite(false)
            }
        } catch (e) {
            console.error("Erreur favori:", e)
        }
    }

    // Formatage du float (ex: 205.5s -> 3:25)
    const formatDuration = (totalSeconds?: number) => {
        if (!totalSeconds) return "-:--"
        const minutes = Math.floor(totalSeconds / 60)
        const seconds = Math.floor(totalSeconds % 60)
        return `${minutes}:${seconds.toString().padStart(2, "0")}`
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
                    {/* On privilégie artist_name de la vue, sinon track_composer de la table track */}
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

// ============================================================================
// COMPOSANT PRINCIPAL
// ============================================================================
export default function PlaylistDetail({ playlistId, isConnected }: PlaylistDetailProps) {
    const [playlist, setPlaylist] = useState<PlaylistData | null>(null)
    const [tracks, setTracks] = useState<TrackData[]>([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        async function loadData() {
            try {
                const token = localStorage.getItem("token")
                const headers = token ? { Authorization: `Bearer ${token}` } : {}

                const resP = await fetch(`http://127.0.0.1:8000/playlists/${playlistId}`, { headers })
                if (resP.ok) setPlaylist(await resP.json())

                const resT = await fetch(`http://127.0.0.1:8000/playlists/${playlistId}/tracks`, { headers })
                if (resT.ok) setTracks(await resT.json())
            } catch (err) {
                console.error(err)
            } finally {
                setLoading(false)
            }
        }
        loadData()
    }, [playlistId])

    if (loading) return <div className="playlist-detail-container">Chargement...</div>

    return (
        <div className="playlist-detail-container">
            <header className="playlist-header">
                <div className="playlist-cover-large">
                    <GeneratedCover title={playlist?.playlist_name || "Playlist"} />
                </div>
                <div className="playlist-info">
                    <span className="playlist-type">PLAYLIST</span>
                    <h1 className="playlist-title-huge">{playlist?.playlist_name}</h1>
                    <div className="playlist-meta-info">
                        <span className="creator-bold">{playlist?.creator_pseudo || "Utilisateur"}</span>
                        <span className="bullet">•</span>
                        <span>{tracks.length} titres</span>
                        <span className="bullet">•</span>
                        <span>{playlist?.playlist_listens} écoutes</span>
                    </div>
                </div>
            </header>

            <div className="playlist-action-bar">
                <button className="btn-play-large" title="Lire la playlist">
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
                    <div className="col-title">TITRE</div>
                    <div className="col-actions"></div>
                    <div className="col-duration">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
                            <circle cx="12" cy="12" r="10"></circle>
                            <polyline points="12 6 12 12 16 14"></polyline>
                        </svg>
                    </div>
                </div>

                <div className="tracklist-body">
                    {tracks.map((t, i) => (
                        <LigneMusique key={t.track_id} track={t} index={i} isConnected={isConnected} />
                    ))}
                </div>
            </div>
        </div>
    )
}