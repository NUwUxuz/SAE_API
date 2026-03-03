import { useState, useEffect } from "react"
import Carousel from "./components/Carousel"
import CarteChanson from "./components/carte_chanson"
import CartePlaylist from "./components/carte_playlist"
import CarteAlbum from "./components/carte_album"
import AddToPlaylistModal from "./components/AddToPlaylistModal" // Import du modal
import { getChansons } from "./services/chansonService"
import type { Playlist } from "./types/Playlist"
import type { Album } from "./types/Album"
import viteLogo from "/vite.svg"

type AccueilProps = {
  isConnected: boolean
  userId: number | null
  onOpenPlaylist: (id: number) => void
  onOpenAlbum: (id: number) => void
}


interface Track {
  track_id: number;
  track_title: string;
  artist_name: string;
  album_image_file: string;
}

interface PlaylistDB {
  playlist_id: number;
  playlist_name: string;
  playlist_listens: number;
  user_id: number;
}

export default function Accueil({ isConnected = false, userId, onOpenPlaylist, onOpenAlbum }: AccueilProps) {


  const [tracks, setTracks] = useState<Track[]>([]);
  const [recoGRU, setRecoGRU] = useState<Track[]>([]);
  const [recoTF_IDF, setRecoTF_IDF] = useState<Track[]>([]);
  const [userPlaylists, setUserPlaylists] = useState<PlaylistDB[]>([]);

  const [loadingGeneral, setLoadingGeneral] = useState(true);
  const [loadingGRU, setLoadingGRU] = useState(false);
  const [loadingTF_IDF, setLoadingTF_IDF] = useState(false);

  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadGeneralTracks() {
      setError(null);
      try {
        const res = await fetch("http://127.0.0.1:8000/viewTrack?limit=100");
        const data = await res.json();
        setTracks(data);
      } catch (e) { console.error(e); }
      finally { setLoadingGeneral(false); }
    }

    async function loadGRU() {
      if (!isConnected) return;

      setLoadingGRU(true);
      try {
        const token = localStorage.getItem("token");

        if (token && isConnected) {
          const res = await fetch("http://127.0.0.1:8000/users/gru_recommendations/detailed?limit=10", {
            method: "GET",
            headers: {
              "Authorization": `Bearer ${token}`, // Envoi du badge d'accès
              "Content-Type": "application/json"
            }
          });

          if (res.ok) {
            const data = await res.json();
            setRecoGRU(data);
          }
        }
      } catch (e) { console.error(e); }
      finally { setLoadingGRU(false); }
    }

    async function loadTF_IDF() {
      if (!isConnected) return;

      setLoadingTF_IDF(true);
      try {
        const token = localStorage.getItem("token");

        if (token && isConnected) {
          const res = await fetch("http://127.0.0.1:8000/users/tf-idf_recommendations?limit=10", {
            method: "GET",
            headers: {
              "Authorization": `Bearer ${token}`, // Envoi du badge d'accès
              "Content-Type": "application/json"
            }
          });

          if (res.ok) {
            const data = await res.json();
            setRecoTF_IDF(data);
          }
        }
      } catch (e) { console.error(e); }
      finally { setLoadingTF_IDF(false); }
    }

    async function loadUserPlaylists() {
      if (!isConnected || !userId) return;
      try {
        const token = localStorage.getItem("token");
        if (token) {
          const res = await fetch(`http://127.0.0.1:8000/users/${userId}/playlists`, {
            method: "GET",
            headers: { "Authorization": `Bearer ${token}` }
          });
          if (res.ok) {
            setUserPlaylists(await res.json());
          }
        }
      } catch (e) {
        console.error("Erreur lors de la récupération des playlists :", e);
      }
    }

    loadGeneralTracks();
    loadGRU();
    loadTF_IDF();
    loadUserPlaylists();
  }, [isConnected, userId]);


  const [modalOpen, setModalOpen] = useState(false)
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null)
  const playlists: Partial<Playlist>[] = Array.from({ length: 50 }, () => ({
    title: "Top Disney",
    creator: "Orchestra",
    pochette: viteLogo,
  }))

  const albums: Album[] = Array.from({ length: 5 }, () => ({
    title: "Frozen",
    artist: "Idina Menzel",
    pochette: viteLogo,
  }))

  /* 🔎 Filtres */
  const playlistsFiltrees = playlists


  const albumsFiltres = albums


  const handleAddTrack = (trackId: number) => {
    setSelectedTrackId(trackId)
    setModalOpen(true)
  }

  return (
    <>

      <div className="accueil-layout">
        <nav className="menu-favoris">
          <ul className="list-aime">
            <li>Écouté récemment</li>
            <li>Titres aimés</li>
            <li>Albums</li>
            <li>Artistes</li>
          </ul>

          <button
            className="btn-add-playlist"
            onClick={() => {
              setSelectedTrackId(null)
              setModalOpen(true)
            }}
          >
            Ajouter une Playlist
          </button>

          {isConnected && (
            <ul className="list-playlist">
              {userPlaylists.map((pl) => (
                <li
                  key={pl.playlist_id}
                  className="playlist-menu-item"
                  style={{ cursor: "pointer" }}
                  onClick={() => onOpenPlaylist(pl.playlist_id)}
                >
                  {pl.playlist_name}
                </li>
              ))}
            </ul>
          )}
        </nav>
        <main className="accueil-content">


          <h2>Musiques recommandées</h2>
          {error ? (
            <div style={{ color: 'red', textAlign: 'center', margin: '20px 0' }}>
              <p>⚠️ {error}</p>
            </div>
          ) : loadingGeneral ? (
            <p>Chargement des musiques...</p>
          ) : (
            <Carousel>
              {tracks.map((track) => (
                <CarteChanson
                  key={track.track_id}
                  trackId={track.track_id}
                  title={track.track_title}
                  artist={track.artist_name}
                  // artist={track.artists.map(a => a.artist_name).join(", ")}
                  pochette={track.album_image_file}
                  isConnected={isConnected}
                  onAdd={() => handleAddTrack(track.track_id)}
                />
              ))}
            </Carousel>
          )}

          {isConnected && (
            <div className="reco-section">
              <h2>Selon vos recherches</h2>

              {loadingGRU ? (
                <div>
                  <p>Chargement des musiques...</p>
                </div>
              ) : (
                <Carousel>
                  {recoGRU.map((track) => (
                    <CarteChanson
                      key={track.track_id}
                      trackId={track.track_id}
                      title={track.track_title}
                      artist={track.artist_name}
                      // artist={track.artists.map(a => a.artist_name).join(", ")}
                      pochette={track.album_image_file}
                      isConnected={isConnected}
                      onAdd={() => handleAddTrack(track.track_id)}
                    />
                  ))}
                </Carousel>
              )}

              <h2>Selon vos préférences</h2>

              {loadingTF_IDF ? (
                <div>
                  <p>Chargement des musiques...</p>
                </div>
              ) : (
                <Carousel>
                  {recoTF_IDF.map((track) => (
                    <CarteChanson
                      key={`reco-${track.track_id}`}
                      trackId={track.track_id}
                      title={track.track_title}
                      artist={track.artist_name}
                      pochette={track.album_image_file}
                      isConnected={isConnected}
                      onAdd={() => handleAddTrack(track.track_id)}
                    />
                  ))}
                </Carousel>
              )}
            </div>
          )}

          <h2>Playlists recommandées</h2>
          <Carousel>
            {playlistsFiltrees.map((playlist, index) => (
              <CartePlaylist
                key={index}
                title={playlist.title || playlist.playlist_name || "Sans titre"}
                creator={playlist.creator || "Inconnu"}
                pochette={playlist.pochette || viteLogo}
                isConnected={isConnected}
              />
            ))}
          </Carousel>

          <h2>Albums recommandés</h2>
          <Carousel>
            {albumsFiltres.map((album, index) => (
              <CarteAlbum
                key={index}
                title={album.title}
                artist={album.artist}
                pochette={album.pochette}
                isConnected={isConnected}
                onClick={() => onOpenAlbum(index)} // Note: assuming index is ID or we need real IDs
              />

            ))}
          </Carousel>
        </main>
      </div>

      {/* Le Modal est rendu ici */}
      <AddToPlaylistModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        trackId={selectedTrackId}
        userId={userId}
      />
    </>
  )
}