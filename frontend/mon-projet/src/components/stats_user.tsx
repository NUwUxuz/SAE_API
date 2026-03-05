import React from "react";
// Assure-toi que les chemins vers tes assets sont corrects
import coeur from "../assets/coeur.png";
import playlistIcon from "../assets/playlist.png";
import "./stats_user.css"

type Track = {
    user_id: number;
    track_id: number;
    nb_listening: number;
};

type Genre = {
    user_id: number;
    genre_id: number;
    genre_rate: number;
};

type StatsUserProps = {
    favoris: number;
    playlists: number;
    track_listen: Track[];
    genre_listen: Genre[];
};

function StatsUser({ favoris, playlists, track_listen, genre_listen }: StatsUserProps) {
    // Calcul du maximum pour les barres de progression
    const maxListens = Math.max(...track_listen.map(t => t.nb_listening), 1);

    return (
        <section className="stats-section">
            <div className="stats-grid-top">
                <div className="stat-card-mini">
                    <img src={coeur} className="stat-icon" alt="Favoris" />
                    <div className="stat-info">
                        <span className="stat-label">Favoris</span>
                        <p className="stat-value">{favoris}</p>
                    </div>
                </div>
                
                <div className="stat-card-mini">
                    <img src={playlistIcon} className="stat-icon" alt="Playlists" />
                    <div className="stat-info">
                        <span className="stat-label">Playlists</span>
                        <p className="stat-value">{playlists}</p>
                    </div>
                </div>
            </div>

            <div className="stats-grid-bottom">
                <div className="stat-card-large">
                    <h3 className="stat-title">Titres les plus écoutés</h3>
                    <ul className="visual-list">
                        {track_listen
                            .sort((a, b) => b.nb_listening - a.nb_listening)
                            .slice(0, 5) // On prend les 5 meilleurs
                            .map((track) => (
                                <li key={track.track_id} className="visual-item">
                                    <div className="item-text">
                                        <span>Track #{track.track_id}</span>
                                        <span className="item-count">{track.nb_listening} écoutes</span>
                                    </div>
                                    <div className="progress-bar">
                                        <div 
                                            className="progress-fill track-fill" 
                                            style={{ width: `${(track.nb_listening / maxListens) * 100}%` }}
                                        ></div>
                                    </div>
                                </li>
                            ))
                        }
                    </ul>
                </div>

                <div className="stat-card-large">
                    <h3 className="stat-title">Top Genres</h3>
                    <ul className="visual-list">
                        {genre_listen
                            .sort((a, b) => b.genre_rate - a.genre_rate)
                            .map((genre) => (
                                <li key={genre.genre_id} className="visual-item">
                                    <div className="item-text">
                                        <span>Genre {genre.genre_id}</span>
                                        <span className="item-count">{(genre.genre_rate * 100).toFixed(0)}%</span>
                                    </div>
                                    <div className="progress-bar">
                                        <div 
                                            className="progress-fill genre-fill" 
                                            style={{ width: `${genre.genre_rate * 100}%` }}
                                        ></div>
                                    </div>
                                </li>
                            ))
                        }
                    </ul>
                </div>
            </div>
        </section>
    );
}

export default StatsUser;