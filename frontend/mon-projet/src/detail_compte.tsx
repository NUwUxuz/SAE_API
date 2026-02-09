import "./user.css"
import {useState} from "react"

import InfoUser from "./components/info_user"
import StatsUser from "./components/stats_user"

export default

function detail_compte() {
    const [showStats, setShowStats] = useState(false)

    return (
        <>
            <InfoUser
                email="adressemail@truc.com"
                id="ncxjfvdst"
                password="motdepass123"
            />
            {showStats && (
                <StatsUser 
                    favoris={35}
                    playlists={12}
                    track_listen={[{user_id:1, track_id:6, nb_listening:27}]}
                    genre_listen={[{user_id:1, genre_id:3, genre_rate:0.24}]}
                />
            )}

            <button id="bouton-user" className="bouton-cliquable" onClick={() => setShowStats(!showStats)}>{showStats ? "Cacher stats utilisateurs" : "Afficher stats utilisateurs"}</button>

            <div id="bouton-footer">
                <button id="deconnecter" className="bouton-cliquable">Se déconnecter</button>
                <button id="supprimer" className="bouton-cliquable">Supprimer le compte</button>
            </div>
        </>
    );
}
