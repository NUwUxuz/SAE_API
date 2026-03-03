import { useState, useEffect } from "react"
// import "./App.css"
import "./index.css"

import Header from "./components/Header"
import Footer from "./components/Footer"

import Accueil from "./accueil"
import DetailCompte from "./detail_compte"
import PageInstallation from "./installation"
import Login from "./login"
import Register from "./register"
import CGU from "./CGU"
import MentionsLegales from "./mentions_legales"
import PlaylistDetail from "./components/PlaylistDetail"

import { getCurrentUser, logout } from "./services/authService"
import type { Page } from "./types/Page"

function App() {
  const [page, setPage] = useState<Page>("accueil")

  // pour mémoriser quelle playlist on veut afficher
  const [selectedPlaylistId, setSelectedPlaylistId] = useState<number | null>(null)

  // 🔐 État de connexion
  const [isConnected, setIsConnected] = useState<boolean>(false)
  const [userId, setUserId] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(true)

  // 🚪 Fonction de déconnexion centralisée avec redirection
  const handleLogout = () => {
    logout() // Supprime le localStorage via authService
    setIsConnected(false)
    setUserId(null)
    setPage("login") // Redirige vers login dès que la session est perdue
  }

  // 🚀 Gestion du succès de connexion
  const handleLoginSuccess = () => {
    setIsConnected(true)
    const storedId = localStorage.getItem("user_id")
    if (storedId) setUserId(parseInt(storedId))
    setPage("accueil")
  }

  // 🛡️ Vérification de la session (au chargement et périodiquement)
  useEffect(() => {
    const verifyAuth = async () => {
      const token = localStorage.getItem("token")

      if (!token) {
        if (isConnected) setIsConnected(false)
        setIsLoading(false)
        return
      }

      try {
        // On vérifie si le token est toujours valide auprès du serveur
        const user = await getCurrentUser()
        setIsConnected(true)
        setUserId(user.user_id)
      } catch (error) {
        // Si le token est expiré ou invalide (Erreur 401)
        handleLogout()
      } finally {
        setIsLoading(false)
      }
    }

    verifyAuth()

    // Vérifie la validité toutes les 30 secondes pour rediriger automatiquement si expiration
    const interval = setInterval(verifyAuth, 30000)
    return () => clearInterval(interval)
  }, [isConnected])

  const handleOpenPlaylist = (id: number) => {
    setSelectedPlaylistId(id)
    setPage("playlist_detail")
  }

  // 📝 Gestionnaire de rendu pour inclure la redirection forcée
  const renderContent = () => {
    if (isLoading) return <div className="loading">Vérification de la session...</div>

    // Protection : Si l'utilisateur tente d'accéder à "detail_compte" sans session
    if (page === "detail_compte" && !isConnected) {
      return (
        <Login
          onLogin={handleLoginSuccess}
          onRegister={() => setPage("register")}
        />
      )
    }

    switch (page) {
      case "accueil":
        return (
          <Accueil
            isConnected={isConnected}
            userId={userId}
            onOpenPlaylist={handleOpenPlaylist}
          />
        )

      case "playlist_detail":
        return (
          <PlaylistDetail
            playlistId={selectedPlaylistId!}
            isConnected={isConnected}
          />
        )

      case "detail_compte":
        return <DetailCompte />

      case "page_installation":
        return <PageInstallation />

      case "login":
        return (
          <Login
            onLogin={handleLoginSuccess}
            onRegister={() => setPage("register")}
          />
        )

      case "register":
        return <Register onNavigate={setPage} />

      case "CGU":
        return <CGU />

      case "mentions_legales":
        return <MentionsLegales />

      default:
        return (
          <Accueil
            isConnected={isConnected}
            userId={userId}
            onOpenPlaylist={handleOpenPlaylist}
          />
        )
    }
  }

  return (
    <>
      <Header
        onNavigate={setPage}
        isConnected={isConnected}
        onLogout={handleLogout}
      />

      <main>
        {renderContent()}
      </main>

      <Footer onNavigate={setPage} />
    </>
  )
}

export default App
