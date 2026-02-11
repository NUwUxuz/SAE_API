import { useState } from "react"
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
// import Contact from "./contact"
import MentionsLegales from "./mentions_legales"


import type { Page } from "./types/Page"

// type Page =
//   | "accueil"
//   | "detail_compte"
//   | "page_installation"
//   | "login"
//   | "register"
//   | "CGU"
//   | "contact"
//   | "mentions_legales"

function App(): JSX.Element {
  const [page, setPage] = useState<Page>("accueil")

  // 🔐 état de connexion
  const [isConnected, setIsConnected] = useState<boolean>(false)

  return (
    <>
      <Header onNavigate={setPage} isConnected={isConnected} />

      {page === "accueil" && <Accueil isConnected= {isConnected} />}

      {page === "detail_compte" && (
        isConnected ? <DetailCompte /> : setPage("login")
      )}

      {page === "page_installation" && <PageInstallation />}

      {page === "login" && (
        <Login
          onLogin={() => {
            setIsConnected(true)
            setPage("accueil")
          }}
          onRegister={() => setPage("register")}
        />
      )}

      {page === "register" && (
        <Register
          onRegister={() => {
            setIsConnected(true)
            setPage("accueil")
          }}
          onCancel={() => setPage("login")}
        />
      )}

      {page === "CGU" && <CGU />}

      {/* {page === "contact" && <Contact />} */}

      {page === "mentions_legales" && <MentionsLegales />}

      <Footer onNavigate={setPage} />
    </>
  )
}

export default App
