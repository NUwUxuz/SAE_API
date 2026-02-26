const API_URL = "http://localhost:8000"

export type UserProfile = {
    user_id: number
    user_login: string
    pseudo: string
    email: string
    image?: string
}

// Typage pour l'inscription (à adapter selon tes colonnes DB)
export type RegisterData = {
    user_login: string
    mdp: string
    pseudo: string
    email: string
}

/**
 * Connecte l'utilisateur et initialise la session locale
 */
export async function login(username: string, password: string) {
    const formData = new FormData()
    formData.append("username", username)
    formData.append("password", password)

    const response = await fetch(`${API_URL}/login`, {
        method: "POST",
        body: formData,
    })

    if (!response.ok) {
        // En cas d'erreur de login, on s'assure que rien ne reste en local
        logout() 
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || "Échec de la connexion")
    }

    const data = await response.json()
    // 1. Stockage du token
    localStorage.setItem("token", data.access_token)

    try {
        // 2. Récupération du profil pour valider le token et avoir l'ID
        const userProfile = await getCurrentUser()
        localStorage.setItem("user_id", userProfile.user_id.toString())
        return { token: data.access_token, user: userProfile }
    } catch (err) {
        logout()
        throw new Error("Erreur lors de la récupération du profil après connexion")
    }
}

/**
 * Récupère les infos de l'utilisateur connecté.
 * Si le token est invalide/expiré, nettoie la session et rejette.
 */
export async function getCurrentUser(): Promise<UserProfile> {
    const token = localStorage.getItem("token")
    
    if (!token) {
        logout()
        throw new Error("Non authentifié")
    }

    try {
        const response = await fetch(`${API_URL}/user`, {
            method: "GET",
            headers: {
                "Authorization": `Bearer ${token}`
            }
        })

        // Si le serveur renvoie 401 (Unauthorized) ou 403 (Forbidden)
        if (response.status === 401 || response.status === 403) {
            logout()
            throw new Error("Session expirée")
        }

        if (!response.ok) {
            throw new Error("Erreur serveur")
        }

        return await response.json()
    } catch (error) {
        // En cas de coupure réseau ou erreur fatale, on considère la session compromise
        if (error instanceof Error && error.message === "Session expirée") {
            throw error
        }
        throw new Error("Impossible de joindre le serveur")
    }
}

/**
 * Inscrit un nouvel utilisateur
 */
export async function register(userData: RegisterData) {
    const response = await fetch(`${API_URL}/user`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(userData)
    })

    if (!response.ok) {
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || "Échec de l'inscription")
    }

    return response.json()
}

/**
 * Supprime proprement la session locale
 */
export function logout() {
    localStorage.removeItem("token")
    localStorage.removeItem("user_id")
    // On peut aussi vider d'autres données sensibles ici si nécessaire
}

/**
 * Helper : Vérifie si un token existe sans faire d'appel API
 */
export function isAuthenticated(): boolean {
    return !!localStorage.getItem("token")
}