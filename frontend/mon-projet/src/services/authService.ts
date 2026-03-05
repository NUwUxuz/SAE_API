const API_URL = "http://localhost:8000"

export type UserProfile = {
    user_id: number
    user_login: string
    pseudo: string
    email: string
    image?: string
    // Ajout des rôles dans le profil pour plus de clarté
    roles?: string[]
}

export type RegisterData = {
    user_login: string
    user_mdp: string // Note: vérifie si c'est 'mdp' ou 'user_mdp' dans ton schéma
    pseudo: string
    email: string
}

/**
 * Connecte l'utilisateur, stocke le token et ses rôles
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
        logout() 
        const error = await response.json().catch(() => ({}))
        throw new Error(error.detail || "Échec de la connexion")
    }

    const data = await response.json()
    
    // 1. Stockage du token et des rôles renvoyés par ton nouveau backend
    localStorage.setItem("token", data.access_token)
    // On stocke les rôles sous forme de chaîne JSON
    localStorage.setItem("user_roles", JSON.stringify(data.roles || []))

    try {
        const userProfile = await getCurrentUser()
        localStorage.setItem("user_id", userProfile.user_id.toString())
        return { token: data.access_token, user: userProfile, roles: data.roles }
    } catch (err) {
        logout()
        throw new Error("Erreur lors de la récupération du profil")
    }
}

/**
 * Récupère les infos de l'utilisateur connecté
 */
export async function getCurrentUser(): Promise<UserProfile> {
    const token = localStorage.getItem("token")
    
    if (!token) {
        logout()
        throw new Error("Non authentifié")
    }

    const response = await fetch(`${API_URL}/user`, {
        method: "GET",
        headers: {
            "Authorization": `Bearer ${token}`
        }
    })

    if (response.status === 401 || response.status === 403) {
        logout()
        throw new Error("Session expirée")
    }

    if (!response.ok) throw new Error("Erreur serveur")

    return await response.json()
}

/**
 * Supprime proprement la session
 */
export function logout() {
    localStorage.removeItem("token")
    localStorage.removeItem("user_id")
    localStorage.removeItem("user_roles")
}

/**
 * NOUVEAU : Vérifie si l'utilisateur possède un rôle spécifique
 * Utile pour : { hasRole('ADMIN') && <button>Supprimer tout</button> }
 */
export function hasRole(roleName: string): boolean {
    const rolesRaw = localStorage.getItem("user_roles")
    if (!rolesRaw) return false
    
    try {
        const roles: string[] = JSON.parse(rolesRaw)
        return roles.includes(roleName) || roles.includes("ADMIN")
    } catch {
        return false
    }
}

/**
 * Helper pour les appels API (ajoute le header Auth automatiquement)
 */
export function getAuthHeader() {
    const token = localStorage.getItem("token")
    return token ? { "Authorization": `Bearer ${token}` } : {}
}

export function isAuthenticated(): boolean {
    return !!localStorage.getItem("token")
}

/**
 * Inscrit un nouvel utilisateur
 * @param userData Objet contenant user_login, user_mdp, pseudo, email
 */
export async function register(userData: RegisterData) {
    const response = await fetch(`${API_URL}/user`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(userData)
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || "Échec de l'inscription");
    }

    return await response.json();
}